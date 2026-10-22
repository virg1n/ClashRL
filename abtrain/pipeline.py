"""One-command supervisor for the whole A/B pipeline (started by ``run_ab.sh``).

    python -m abtrain.pipeline run      # supervise; resumes from WORK/state.json
    python -m abtrain.pipeline status   # print WORK/STATUS.md (refreshed every 2 minutes)
    python -m abtrain.pipeline stop     # ask a running supervisor to stop everything cleanly

Lanes run as threads; every stage is a subprocess with retries, hang detection (no log or
heartbeat activity for ``hang_minutes``) and its own log in ``LOGS/<stage>.out``. A stage that
finished is never redone; a rerun after a crash or a stopped Jupyter session resumes inside
each stage from its latest checkpoint.

  main  download -> extract -> stage shards to /dev/shm -> stage 1 BC (all GPUs) -> calibration
        -> V_ref (all GPUs) -> stage 2 for A (all GPUs) -> keep A's stage 2 -> A/B test
        -> split: lane A (PPO on GPUs 0-1) and lane B (stage 2 for B -> A/B test -> PPO on GPUs 2-3)
  fl    after extraction: record FirstLight games -> after stage 1: train the General and Hog 2
        stand-ins -> check each against the real model -> into the league
  eval  after extraction: FL Hog 2 vs General baseline -> screen stage 1/2 and every PPO
        snapshot against FL General, promote the best (``abtrain.evaluator``)
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path

from .common import (CKPT, DATA, FL_REPO, FL_SHARDS, LEAGUE, LOGS, RUNS, SHARDS, SHM, STATE, WORK, atomic_write_json,
                     ensure_dirs, log, read_json, settings, torch_load, torch_save_atomic)

PY = sys.executable
NAME = "pipeline"
STOP_FILE = WORK / "STOP"
PID_FILE = WORK / "supervisor.json"
PROCS = WORK / "procs"  # one file per running stage process group, to reap after a supervisor crash
STATUS = WORK / "STATUS.md"
FL_COMMIT = "3ade158"


# ============================================================================ hardware


def detect_gpus() -> list:
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if vis is not None:  # respect a restriction set by the session (ids stay as given)
        return [v.strip() for v in vis.split(",") if v.strip() and v.strip() != "-1"]
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=60).stdout
        n = sum(1 for line in out.splitlines() if line.startswith("GPU "))
        if n:
            return list(range(n))
    except Exception:
        pass
    try:
        import torch

        return list(range(torch.cuda.device_count()))
    except Exception:
        return []


def detect_cores() -> list[int]:
    try:
        return sorted(os.sched_getaffinity(0))
    except AttributeError:
        return list(range(os.cpu_count() or 4))


class Layout:
    """GPU and core split. Shared stages use every GPU; after the split A and B each get half."""

    def __init__(self):
        self.gpus = detect_gpus()
        cores = detect_cores()
        n = len(cores)
        g = self.gpus
        if len(g) >= 4:
            self.a_gpus, self.b_gpus = g[:2], g[2:4]
        elif len(g) >= 2:
            self.a_gpus, self.b_gpus = g[:1], g[1:2]
        else:
            self.a_gpus = self.b_gpus = g[:1]
        n_eval = max(1, min(6, n // 8))
        n_ctl = 2 if n >= 16 else 0
        per_run = max(1, (n - n_eval - n_ctl) // 2)
        self.a_cores = cores[:per_run]
        self.b_cores = cores[per_run : 2 * per_run]
        self.eval_cores = cores[2 * per_run : 2 * per_run + n_eval] or cores[-1:]
        self.n_cores = n

    def describe(self) -> str:
        return (f"GPUs {self.gpus} (A {self.a_gpus}, B {self.b_gpus}); {self.n_cores} cores "
                f"(A {_span(self.a_cores)}, B {_span(self.b_cores)}, eval {_span(self.eval_cores)})")


def _span(c: list[int]) -> str:
    return f"{c[0]}-{c[-1]}" if c else "-"


def _csv(xs) -> str:
    return ",".join(str(x) for x in xs)


# ============================================================================ supervisor


class Supervisor:
    def __init__(self):
        ensure_dirs()
        self.cfg = settings()
        self.lock = threading.Lock()
        self.fl_lock = threading.Lock()
        self.stop = threading.Event()  # requested by the user (exit 3)
        self.cancel = threading.Event()  # the main lane gave up (exit 4: run_ab.sh retries later)
        self.procs: dict[str, subprocess.Popen] = {}
        self.state = read_json(STATE, {}) or {}
        self.state.setdefault("started_at", time.time())
        self.state.setdefault("stages", {})
        self.state.setdefault("runs", 0)
        self.state["runs"] += 1
        self.state["last_start"] = time.time()
        self.deadline = self.state["started_at"] + self.cfg["deadline_hours"] * 3600
        self.layout = Layout()
        self.errors: list[str] = []
        self.save_state()
        reap_stale()

    # ---------------------------------------------------------------- state
    def save_state(self):
        with self.lock:
            atomic_write_json(STATE, self.state)

    def stage_info(self, name: str) -> dict:
        with self.lock:
            return dict(self.state["stages"].get(name, {}))

    def set_stage(self, name: str, **kw):
        with self.lock:
            st = self.state["stages"].setdefault(name, {})
            st.update(kw, updated=time.time())
        self.save_state()

    def is_done(self, name: str) -> bool:
        return self.stage_info(name).get("status") == "done"

    def halted(self) -> bool:
        return self.stop.is_set() or self.cancel.is_set()

    def nap(self, seconds: float) -> None:
        t0 = time.time()
        while time.time() - t0 < seconds and not self.halted():
            time.sleep(5)

    def wait_for(self, name: str, poll: float = 30.0) -> bool:
        """Block until stage ``name`` is done (True) or failed / halted / past the deadline (False)."""
        while not self.halted():
            if self.past_deadline():
                return False
            st = self.stage_info(name).get("status")
            if st == "done":
                return True
            if st in ("failed", "skipped"):
                return False
            time.sleep(poll)
        return False

    def past_deadline(self, grace_h: float = 0.0) -> bool:
        return time.time() > self.deadline + grace_h * 3600

    # ---------------------------------------------------------------- stages
    def run(self, name: str, cmd: list[str], *, gpus: list[int] | None = None, done=None, retries: int | None = None,
            heartbeats: tuple = (), critical: bool = True, env: dict | None = None, deadline_grace_h: float = 0.25,
            hang_minutes: float | None = None, stop_grace: float = 180.0) -> bool:
        """Run a stage to completion with retries. ``done()`` confirms completion beyond exit code 0.
        ``stop_grace``: seconds the stage's own process gets to shut down before its group is killed."""
        done = done or (lambda: True)
        if self.is_done(name) and done():
            return True
        retries = self.cfg["stage_retries"] if retries is None else retries
        hang = (hang_minutes or self.cfg["hang_minutes"]) * 60
        out = LOGS / f"{name}.out"
        attempts = self.stage_info(name).get("attempts", 0)
        for attempt in range(retries + 1):
            if self.halted():
                return False
            if self.past_deadline(deadline_grace_h):
                log(f"{name}: deadline passed, not starting", NAME)
                self.set_stage(name, status="skipped", reason="deadline")
                return False
            attempts += 1
            self.set_stage(name, status="running", attempts=attempts, started=time.time())
            e = dict(os.environ)
            e.update(env or {})
            e["PYTHONUNBUFFERED"] = "1"
            token = uuid.uuid4().hex  # every process of this attempt inherits it (torchrun ranks too)
            e["AB_STAGE_TOKEN"] = token
            if gpus is not None:
                e["CUDA_VISIBLE_DEVICES"] = _csv(gpus) if gpus else ""
            log(f"{name}: start (attempt {attempts}): {' '.join(cmd)}", NAME)
            with open(out, "a", encoding="utf-8") as fh:
                fh.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} attempt {attempts}: {' '.join(cmd)}\n")
                fh.flush()
                proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=e, cwd=str(_repo()),
                                        start_new_session=True)
                self.procs[name] = proc
                atomic_write_json(PROCS / f"{name}.json", {"pid": proc.pid, "host": socket.gethostname(),
                                                           "token": token, "time": time.time()})
                code, why = self._watch(name, proc, out, heartbeats, hang, deadline_grace_h, stop_grace)
                reap_token(token)  # descendants that outlived the stage's own process
                self.procs.pop(name, None)
                (PROCS / f"{name}.json").unlink(missing_ok=True)
            if code == 0 and done():
                self.set_stage(name, status="done", finished=time.time())
                log(f"{name}: done", NAME)
                return True
            if self.halted():
                self.set_stage(name, status="stopped")
                return False
            if why == "deadline":
                self.set_stage(name, status="skipped", reason="deadline")
                return False
            tail = _tail(out, 25)
            msg = f"{name}: attempt {attempts} failed ({why or f'exit {code}'}); log tail:\n{tail}"
            log(msg, NAME)
            self.errors.append(f"{time.strftime('%H:%M')} {name}: {why or f'exit {code}'}")
            self.set_stage(name, status="retrying", last_error=why or f"exit {code}")
            self.nap(min(900, 60 * 2 ** min(attempt, 4)))
        self.set_stage(name, status="failed")
        log(f"{name}: FAILED after {retries + 1} attempts" + ("" if critical else " (non-critical, continuing)"), NAME)
        return False

    def _watch(self, name, proc, out: Path, heartbeats: tuple, hang: float, grace_h: float, stop_grace: float):
        t0 = time.time()
        while True:
            try:
                code = proc.wait(timeout=15)
                return code, ""
            except subprocess.TimeoutExpired:
                pass
            if self.halted():
                _kill(proc, stop_grace)
                return -1, "stopped"
            if self.past_deadline(grace_h):
                _kill(proc, stop_grace)
                return -1, "deadline"
            paths = [out, *[Path(h) for h in heartbeats]]
            newest = max([t0] + [p.stat().st_mtime for p in paths if p.exists()])
            if time.time() - newest > hang:
                log(f"{name}: no activity for {hang / 60:.0f} min, killing it", NAME)
                _kill(proc, stop_grace)
                return -1, "hang"

    def torchrun(self, n: int, module: str, args: list[str]) -> list[str]:
        if n <= 1:
            return [PY, "-m", module, *args]
        return [PY, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={n}", "-m", module, *args]

    # ---------------------------------------------------------------- lanes
    def lane(self, fn, name):
        def wrapped():
            try:
                fn()
            except Exception:
                msg = traceback.format_exc()
                log(f"lane {name} crashed:\n{msg}", NAME)
                self.errors.append(f"{time.strftime('%H:%M')} lane {name} crashed")
                # a supervisor bug: cancel everything; run_ab.sh restarts it and every stage resumes
                self.abort(f"lane {name} crashed: {msg.strip().splitlines()[-1]}")
        t = threading.Thread(target=wrapped, name=name, daemon=True)
        t.start()
        return t

    def main_lane(self):
        cfg, L = self.cfg, self.layout
        if not self.run("download", [PY, "-m", "abtrain.data", "download"], gpus=[], done=_raw_ready, retries=10):
            return self.abort("download failed")
        ex = [PY, "-m", "abtrain.extract", "--workers", str(min(cfg["extract_workers"], max(1, L.n_cores - 2)))]
        if not self.run("extract", ex, gpus=[], done=_extract_done, hang_minutes=60):
            return self.abort("extraction failed")
        self.run("stage_shm", [PY, "-m", "abtrain.data", "stage"], gpus=[], critical=False, done=_staged_or_skipped)
        n = max(1, len(L.gpus))
        if not self.run("bc1", self.torchrun(n, "abtrain.bc", ["--mode", "bc1", "--out", str(RUNS / "bc1")]),
                        gpus=L.gpus, done=lambda: _final_ok(RUNS / "bc1"), heartbeats=(LOGS / "bc_bc1.log",)):
            return self.abort("stage 1 BC failed")
        protect(RUNS / "bc1" / "final.pt", CKPT / "A" / "bc_stage1_human_only.pt", {"stage": 1})
        league_copy(RUNS / "bc1" / "final.pt", LEAGUE / "bc" / "A1.pt")
        # calibration decides whether stage 2 can trust the win-prob head
        self.run("calib", [PY, "-m", "abtrain.vref", "--model", str(RUNS / "bc1" / "final.pt"), "--calib"],
                 gpus=L.gpus[:1], critical=False, done=lambda: (RUNS / "bc1" / "calib.json").exists())
        calib = read_json(RUNS / "bc1" / "calib.json", {}) or {}
        plain = not calib.get("pass", False)
        if not plain:
            ok = self.run("vref", self.torchrun(n, "abtrain.vref", ["--model", str(RUNS / "bc1" / "final.pt")]),
                          gpus=L.gpus, done=_vref_done, heartbeats=(LOGS / "vref.log",), critical=False)
            plain = not ok
        if plain:
            log("stage 2 runs as plain BC: the stage-1 value is not calibrated (or V_ref failed)", NAME)
        args = ["--mode", "awbc_A", "--out", str(RUNS / "awbc_A"), "--init", str(RUNS / "bc1" / "final.pt")]
        if plain:
            args.append("--plain")
        # A's stage 2 is also the start of B's stage 2, so by default it keeps all GPUs (B has
        # nothing to run before it exists); "stage2_A_all_gpus": false gives it only A's half
        g2 = L.gpus if cfg["stage2_A_all_gpus"] else L.a_gpus
        if not self.run("awbc_A", self.torchrun(len(g2) or 1, "abtrain.bc", args), gpus=g2,
                        done=lambda: _final_ok(RUNS / "awbc_A"), heartbeats=(LOGS / "bc_awbc_A.log",)):
            return self.abort("stage 2 for A failed")
        protect(RUNS / "awbc_A" / "final.pt", CKPT / "A" / "bc_stage2_human_only.pt",
                {"stage": 2, "advantage_weighted": not plain, "calibration": calib,
                 "note": "human matches only, before any PPO or FirstLight exposure; D reuses its win-prob head"})
        league_copy(RUNS / "awbc_A" / "final.pt", LEAGUE / "bc" / "A2.pt")
        # A/B test: stage 2 has to beat stage 1 clearly, or A's PPO starts from stage 1
        res_path = RUNS / "abtest_A.json"
        self.run("abtest_A", [PY, "-m", "abtrain.simeval", "--a", str(RUNS / "awbc_A" / "final.pt"),
                              "--b", str(RUNS / "bc1" / "final.pt"), "--decks", "popular", "--out", str(res_path),
                              "--workers", str(max(1, len(L.a_cores) // 2))],
                 gpus=L.gpus[:1], critical=False, done=res_path.exists)
        if self.halted():
            return
        ab = read_json(res_path, {}) or {}
        a_init = RUNS / "awbc_A" / "final.pt"
        if not _ab_ok(ab, cfg):
            a_init = RUNS / "bc1" / "final.pt"
            log(f"A/B test: stage 2 not accepted ({_ab_text(ab)} vs stage 1, needs {cfg['abtest_min_winrate']}): "
                "A's PPO starts from stage 1", NAME)
        self.state["a_init"] = str(a_init)
        self.save_state()
        ta = self.lane(lambda: self.ppo_lane("A", a_init, L.a_gpus, L.a_cores), "lane_A")
        tb = self.lane(lambda: self.b_lane(a_init), "lane_B")
        ta.join()
        tb.join()

    def b_lane(self, a_init: Path):
        cfg, L = self.cfg, self.layout
        args = ["--mode", "awbc_B", "--out", str(RUNS / "awbc_B"), "--init", str(RUNS / "awbc_A" / "final.pt")]
        if not _vref_done():
            args.append("--plain")
        ok = self.run("awbc_B", self.torchrun(len(L.b_gpus), "abtrain.bc", args), gpus=L.b_gpus,
                      done=lambda: _final_ok(RUNS / "awbc_B"), heartbeats=(LOGS / "bc_awbc_B.log",), critical=False)
        b_init = a_init
        if ok:
            protect(RUNS / "awbc_B" / "final.pt", CKPT / "B" / "bc_stage2_human_only.pt",
                    {"stage": 2, "run": "B", "note": "human Hog 2.6 matches + 30% general, before any PPO"})
            league_copy(RUNS / "awbc_B" / "final.pt", LEAGUE / "bc" / "B2.pt")
            res_path = RUNS / "abtest_B.json"
            self.run("abtest_B", [PY, "-m", "abtrain.simeval", "--a", str(RUNS / "awbc_B" / "final.pt"),
                                  "--b", str(RUNS / "awbc_A" / "final.pt"), "--decks", "hog26", "--out",
                                  str(res_path), "--workers", str(max(1, len(L.b_cores) // 2))],
                     gpus=L.b_gpus[:1], critical=False, done=res_path.exists)
            if self.halted():
                return
            ab = read_json(res_path, {}) or {}
            if _ab_ok(ab, cfg):
                b_init = RUNS / "awbc_B" / "final.pt"
            else:
                log(f"A/B test: B's stage 2 not accepted ({_ab_text(ab)} vs A's stage 2 in the Hog mirror): "
                    f"B's PPO starts from {a_init.parent.name}", NAME)
        self.state["b_init"] = str(b_init)
        self.save_state()
        self.ppo_lane("B", b_init, L.b_gpus, L.b_cores)

    def ppo_lane(self, run: str, init: Path, gpus: list[int], cores: list[int]):
        rd = RUNS / f"ppo_{run}"
        actor, learner = (0, 1) if len(gpus) >= 2 else (0, 0)
        cmd = [PY, "-m", "abtrain.ppo", "--run", run, "--actor-gpu", str(actor), "--learner-gpu", str(learner),
               "--cores", _csv(cores), "--init", str(init)]
        from .ppo import run_finished

        if not self.wait_standins():
            if not self.halted():
                log(f"PPO {run} not started: the FirstLight stand-ins never became ready before the deadline", NAME)
            return
        ok = self.run(f"ppo_{run}", cmd, gpus=gpus, done=lambda: run_finished(run), retries=30,
                      heartbeats=(rd / "heartbeat", rd / "actor_heartbeat"), deadline_grace_h=0.5,
                      stop_grace=780)
        final = rd / "final.pt"
        if final.exists():
            dst = CKPT / run / "ppo_final.pt"
            shutil.copyfile(final, dst.with_name(".ppo_final.tmp"))
            os.replace(dst.with_name(".ppo_final.tmp"), dst)
        if not ok and self.stage_info(f"ppo_{run}").get("status") == "failed":
            self.abort(f"PPO {run} failed after its retries (see LOGS/ppo_{run}.out)")

    def standins_ready(self) -> bool:
        return all((LEAGUE / "standins" / f"{t}.pt").exists() for t in ("general", "hog2"))

    def wait_standins(self) -> bool:
        """Both FirstLight stand-ins must be in the league before PPO starts (the fl lane keeps
        retrying the checkout, recording, cloning and checks meanwhile). With
        ``require_standins: false`` PPO instead starts after ``standin_wait_hours`` and the
        stand-ins join the league when they exist. Returns whether PPO may start."""
        strict = self.cfg["require_standins"]
        until = time.time() + self.cfg["standin_wait_hours"] * 3600
        warned = False
        while not self.standins_ready() and not self.halted() and not self.past_deadline():
            if not strict and time.time() >= until:
                log(f"FirstLight stand-ins still missing after {self.cfg['standin_wait_hours']} h: PPO starts "
                    "without them (require_standins is off); they join the league when ready", NAME)
                return True
            if not warned:
                log("PPO waits for both FirstLight stand-ins to be in the league", NAME)
                warned = True
            self.nap(60)
        return self.standins_ready() or (not strict and not self.halted() and not self.past_deadline())

    def ensure_fl(self) -> bool:
        """FirstLight checkout present, cloning it again if setup could not (both FL lanes call this)."""
        from .flmatch import fl_ready

        with self.fl_lock:
            if fl_ready() or os.name == "nt":
                return fl_ready()
            self.run("fl_clone", ["bash", str(_repo() / "abtrain" / "fl_clone.sh"), str(FL_REPO), FL_COMMIT],
                     gpus=[], retries=8, critical=False, done=fl_ready)
            return fl_ready()

    def wait_fl(self) -> bool:
        """FirstLight checkout present, retrying every 30 min until the deadline."""
        while not self.halted() and not self.past_deadline():
            if self.ensure_fl():
                return True
            log("FirstLight checkout unavailable: no stand-ins or FL evaluation yet; retrying in 30 min", NAME)
            self.nap(1800)
        return False

    def fl_lane(self):
        if not self.wait_for("extract") or not self.wait_fl():
            return
        # A and B must keep FirstLight stand-ins in the league: keep trying until both are in
        # (or the deadline), with a pause between rounds of retries
        while not self.halted() and not self.past_deadline():
            missing = self.standins_round()
            if not missing:  # done, or stage 1 never finished (nothing to clone into)
                return
            log(f"stand-ins still missing: {missing}; trying again in 30 min", NAME)
            self.nap(1800)

    def standins_round(self) -> list[str] | None:
        """One attempt at recording, cloning, checking and installing both stand-ins.
        Returns the tags still missing from the league, or None if stage 1 failed."""
        from .flmatch import record_complete

        cfg, L = self.cfg, self.layout
        missing = [t for t in ("general", "hog2") if not (LEAGUE / "standins" / f"{t}.pt").exists()]
        if not self.run("fl_record", [PY, "-m", "abtrain.flmatch", "record", "--workers", str(cfg["fl_record_workers"])],
                        gpus=L.gpus, critical=False, done=record_complete, hang_minutes=90):
            return missing  # train only on complete recordings
        if not self.wait_for("bc1"):
            return None
        for tag, suite in (("general", "mixed"), ("hog2", "hog26")):
            if (LEAGUE / "standins" / f"{tag}.pt").exists() and self.is_done(f"standin_{tag}"):
                continue
            src = FL_SHARDS / tag
            if not any(src.glob("*.npz")):
                log(f"no recorded FirstLight {tag} games yet: no {tag} stand-in this round", NAME)
                continue
            out = RUNS / f"standin_{tag}"
            ok = self.run(f"standin_{tag}", [PY, "-m", "abtrain.bc", "--mode", "standin", "--shards", str(src),
                                             "--init", str(RUNS / "bc1" / "final.pt"), "--out", str(out),
                                             "--workers", "2", "--batch", str(min(32, cfg["bc_batch_per_gpu"])),
                                             "--token-budget", str(min(100_000, cfg["bc_token_budget"]))],
                          gpus=L.gpus[-1:], critical=False, done=lambda o=out: _final_ok(o),
                          heartbeats=(LOGS / "bc_standin.log",))
            if not ok:
                continue
            check = RUNS / f"standin_{tag}_check.json"
            self.run(f"standin_check_{tag}", [PY, "-m", "abtrain.flmatch", "eval", "--ckpt", str(out / "final.pt"),
                                              "--opp", tag, "--suite", suite, "--decode", "sample",
                                              "--games", str(cfg["standin_check_games"]), "--out", str(check),
                                              "--workers", str(len(L.eval_cores))],
                     gpus=L.gpus, critical=False, done=check.exists)
            res = read_json(check, {}) or {}
            score = res.get("score")
            if score is None:  # not validated: not installed this round
                continue
            if score < 0.2:
                log(f"stand-in {tag} scores only {score:.2f} against the real FirstLight {tag}; "
                    "it still joins the league (kept by request), weighted like the other hard opponents", NAME)
            league_copy(out / "final.pt", LEAGUE / "standins" / f"{tag}.pt", force=True)
            log(f"stand-in {tag} installed in the league (check vs real: {res.get('score')})", NAME)
        return [t for t in ("general", "hog2") if not (LEAGUE / "standins" / f"{t}.pt").exists()]

    def eval_lane(self):
        if not self.wait_for("extract") or not self.wait_fl():
            return
        L = self.layout
        base = RUNS / "eval" / "fl_baseline.json"
        self.run("fl_baseline", [PY, "-m", "abtrain.flmatch", "baseline", "--out", str(base),
                                 "--workers", str(len(L.eval_cores))],
                 gpus=L.gpus, critical=False, done=base.exists)
        if not self.wait_for("bc1"):
            return
        self.run("evaluator", [PY, "-m", "abtrain.evaluator"], gpus=L.gpus, retries=30,
                 done=lambda: (read_json(RUNS / "eval" / "DONE", {}) or {}).get("deadline", 0) >= self.deadline - 60,
                 deadline_grace_h=1.6, hang_minutes=120,
                 heartbeats=(RUNS / "eval" / "heartbeat",))
        if self.stage_info("evaluator").get("status") == "failed":  # evaluation vs FL General is required
            self.abort("the evaluator failed after its retries (see LOGS/evaluator.out)")

    def abort(self, why: str):
        """The pipeline cannot go on (a required stage failed, a lane crashed): stop every stage and
        exit with code 4, so run_ab.sh tries again later; finished stages are kept."""
        if self.stop.is_set():  # a requested stop is not a failure
            return
        log(f"pipeline cancelled: {why}", NAME)
        self.errors.append(f"{time.strftime('%H:%M')} {why}")
        self.state["aborted"] = why
        self.save_state()
        self.cancel.set()

    # ---------------------------------------------------------------- loop
    def supervise(self):
        log(f"supervisor up (run {self.state['runs']}), deadline {time.strftime('%Y-%m-%d %H:%M', time.localtime(self.deadline))}; "
            f"{self.layout.describe()}", NAME)
        self.state.pop("aborted", None)
        self.state.pop("unmet", None)
        threads = [self.lane(self.main_lane, "main"), self.lane(self.fl_lane, "fl"), self.lane(self.eval_lane, "eval")]
        t_status = 0.0
        while any(t.is_alive() for t in threads) or any(
                t.is_alive() for t in threading.enumerate() if t.name.startswith("lane_")):
            if STOP_FILE.exists():
                log("STOP file found: stopping every stage", NAME)
                self.stop.set()  # each stage watcher shuts its stage down with that stage's grace
            if time.time() - t_status > 120:
                t_status = time.time()
                _heartbeat()
                try:
                    write_status(self)
                except Exception as exc:
                    log(f"status report failed: {exc!r}", NAME)
            time.sleep(5)
        if not self.standins_ready() and not self.halted():
            self.state["unmet"] = "FirstLight stand-ins were never ready (see LOGS/fl_record.out, standin_*.out)"
            self.save_state()
            log("REQUIREMENT NOT MET: the FirstLight stand-ins never made it into the league", NAME)
        write_status(self)
        log("supervisor: all lanes finished", NAME)


# ============================================================================ helpers


def _repo() -> Path:
    return Path(__file__).resolve().parent.parent


def _kill(proc: subprocess.Popen, grace: float = 180.0):
    """SIGTERM to the stage's own process first (PPO's coordinator then has its learner write a
    checkpoint; torchrun forwards it to its ranks), then the whole process group, then SIGKILL."""
    if proc.poll() is None:
        try:
            proc.terminate()
            proc.wait(timeout=grace)
        except Exception:
            pass
    if not hasattr(os, "killpg"):
        if proc.poll() is None:
            proc.kill()
        return
    for sig, wait in ((signal.SIGTERM, 20), (signal.SIGKILL, 5)):
        try:
            os.killpg(proc.pid, sig)  # leftovers in the group (workers of a dead parent)
        except OSError:
            return
        t0 = time.time()
        while time.time() - t0 < wait:
            try:
                os.killpg(proc.pid, 0)
            except OSError:
                return
            time.sleep(1)


def _tail(path: Path, n: int) -> str:
    try:
        with open(path, "rb") as fh:
            fh.seek(0, 2)
            fh.seek(max(0, fh.tell() - 8000))
            return "\n".join(fh.read().decode("utf-8", "replace").splitlines()[-n:])
    except OSError:
        return ""


def _ab_ok(ab: dict, cfg: dict) -> bool:
    """Stage 2 is accepted only on a complete A/B result that clears the bar."""
    return bool(ab) and ab.get("games", 0) >= cfg["abtest_games"] and ab["a_score"] >= cfg["abtest_min_winrate"]


def _ab_text(ab: dict) -> str:
    return f"scored {ab['a_score']:.3f} over {ab.get('games')} games" if ab else "no result"


def _raw_ready() -> bool:
    from .data import raw_ready

    return raw_ready()


def _extract_done() -> bool:
    return (DATA / "index.json").exists() and (SHARDS / ".complete").exists()


def _staged_or_skipped() -> bool:
    return True  # staging is an optimisation; training falls back to the network copy


def _final_ok(out: Path) -> bool:
    f = out / "final.pt"
    if not f.exists():
        return False
    try:
        return int(torch_load(f).get("step", 0)) > 0
    except Exception:
        return False


def _vref_done() -> bool:
    from .bc import vref_path

    idx = read_json(DATA / "index.json", {}) or {}
    shards = idx.get("shards", [])
    return bool(shards) and all(vref_path(s["path"]).exists() for s in shards)


def protect(src: Path, dst: Path, meta: dict) -> None:
    """Keep a copy that is never overwritten (read-only, skipped if it already exists)."""
    if dst.exists() or not src.exists():
        return
    ck = torch_load(src)
    sd = ck["model"] if "model" in ck else ck
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch_save_atomic({"model": sd, "meta": {**meta, "source": str(src), "saved": time.time()}}, dst)
    atomic_write_json(dst.with_suffix(".json"), {**meta, "source": str(src), "saved": time.time()})
    for p in (dst, dst.with_suffix(".json")):
        try:
            os.chmod(p, 0o444)
        except OSError:
            pass
    log(f"kept {dst} (never overwritten)", NAME)


def league_copy(src: Path, dst: Path, force: bool = False) -> None:
    if (dst.exists() and not force) or not src.exists():
        return
    ck = torch_load(src)
    sd = ck["model"] if "model" in ck else ck
    torch_save_atomic({"model": {k: v.half() for k, v in sd.items()}}, dst)


def reap_stale() -> None:
    """Kill every process of stages a crashed supervisor left running on this host (stages run in
    their own sessions and torchrun puts its ranks in new ones, so they outlive their parents)."""
    PROCS.mkdir(parents=True, exist_ok=True)
    for f in PROCS.glob("*.json"):
        info = read_json(f, {}) or {}
        if info.get("host") == socket.gethostname() and info.get("token"):
            if _token_pids(info["token"]):
                log(f"killing processes of stage {f.stem} left by a previous supervisor", NAME)
            reap_token(info["token"], grace=90)
        f.unlink(missing_ok=True)


def _token_pids(token: str) -> list[int]:
    """Processes of one stage attempt: they all carry AB_STAGE_TOKEN in their environment."""
    proc = Path("/proc")
    if not proc.is_dir():
        return []
    want = f"AB_STAGE_TOKEN={token}".encode()
    me = os.getpid()
    out = []
    for d in proc.iterdir():
        if not d.name.isdigit() or int(d.name) == me:
            continue
        try:
            if want in (d / "environ").read_bytes().split(b"\0"):
                out.append(int(d.name))
        except OSError:
            pass
    return out


def reap_token(token: str, grace: float = 20.0) -> None:
    pids = _token_pids(token)
    for sig, wait in ((signal.SIGTERM, grace), (getattr(signal, "SIGKILL", signal.SIGTERM), 5)):
        if not pids:
            return
        for pid in pids:
            try:
                os.kill(pid, sig)
            except OSError:
                pass
        t0 = time.time()
        while time.time() - t0 < wait and pids:
            time.sleep(1)
            pids = [p for p in pids if _alive(p)]


def acquire_lock():
    """One supervisor per host: an exclusive lock on a local (non-NFS) file, held for life."""
    try:
        import fcntl
    except ImportError:  # Windows dev box
        return True
    SHM.mkdir(parents=True, exist_ok=True)
    fh = open(SHM / "supervisor.lock", "a+")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        return None
    return fh


def _heartbeat():
    atomic_write_json(PID_FILE, {"pid": os.getpid(), "host": socket.gethostname(), "time": time.time()})


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


# ============================================================================ status


def _last_jsonl(path: Path) -> dict | None:
    try:
        with open(path, "rb") as fh:
            fh.seek(0, 2)
            fh.seek(max(0, fh.tell() - 20000))
            lines = fh.read().decode("utf-8", "replace").strip().splitlines()
        return json.loads(lines[-1]) if lines else None
    except Exception:
        return None


def write_status(sup: Supervisor | None = None) -> str:
    st = read_json(STATE, {}) or {}
    cfg = settings()
    started = st.get("started_at", time.time())
    deadline = started + cfg["deadline_hours"] * 3600
    lines = [f"# A/B training status ({time.strftime('%Y-%m-%d %H:%M:%S')})", "",
             f"Started {time.strftime('%Y-%m-%d %H:%M', time.localtime(started))}, deadline "
             f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(deadline))} ({(deadline - time.time()) / 3600:.1f} h left), "
             f"supervisor start #{st.get('runs', 0)}" + (f", MAIN LANE STOPPED: {st['aborted']}" if st.get("aborted") else "")
             + (f", REQUIREMENT NOT MET: {st['unmet']}" if st.get("unmet") else "")
             + ("" if all((LEAGUE / "standins" / f"{t}.pt").exists() for t in ("general", "hog2"))
                else ", FirstLight stand-ins not in the league yet"),
             "", "## Stages", "", "| stage | status | attempts | updated |", "|---|---|---|---|"]
    for name, s in st.get("stages", {}).items():
        lines.append(f"| {name} | {s.get('status')} | {s.get('attempts', 0)} | "
                     f"{time.strftime('%m-%d %H:%M', time.localtime(s.get('updated', 0)))} |")
    lines += ["", "## Supervised stages", ""]
    for stage in ("bc1", "awbc_A", "awbc_B", "standin_general", "standin_hog2"):
        m = _last_jsonl(RUNS / stage / "metrics.jsonl")
        v = _last_jsonl(RUNS / stage / "val.jsonl")
        if m or v:
            lines.append(f"- {stage}: step {m and m.get('step')}/{m and m.get('total')}, "
                         f"{(m or {}).get('rows_per_s', 0):.0f} rows/s, loss {(m or {}).get('loss', 0):.3f}; "
                         f"held-out policy NLL {(v or {}).get('policy_nll', 0):.4f}, value CE {(v or {}).get('value_ce', 0):.4f}")
    idx = read_json(DATA / "index.json", None)
    if idx:
        lines.append(f"- data: {idx['stats'].get('matches', 0)} matches, {idx['total_steps'] / 1e6:.1f}M decisions, "
                     f"{len(idx['shards'])} shards")
    lines += ["", "## PPO", ""]
    for run in ("A", "B"):
        rd = RUNS / f"ppo_{run}"
        lm = _last_jsonl(rd / "learner_metrics.jsonl")
        am = _last_jsonl(rd / "actor_metrics.jsonl")
        if not lm:
            lines.append(f"- {run}: not started")
            continue
        lines.append(f"- {run}: {lm.get('decisions', 0) / 1e6:.2f}M decisions, iteration {lm.get('iteration')}, "
                     f"KL to stage 2 {lm.get('kl_ref', 0):.4f} (coef {lm.get('kl_coef', 0):.4f}, budget "
                     f"{lm.get('kl_budget', 0):.3f}), explained var {lm.get('ev', 0):.2f}, entropy {lm.get('ent', 0):.3f}"
                     + (" — DONE" if (rd / "DONE").exists() else ""))
        if am:
            lines.append(f"  - games: score by opponent {am.get('score_by_opponent')}, plays/game "
                         f"{am.get('plays_per_game', 0):.1f}, first play {am.get('first_play_s')}, wins vs passive "
                         f"{am.get('wins_vs_passive_share', 0):.2f}, delay {am.get('delay_ticks', 0):.1f} ticks")
    lines += ["", "## Against the real FirstLight General", ""]
    base = read_json(RUNS / "eval" / "fl_baseline.json", None)
    if base:
        lines.append("- FL Hog 2 vs FL General in this simulator (the bar): "
                     + ", ".join(f"{k} {v['win_rate']:.3f} ({v['games']} games)" for k, v in base.items()
                                 if isinstance(v, dict)) + f"; target {cfg['target_winrate']:.2f}")
    for run in ("A", "B"):
        es = read_json(RUNS / "eval" / f"{run}_state.json", None)
        if not es:
            continue
        latest = es.get("latest") or {}
        best = es.get("best") or {}
        lat = ", ".join(f"{k} {v['win_rate']:.3f}/{v['games']}" for k, v in latest.items() if isinstance(v, dict))
        lines.append(f"- {run}: latest {latest.get('label')} ({lat}); best {best.get('win_rate')} on {best.get('suite')} "
                     f"({best.get('games')} games{', GOAL REACHED' if best.get('goal_reached') else ''}); "
                     f"improving {es.get('improving')}; refs {es.get('references')}")
    lines += ["", "## Machine", ""]
    for p in (WORK, SHM):
        try:
            du = shutil.disk_usage(p)
            lines.append(f"- {p}: {du.free / 2**30:.0f} GB free of {du.total / 2**30:.0f} GB")
        except OSError:
            pass
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=30).stdout.strip()
        lines += [f"- GPU {r}" for r in q.splitlines()]
    except Exception:
        pass
    if sup and sup.errors:
        lines += ["", "## Recent problems", ""] + [f"- {e}" for e in sup.errors[-15:]]
    lines += ["", f"Logs: {LOGS}. Checkpoints kept in {CKPT}."]
    text = "\n".join(lines) + "\n"
    try:
        STATUS.write_text(text, encoding="utf-8")
    except OSError:
        pass
    return text


# ============================================================================ CLI


def main(argv=None):
    cmd = (argv or sys.argv[1:] or ["run"])[0]
    ensure_dirs()
    if cmd == "status":
        print(STATUS.read_text(encoding="utf-8") if STATUS.exists() else write_status())
        return
    if cmd == "stop":
        STOP_FILE.write_text(str(time.time()))
        print(f"stop requested ({STOP_FILE}); the supervisor stops every stage within a minute")
        return
    if cmd != "run":
        raise SystemExit(f"unknown command {cmd}")
    lock = acquire_lock()
    if lock is None:
        print(f"a supervisor is already running on this host; see {STATUS}")
        return
    STOP_FILE.unlink(missing_ok=True)
    _heartbeat()
    sup = Supervisor()

    def _term(*_):  # every stage watcher kills its process within ~15 s
        sup.stop.set()

    signal.signal(signal.SIGTERM, _term)
    signal.signal(signal.SIGINT, _term)
    try:
        sup.supervise()
    finally:
        try:
            PID_FILE.unlink()
        except OSError:
            pass
    if sup.stop.is_set():
        sys.exit(3)
    if sup.state.get("aborted"):
        sys.exit(4)
    if sup.state.get("unmet"):  # finished, but a required part never happened: not a success
        sys.exit(5)


if __name__ == "__main__":
    main()
