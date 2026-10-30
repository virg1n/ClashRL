"""Continuous evaluation against the real FirstLight General, and promotion.

    python -m abtrain.evaluator

Runs beside both PPO runs until the deadline (or until both are DONE and their final snapshots
are evaluated). Each turn it takes the newest not-yet-evaluated league snapshot of A or B
(alternating; older ones are skipped when it falls behind) and screens it against FirstLight
(argmax, FirstLight's benchmark protocol, seat-swapped pairs) on the run's suites
(``flmatch.suite_decks``): the first, primary suite gets ``fl_eval_games`` games, the others
half as many each.
  A (generalist): mixed (FL focus decks mirrored, vs FL General; primary), pop_vs_focus (A on
     a popular human deck vs FL General on a focus deck), pop_vs_hog2 (A on a popular human
     deck vs FL Hog 2 on Hog 2.6)
  B (Hog specialist): hog_vs_focus (B on Hog 2.6 vs FL General on a focus deck; primary)
The stage-1/2 checkpoints are screened first, as the starting point (in A's phase 2 or 3: only
the phase's start, ``checkpoints/A/phase<n>_init.pt``). B is skipped while it is paused (``run_B``).

A, phase 3: the candidate always plays with its handicap (command delay, late state, missing
inputs; ``a3_*``). Its screen is ``a3_eval_best_games`` games against the phase's start with no
handicap (``vs_best``) and ``a3_eval_fl_games`` against FirstLight General (``fl_general``), both
on ``mixed``'s decks. The score is the win rate over both (``pooled``); there are no extra
promotion games, and the best goes to ``checkpoints/A/best_delayed.pt``.

Every game uses one decoding, ``fl_eval_decode`` (argmax), so the screen, the promotion games
and the best's score are the same measurement (phase 1 screened with t05 but stored the best's
argmax score, so the gate compared two different things).

Promotion: a screen at or above the gate (``promote_gate``) and within 3 points of the best
so far earns ``fl_promo_games`` more games on the primary suite; if the pooled win rate beats
the best, the snapshot is promoted: copied to ``LEAGUE/promoted`` (a hard opponent for both
runs) and to ``checkpoints/<run>/best_vs_general.pt`` with its numbers.

Outputs: ``RUNS/eval/<run>.jsonl`` (every screen), ``RUNS/eval/<run>_state.json`` (best,
history, the ``improving`` flag the learner uses to relax its KL budget, goal flags).
"""

from __future__ import annotations

import time
import zlib
from pathlib import Path

import numpy as np

from .common import (CKPT, LEAGUE, RUNS, append_jsonl, atomic_write_json, deadline_ts, log, read_json, settings,
                     torch_load, torch_save_atomic, wilson)
from .flmatch import FLPool, eval_tasks, fl, ours, suite_opponent, summarize, visible_gpus

EV = RUNS / "eval"
NAME = "evaluator"
SUITES = {"A": ("mixed", "pop_vs_focus", "pop_vs_hog2"), "B": ("hog_vs_focus",)}
# bumped per run when its suites change: that run's stage-2 reference is screened again
VERSIONS = {"A": 2, "B": 3}


def _snapshots(run: str) -> list[Path]:
    return sorted((LEAGUE / run).glob("snap_*.pt"))


def _seed(path: str) -> int:
    return zlib.crc32(str(path).encode()) % 100000


class Evaluator:
    def __init__(self):
        self.cfg = settings()
        self.runs = ("A", "B") if self.cfg["run_B"] else ("A",)
        self.a3 = self.cfg["a_phase"] >= 3
        self.suites = dict(SUITES, A=("pooled", "vs_best", "fl_general")) if self.a3 else dict(SUITES)
        self.primary = {r: s[0] for r, s in self.suites.items()}
        self.decode = self.cfg["fl_eval_decode"]
        EV.mkdir(parents=True, exist_ok=True)
        self.done = read_json(EV / "evaluated.json", {}) or {}
        # checkpoints whose screen could not complete get another try after a restart
        self.done = {k: v for k, v in self.done.items() if not str(v).endswith("(incomplete)")}
        self.state = {r: read_json(EV / f"{r}_state.json", None) or {"best": None, "history": [], "improving": True,
                                                                      "promotions": []} for r in ("A", "B")}
        for r, st in self.state.items():  # a best or a trend measured on another main suite no longer applies
            if st.get("best") and st["best"].get("suite") != self.primary[r]:
                st["best"], st["history"], st["improving"] = None, [], True
        gpus = visible_gpus()
        self.pool = FLPool(self.cfg["fl_eval_workers"], gpus)
        self.turn = 0
        self.failures = 0
        self.baseline_done = False

    def save(self):
        atomic_write_json(EV / "evaluated.json", self.done)
        for r in ("A", "B"):
            atomic_write_json(EV / f"{r}_state.json", self.state[r])

    def plan(self, run: str) -> list[tuple[str, int]]:
        """(suite, games) of one screen."""
        cfg = self.cfg
        if run == "A" and self.a3:
            return [("vs_best", cfg["a3_eval_best_games"]), ("fl_general", cfg["a3_eval_fl_games"])]
        return [(s, cfg["fl_eval_games"] if k == 0 else max(2, cfg["fl_eval_games"] // 2))
                for k, s in enumerate(self.suites[run])]

    def play(self, run: str, path: str, suite: str, games: int, decode: str, seed: int) -> dict:
        if run == "A" and self.a3:
            cfg = self.cfg
            cand = ours(path, decode, delay=cfg["a3_cmd_delay_ticks"], lag=cfg["a3_obs_lag_ticks"], blind=cfg["a3_blind"])
            opp = ours(str(CKPT / "A" / "phase3_init.pt"), decode) if suite == "vs_best" else fl("general")
            tasks = eval_tasks(cand, opp, "mixed", games, seed)
        else:
            tasks = eval_tasks(ours(path, decode), fl(suite_opponent(suite)), suite, games, seed)
        res = summarize(self.pool.run(tasks, progress=lambda i, n: _beat()))
        if res["errors"]:  # only complete, seat-balanced evaluations count (failed games were replayed twice)
            raise EvalFailed(f"{res['errors']}/{len(tasks)} games failed, e.g. {res['error_sample']}")
        return res

    def screen(self, run: str, path: Path, label: str) -> dict:
        cfg = self.cfg
        prim = self.primary[run]
        seed = _seed(path)
        t0 = time.time()
        plan = self.plan(run)
        games = sum(g for _, g in plan)
        log(f"now screening {run} {label} ({games} games: {', '.join(s for s, _ in plan)})", NAME)
        atomic_write_json(EV / "current.json", {"run": run, "label": label, "suites": [s for s, _ in plan],
                                                "games": games, "started": t0})
        res = {"run": run, "label": label, "path": str(path), "time": time.time(), "decode": self.decode,
               "suites": list(self.suites[run]), "version": VERSIONS[run]}
        for k, (suite, games) in enumerate(plan):
            res[suite] = self.play(run, str(path), suite, games, self.decode, seed + k)
        if prim not in res:  # A phase 3: the score over both suites
            n = sum(res[s]["games"] for s, _ in plan)
            wins = sum(res[s]["wins"] for s, _ in plan)
            res[prim] = {"games": n, "wins": wins, "win_rate": wins / max(1, n), "wilson95": list(wilson(wins, n)),
                         "plays_per_game": sum(res[s]["plays_per_game"] * res[s]["games"] for s, _ in plan) / max(1, n)}
        try:
            res["decisions"] = int(torch_load(path).get("decisions", 0))
        except Exception:
            res["decisions"] = None
        res["primary"] = prim
        res["minutes"] = (time.time() - t0) / 60
        rest = "".join(f"{s} {res[s]['win_rate']:.3f} ({res[s]['games']}), " for s in self.suites[run][1:])
        log(f"screen {run} {label}: {prim} {res[prim]['win_rate']:.3f} ({res[prim]['games']} games), {rest}"
            f"plays/game {res[prim]['plays_per_game']:.1f}, {res['minutes']:.1f} min", NAME)
        return res

    def maybe_promote(self, run: str, path: Path, res: dict) -> None:
        cfg = self.cfg
        st = self.state[run]
        prim = self.primary[run]
        wr = res[prim]["win_rate"]
        best = st["best"]["win_rate"] if st["best"] else 0.0
        a3 = run == "A" and self.a3
        if not a3 and (wr < cfg["promote_gate"] or wr < best - 0.03):
            return
        wins, n = res[prim]["wins"], res[prim]["games"]
        if not a3:
            seed = _seed(path) + 7
            log(f"now playing {cfg['fl_promo_games']} promotion games for {run} {path.stem} ({wr:.3f} on {prim})", NAME)
            atomic_write_json(EV / "current.json", {"run": run, "label": f"{path.stem} promotion check", "suites": [prim],
                                                    "games": cfg["fl_promo_games"], "started": time.time()})
            more = self.play(run, str(path), prim, cfg["fl_promo_games"], self.decode, seed)
            wins += more["wins"]
            n += more["games"]
        pooled = wins / max(1, n)
        lo, hi = wilson(wins, n)
        log(f"promotion check {run} {path.name}: pooled {pooled:.3f} over {n} games (best so far {best:.3f})", NAME)
        res["promotion"] = {"games": n, "wins": wins, "win_rate": pooled, "wilson95": [lo, hi], "decode": self.decode}
        if pooled <= best:
            return
        decode = self.decode
        ck = torch_load(path)
        sd = ck["model"] if "model" in ck else ck
        tag = f"{run}_{path.stem[5:]}"
        for old in (LEAGUE / "promoted").glob(f"{run}_*.pt"):  # one promoted copy per run: the current best
            old.unlink(missing_ok=True)
        torch_save_atomic({"model": {k: v.half() for k, v in sd.items()}, "run": run}, LEAGUE / "promoted" / f"{tag}.pt")
        info = {"run": run, "source": str(path), "decisions": res.get("decisions"), "suite": prim,
                "opponent": "best A without handicap + FirstLight General" if a3 else "FirstLight General (argmax)",
                "win_rate": pooled, "games": n, "wilson95": [lo, hi],
                "decode": decode, "other_suites": {s: res[s] for s in self.suites[run][1:]},
                "time": time.time(),
                "target": cfg["target_winrate"], "goal_reached": lo > cfg["target_winrate"]}
        dst = CKPT / run / ("best_delayed.pt" if a3 else "best_vs_general.pt")  # phase 3 keeps the undelayed best
        torch_save_atomic({"model": sd, "decode": decode, "eval": info}, dst)
        atomic_write_json(dst.with_suffix(".json"), info)
        st["best"] = info
        st["promotions"].append({"tag": tag, "win_rate": pooled, "games": n, "time": time.time()})
        log(f"PROMOTED {run} {tag}: {pooled:.3f} on {prim} ({n} games, 95% CI {lo:.3f}-{hi:.3f}), "
            f"decode {decode}{'  GOAL REACHED' if info['goal_reached'] else ''}", NAME)

    def record(self, run: str, res: dict, ref: bool = False) -> None:
        st = self.state[run]
        prim = self.primary[run]
        append_jsonl(EV / f"{run}.jsonl", res)
        if ref:
            st.setdefault("references", {})[res["label"]] = {s: res[s]["win_rate"] for s in self.suites[run]}
            # phase 2 carries phase 1's best (200 games) over as the bar; a reference never replaces it
            if st["best"] is None or (res[prim]["win_rate"] > st["best"]["win_rate"]
                                      and not st["best"].get("carried_from_phase1")):
                # the human-only start is the bar PPO snapshots must beat; it is not re-promoted
                st["best"] = {"source": res["path"], "win_rate": res[prim]["win_rate"], "games": res[prim]["games"],
                              "suite": prim, "reference": True}
        else:
            h = st["history"]
            h.append({"decisions": res.get("decisions"), "win_rate": res[prim]["win_rate"],
                      "other": {s: res[s]["win_rate"] for s in self.suites[run][1:]}, "time": res["time"]})
            recent_promo = any(time.time() - p["time"] < 6 * 3600 for p in st["promotions"][-1:])
            scores = [x["win_rate"] for x in h]
            st["improving"] = bool(len(scores) < 4 or recent_promo
                                   or np.mean(scores[-2:]) > np.mean(scores[-4:-2]) + 1e-9)
        st["latest"] = {"label": res["label"], "decisions": res.get("decisions"),
                        **{s: res[s] for s in self.suites[run]}}
        try:
            self.done[self._key(Path(res["path"]), run)] = res["label"]
        except FileNotFoundError:  # pruned meanwhile; screened anyway
            self.done[f"{res['path']}@pruned@v{VERSIONS[run]}"] = res["label"]
        self.save()

    def _key(self, path: Path, run: str) -> str:
        return f"{path}@{path.stat().st_mtime_ns}@v{VERSIONS[run]}"

    def _seen(self, path: Path, run: str, any_version: bool = True) -> bool:
        """Snapshots screened under any suite version stay done; references (stage 1/2) are
        screened again once with new suites (``any_version=False``)."""
        if self._key(path, run) in self.done:
            return True
        base = f"{path}@{path.stat().st_mtime_ns}"
        return any_version and any(k == base or k.startswith(base + "@v") for k in self.done)

    def next_reference(self) -> tuple[str, Path, str] | None:
        """The stage-1/2 checkpoints (in A's phase 2: its start), screened once as the starting point."""
        if self.a3:
            refs = [("A", "phase3_start", CKPT / "A" / "phase3_init.pt", False)]
        elif self.cfg["a_phase"] >= 2:
            refs = [("A", "phase2_start", CKPT / "A" / "phase2_init.pt", False)]
        else:
            refs = [("A", "stage1", RUNS / "bc1" / "final.pt", False), ("A", "stage2", RUNS / "awbc_A" / "final.pt", True)]
        refs.append(("B", "stage2", RUNS / "awbc_B" / "final.pt", True))
        for run, label, path, rescreen in refs:
            if run in self.runs and path.exists() and not self._seen(path, run, any_version=not rescreen):
                return run, path, label
        return None

    def baseline_extra(self) -> None:
        """FL Hog 2 in B's setting (Hog 2.6 vs FL General on focus decks), once, with as many
        games as the other baselines: the bar for ``hog_vs_focus``."""
        path = EV / "fl_baseline.json"
        base = read_json(path, {}) or {}
        n = self.cfg["fl_baseline_games"]
        if not base or (isinstance(base.get("hog_vs_focus"), dict) and base["hog_vs_focus"].get("games", 0) >= n):
            return
        log(f"now measuring the FL Hog 2 bar on hog_vs_focus ({n} games)", NAME)
        atomic_write_json(EV / "current.json", {"run": "FL Hog 2", "label": "baseline", "suites": ["hog_vs_focus"],
                                                "games": n, "started": time.time()})
        tasks = eval_tasks(fl("hog2"), fl("general"), "hog_vs_focus", n, 4243)
        res = summarize(self.pool.run(tasks, progress=lambda i, n: _beat()))
        if res["errors"]:
            log(f"baseline hog_vs_focus incomplete ({res['errors']} errors); trying again later", NAME)
            return
        base["hog_vs_focus"] = res
        atomic_write_json(path, base)
        log(f"baseline hog_vs_focus (FL Hog 2 on Hog 2.6 vs FL General on focus decks): "
            f"{res['win_rate']:.3f} ({res['games']} games)", NAME)

    def next_snapshot(self) -> tuple[str, Path] | None:
        n = len(self.runs)
        for k in range(n):
            run = self.runs[(self.turn + k) % n]
            snaps = _snapshots(run)
            if snaps and not self._seen(snaps[-1], run):
                self.turn = (self.turn + k + 1) % n
                return run, snaps[-1]
        return None

    def finished(self) -> bool:
        from .ppo import run_finished

        for run in self.runs:
            if not run_finished(run):
                return False
            snaps = _snapshots(run)
            if snaps and not self._seen(snaps[-1], run):
                return False
        return True

    def loop(self):
        deadline = deadline_ts()
        log("evaluator up", NAME)
        # past the deadline the PPO runs stop; their final snapshots still get screened (1.5 h grace)
        while time.time() < deadline + 1.5 * 3600:
            _beat()
            ref = self.next_reference()
            if ref is not None:
                run, path, label = ref
            elif not self.baseline_done and "B" in self.runs:
                self.baseline_extra()
                self.baseline_done = True
                continue
            else:
                nxt = self.next_snapshot()
                if nxt is None:
                    if self.finished():
                        break
                    time.sleep(60)
                    continue
                run, path = nxt
                label = path.stem
            try:
                res = self.screen(run, path, label)
                if ref is None:
                    self.maybe_promote(run, path, res)
            except EvalFailed as exc:
                # broken workers (e.g. a CUDA error): fresh pool, same checkpoint again. After three
                # failures it is recorded as incomplete: no promotion, not in the trend
                self.failures += 1
                log(f"screen of {path.name} failed ({exc}); restarting the worker pool ({self.failures})", NAME)
                self.pool.close()
                self.pool = FLPool(self.cfg["fl_eval_workers"], visible_gpus())
                if self.failures >= 3:
                    self.failures = 0
                    append_jsonl(EV / f"{run}.jsonl", {"run": run, "label": label, "path": str(path),
                                                        "time": time.time(), "incomplete": str(exc)})
                    self.done[self._key(path, run)] = label + " (incomplete)"
                    self.save()
                continue
            self.failures = 0
            self.record(run, res, ref=ref is not None)
        atomic_write_json(EV / "DONE", {"time": time.time(), "deadline": deadline})
        log("evaluator finished", NAME)


class EvalFailed(RuntimeError):
    pass


def _beat():
    (EV / "heartbeat").write_text(str(time.time()))


def main(argv=None):
    ev = Evaluator()
    try:
        ev.loop()
    finally:
        ev.pool.close()


if __name__ == "__main__":
    main()
