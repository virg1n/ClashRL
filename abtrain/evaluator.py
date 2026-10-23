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
  B (Hog specialist): hog26 (Hog 2.6 mirror vs FL General; primary), hog_vs_focus (B on
     Hog 2.6 vs FL General on a focus deck)
The stage-1/2 checkpoints are screened first, as the starting point.

Promotion: a screen at or above the gate (``promote_gate``) and within 3 points of the best
so far earns ``fl_promo_games`` more games on the primary suite; if the pooled win rate beats
the best, the snapshot is promoted: copied to ``LEAGUE/promoted`` (a hard opponent for both
runs) and to ``checkpoints/<run>/best_vs_general.pt`` with its numbers. The better of the
``t05`` and ``argmax`` decodings is measured on promotion and recorded (Stage 5).

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
SUITES = {"A": ("mixed", "pop_vs_focus", "pop_vs_hog2"), "B": ("hog26", "hog_vs_focus")}
PRIMARY = {r: s[0] for r, s in SUITES.items()}
EVAL_VERSION = 2  # bumped when the suites change: stage-2 references are screened again


def _snapshots(run: str) -> list[Path]:
    return sorted((LEAGUE / run).glob("snap_*.pt"))


def _seed(path: str) -> int:
    return zlib.crc32(str(path).encode()) % 100000


class Evaluator:
    def __init__(self):
        self.cfg = settings()
        EV.mkdir(parents=True, exist_ok=True)
        self.done = read_json(EV / "evaluated.json", {}) or {}
        # checkpoints whose screen could not complete get another try after a restart
        self.done = {k: v for k, v in self.done.items() if not str(v).endswith("(incomplete)")}
        self.state = {r: read_json(EV / f"{r}_state.json", None) or {"best": None, "history": [], "improving": True,
                                                                      "promotions": []} for r in ("A", "B")}
        gpus = visible_gpus()
        self.pool = FLPool(self.cfg["fl_eval_workers"], gpus)
        self.turn = 0
        self.failures = 0
        self.baseline_done = False

    def save(self):
        atomic_write_json(EV / "evaluated.json", self.done)
        for r in ("A", "B"):
            atomic_write_json(EV / f"{r}_state.json", self.state[r])

    def play(self, path: str, suite: str, games: int, decode: str, seed: int) -> dict:
        tasks = eval_tasks(ours(path, decode), fl(suite_opponent(suite)), suite, games, seed)
        res = summarize(self.pool.run(tasks, progress=lambda i, n: _beat()))
        if res["errors"]:  # only complete, seat-balanced evaluations count (failed games were replayed twice)
            raise EvalFailed(f"{res['errors']}/{len(tasks)} games failed, e.g. {res['error_sample']}")
        return res

    def screen(self, run: str, path: Path, label: str) -> dict:
        cfg = self.cfg
        prim = PRIMARY[run]
        seed = _seed(path)
        t0 = time.time()
        res = {"run": run, "label": label, "path": str(path), "time": time.time(), "decode": "t05",
               "suites": list(SUITES[run]), "version": EVAL_VERSION}
        for k, suite in enumerate(SUITES[run]):
            games = cfg["fl_eval_games"] if k == 0 else max(2, cfg["fl_eval_games"] // 2)
            res[suite] = self.play(str(path), suite, games, "t05", seed + k)
        try:
            res["decisions"] = int(torch_load(path).get("decisions", 0))
        except Exception:
            res["decisions"] = None
        res["primary"] = prim
        res["minutes"] = (time.time() - t0) / 60
        rest = ", ".join(f"{s} {res[s]['win_rate']:.3f} ({res[s]['games']})" for s in SUITES[run][1:])
        log(f"screen {run} {label}: {prim} {res[prim]['win_rate']:.3f} ({res[prim]['games']} games), {rest}, "
            f"plays/game {res[prim]['plays_per_game']:.1f}, {res['minutes']:.1f} min", NAME)
        return res

    def maybe_promote(self, run: str, path: Path, res: dict) -> None:
        cfg = self.cfg
        st = self.state[run]
        prim = PRIMARY[run]
        wr = res[prim]["win_rate"]
        best = st["best"]["win_rate"] if st["best"] else 0.0
        if wr < cfg["promote_gate"] or wr < best - 0.03:
            return
        seed = _seed(path) + 7
        more = self.play(str(path), prim, cfg["fl_promo_games"], "t05", seed)
        wins = res[prim]["wins"] + more["wins"]
        n = res[prim]["games"] + more["games"]
        pooled = wins / max(1, n)
        lo, hi = wilson(wins, n)
        log(f"promotion check {run} {path.name}: pooled {pooled:.3f} over {n} games (best so far {best:.3f})", NAME)
        res["promotion"] = {"games": n, "wins": wins, "win_rate": pooled, "wilson95": [lo, hi]}
        if pooled <= best:
            return
        # Stage 5 decoding: argmax (cards/tiles) gets the same number of games as t05 had; the
        # decoder kept is the one with the higher win rate, with its own statistics
        alt = self.play(str(path), prim, n, "argmax", seed + 1)
        decode = "t05"
        if alt["win_rate"] > pooled:
            decode, wins, n = "argmax", alt["wins"], alt["games"]
            pooled = wins / max(1, n)
            lo, hi = wilson(wins, n)
            if pooled <= best:
                return
        ck = torch_load(path)
        sd = ck["model"] if "model" in ck else ck
        tag = f"{run}_{path.stem[5:]}"
        torch_save_atomic({"model": {k: v.half() for k, v in sd.items()}, "run": run}, LEAGUE / "promoted" / f"{tag}.pt")
        info = {"run": run, "source": str(path), "decisions": res.get("decisions"), "suite": prim,
                "opponent": "FirstLight General (argmax)", "win_rate": pooled, "games": n, "wilson95": [lo, hi],
                "decode": decode, "argmax_check": alt, "other_suites": {s: res[s] for s in SUITES[run][1:]},
                "time": time.time(),
                "target": cfg["target_winrate"], "goal_reached": lo > cfg["target_winrate"]}
        dst = CKPT / run / "best_vs_general.pt"
        torch_save_atomic({"model": sd, "decode": decode, "eval": info}, dst)
        atomic_write_json(dst.with_suffix(".json"), info)
        st["best"] = info
        st["promotions"].append({"tag": tag, "win_rate": pooled, "games": n, "time": time.time()})
        log(f"PROMOTED {run} {tag}: {pooled:.3f} vs General on {prim} ({n} games, 95% CI {lo:.3f}-{hi:.3f}), "
            f"decode {decode}{'  GOAL REACHED' if info['goal_reached'] else ''}", NAME)

    def record(self, run: str, res: dict, ref: bool = False) -> None:
        st = self.state[run]
        prim = PRIMARY[run]
        append_jsonl(EV / f"{run}.jsonl", res)
        if ref:
            st.setdefault("references", {})[res["label"]] = {s: res[s]["win_rate"] for s in SUITES[run]}
            if st["best"] is None or res[prim]["win_rate"] > st["best"]["win_rate"]:
                # the human-only start is the bar PPO snapshots must beat; it is not re-promoted
                st["best"] = {"source": res["path"], "win_rate": res[prim]["win_rate"], "games": res[prim]["games"],
                              "suite": prim, "reference": True}
        else:
            h = st["history"]
            h.append({"decisions": res.get("decisions"), "win_rate": res[prim]["win_rate"],
                      "other": {s: res[s]["win_rate"] for s in SUITES[run][1:]}, "time": res["time"]})
            recent_promo = any(time.time() - p["time"] < 6 * 3600 for p in st["promotions"][-1:])
            scores = [x["win_rate"] for x in h]
            st["improving"] = bool(len(scores) < 4 or recent_promo
                                   or np.mean(scores[-2:]) > np.mean(scores[-4:-2]) + 1e-9)
        st["latest"] = {"label": res["label"], "decisions": res.get("decisions"),
                        **{s: res[s] for s in SUITES[run]}}
        self.done[self._key(Path(res["path"]))] = res["label"]
        self.save()

    def _key(self, path: Path) -> str:
        return f"{path}@{path.stat().st_mtime_ns}@v{EVAL_VERSION}"

    def _seen(self, path: Path, any_version: bool = True) -> bool:
        """Snapshots screened under any suite version stay done; references (stage 1/2) are
        screened again once with new suites (``any_version=False``)."""
        if self._key(path) in self.done:
            return True
        base = f"{path}@{path.stat().st_mtime_ns}"
        return any_version and any(k == base or k.startswith(base + "@v") for k in self.done)

    def next_reference(self) -> tuple[str, Path, str] | None:
        """The stage-1/2 checkpoints, screened once as the starting point."""
        for run, label, path, rescreen in (("A", "stage1", RUNS / "bc1" / "final.pt", False),
                                           ("A", "stage2", RUNS / "awbc_A" / "final.pt", True),
                                           ("B", "stage2", RUNS / "awbc_B" / "final.pt", True)):
            if path.exists() and not self._seen(path, any_version=not rescreen):
                return run, path, label
        return None

    def baseline_extra(self) -> None:
        """FL Hog 2 in B's second setting (Hog 2.6 vs FL General on focus decks), once: the bar
        for ``hog_vs_focus``. The original baseline covers hog26 and mixed."""
        path = EV / "fl_baseline.json"
        base = read_json(path, {}) or {}
        if isinstance(base.get("hog_vs_focus"), dict) or not base:
            return
        tasks = eval_tasks(fl("hog2"), fl("general"), "hog_vs_focus", self.cfg["fl_eval_games"], 4243)
        res = summarize(self.pool.run(tasks, progress=lambda i, n: _beat()))
        if res["errors"]:
            log(f"baseline hog_vs_focus incomplete ({res['errors']} errors); trying again later", NAME)
            return
        base["hog_vs_focus"] = res
        atomic_write_json(path, base)
        log(f"baseline hog_vs_focus (FL Hog 2 on Hog 2.6 vs FL General on focus decks): "
            f"{res['win_rate']:.3f} ({res['games']} games)", NAME)

    def next_snapshot(self) -> tuple[str, Path] | None:
        for k in range(2):
            run = ("A", "B")[(self.turn + k) % 2]
            snaps = _snapshots(run)
            if snaps and not self._seen(snaps[-1]):
                self.turn = (self.turn + k + 1) % 2
                return run, snaps[-1]
        return None

    def finished(self) -> bool:
        from .ppo import run_finished

        for run in ("A", "B"):
            if not run_finished(run):
                return False
            snaps = _snapshots(run)
            if snaps and not self._seen(snaps[-1]):
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
            elif not self.baseline_done:
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
                    self.done[self._key(path)] = label + " (incomplete)"
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
