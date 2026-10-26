"""One of our checkpoints vs a FirstLight model with a command delay, e.g. the newest A snapshot
vs FL General on FirstLight's focus decks, both sides at the live 21 ticks:

    python -m abtrain.fl_delay_eval --games 20 --delay 21

Seat-swapped pairs as in the evaluator. ``--fl-delay`` defaults to ``--delay`` (both sides
delayed); ``--fl-delay 0`` gives only our side the delay.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import LEAGUE, RUNS, atomic_write_json, log
from .flmatch import FLPool, eval_tasks, fl, ours, summarize, visible_gpus


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="", help="default: the newest league/<run>/snap_*.pt")
    ap.add_argument("--run", default="A")
    ap.add_argument("--opp", default="general", help="FirstLight model: general / hog2")
    ap.add_argument("--suite", default="mixed", help="mixed (same focus deck both sides), pop_vs_focus, ...")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--delay", type=int, default=21, help="our side's command delay, ticks (21 = live)")
    ap.add_argument("--fl-delay", type=int, default=None, help="FirstLight's; default = --delay")
    ap.add_argument("--decode", default="argmax")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    ckpt = a.ckpt or str(sorted((LEAGUE / a.run).glob("snap_*.pt"))[-1])
    fl_delay = a.delay if a.fl_delay is None else a.fl_delay
    tasks = eval_tasks(ours(ckpt, a.decode), fl(a.opp), a.suite, a.games, a.seed)
    for t in tasks:
        t["delay"], t["fl_delay"] = a.delay, fl_delay
    log(f"{Path(ckpt).name} vs FL {a.opp} on {a.suite}: {len(tasks)} games, delay {a.delay} (FL {fl_delay}), "
        f"decode {a.decode}", "fl_delay_eval")
    pool = FLPool(a.workers, visible_gpus())
    try:
        res = summarize(pool.run(tasks, progress=lambda i, n: print(f"\r{i}/{n} games", end="", flush=True)))
    finally:
        pool.close()
    print()
    res.update({"ckpt": ckpt, "opp": a.opp, "suite": a.suite, "delay": a.delay, "fl_delay": fl_delay,
                "decode": a.decode})
    out = Path(a.out) if a.out else RUNS / "eval" / f"delay_{Path(ckpt).stem}_{a.suite}_d{a.delay}_fl{fl_delay}.json"
    atomic_write_json(out, res)
    lo, hi = res["wilson95"]
    print(f"{Path(ckpt).name}: win rate {res['win_rate']:.3f} over {res['games']} games "
          f"(95% CI {lo:.2f}-{hi:.2f}), {res['errors']} failed, plays/game {res['plays_per_game']:.1f} -> {out}")
    print(json.dumps({k: res[k] for k in ("games", "wins", "draws", "win_rate", "errors", "error_sample")}))


if __name__ == "__main__":
    main()
