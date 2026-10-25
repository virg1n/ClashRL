"""Head-to-head matches between two of our checkpoints in RoyaleSim, batched on one GPU.

    python -m abtrain.simeval --a RUNS/awbc_A/final.pt --b RUNS/bc1/final.pt --games 200 --decks popular

Used for the stage-2 A/B tests (stage 2 vs stage 1 for A on popularity decks; B's stage 2 vs A's
in the Hog 2.6 mirror). Games come in seat-swapped pairs with the same seed and decks, so deck
and seat advantages cancel. Writes a JSON summary with the win rate of ``a`` and a Wilson
interval, plus the anti-passivity numbers.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from .common import atomic_write_json, log, settings, wilson


def _deck_fn(kind: str, cat, rng):
    from .game import HOG26, deck_key
    from .league import DeckBook

    book = DeckBook(cat, rng)
    if kind == "hog26":
        return lambda: (deck_key(HOG26), deck_key(HOG26)), book
    if kind == "hog_vs_popular":  # a on Hog 2.6, b on a popularity-weighted human deck
        return lambda: (deck_key(HOG26), book.popular()), book
    if kind == "focus":
        def f():
            d = book.focus_deck()
            return d, d
        return f, book

    def pop():
        d = book.popular()
        return d, d
    return pop, book


def head_to_head(path_a: str, path_b: str, games: int, decks: str = "popular", device: str = "cuda",
                 workers: int = 4, per_worker: int = 8, decode: str = "sample", delay: int = 0,
                 seed: int = 0, cores: list[int] | None = None, log_name: str = "simeval") -> dict:
    import torch

    from .env import BattleSpec, SeatSpec
    from .game import Catalog, make_engine, parse_deck_key
    from .model import DECODES, load_policy
    from .vec import VecBattles, batch_obs

    dev = torch.device(device if torch.cuda.is_available() or not device.startswith("cuda") else "cpu")
    rng = np.random.default_rng(seed)
    cat = Catalog.build(make_engine())
    deck_fn, book = _deck_fn(decks, cat, rng)
    models = {"a": load_policy(path_a, dev), "b": load_policy(path_b, dev)}
    dec = DECODES[decode]
    games += games % 2
    plan = []
    for g in range(0, games, 2):
        da, db = deck_fn()
        s = int(rng.integers(2**62))
        plan.append((g, 0, da, db, s))
        plan.append((g + 1, 1, da, db, s))
    workers = max(1, min(workers, (len(plan) + per_worker - 1) // per_worker))
    vec = VecBattles(workers, per_worker, {}, cores)
    nb = vec.n
    queue = list(plan)

    def spec_for(item):
        if item is None:  # filler battle once every game is handed out; never counted
            d = book.dealt(deck_fn()[0])
            return BattleSpec((SeatSpec(d), SeatSpec(list(d))), delay=delay, seed=int(rng.integers(2**62)),
                              meta={"filler": True})
        g, a_team, da, db, s = item
        dk = [None, None]
        dk[a_team], dk[1 - a_team] = da, db
        r = np.random.default_rng(s)
        seats = []
        for t in (0, 1):
            deck = parse_deck_key(dk[t])
            seats.append(SeatSpec([deck[i] for i in r.permutation(8)]))
        return BattleSpec(tuple(seats), delay=delay, seed=s, meta={"game": g, "a_team": a_team})

    def take():
        return spec_for(queue.pop(0) if queue else None)

    specs = [take() for _ in range(nb)]
    obs = vec.reset(specs)
    metas = [s.meta for s in specs]
    nxt = [take() for _ in range(nb)]
    core = models["a"].core_dim
    H = torch.zeros(nb, 2, core, device=dev)
    C = torch.zeros(nb, 2, core, device=dev)
    results = []
    t0 = time.time()
    amp = dict(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda")
    while len(results) < len(plan):
        labs = [[None, None] for _ in range(nb)]
        groups: dict[str, list] = {"a": [], "b": []}
        for k in range(nb):
            at = metas[k].get("a_team", 0)
            for s in (0, 1):
                groups["a" if s == at else "b"].append((k, s))
        with torch.no_grad(), torch.autocast(**amp):
            for key, seats in groups.items():
                if not seats:
                    continue
                ks = torch.tensor([k for k, _ in seats], device=dev)
                ss = torch.tensor([s for _, s in seats], device=dev)
                bt = batch_obs([obs[k][s] for k, s in seats], dev)
                lab, _lp, _v, _vl, (h2, c2) = models[key].act(bt, (H[ks, ss][None].contiguous(), C[ks, ss][None].contiguous()), **dec)
                H[ks, ss], C[ks, ss] = h2[0].float(), c2[0].float()
                for i, (k, s) in enumerate(seats):
                    labs[k][s] = lab[i].tolist()
        res = vec.step([(labs[k][0], labs[k][1]) for k in range(nb)], nxt)
        for k, (o2, _rew, done, info) in enumerate(res):
            if done:
                m = metas[k]
                if not m.get("filler"):
                    at = m["a_team"]
                    results.append({"game": m["game"], "a_team": at, "score": (info["result"][at] + 1) / 2.0,
                                    "plays": info["plays"][at], "opp_plays": info["plays"][1 - at],
                                    "first_play": info["first_play"][at], "ticks": info["ticks"]})
                    if len(results) % 20 == 0:
                        log(f"simeval {Path(path_a).parent.name} vs {Path(path_b).parent.name}: "
                            f"{len(results)}/{len(plan)} games, a scores {np.mean([r['score'] for r in results]):.3f}",
                            log_name)
                metas[k] = nxt[k].meta
                nxt[k] = take()
                H[k] = 0
                C[k] = 0
            obs[k] = o2
    vec.close()
    sc = [r["score"] for r in results]
    wins = sum(1 for x in sc if x == 1.0)
    lo, hi = wilson(sum(sc), len(sc))
    won = [r for r in results if r["score"] == 1.0]
    return {"a": str(path_a), "b": str(path_b), "decks": decks, "decode": decode, "delay": delay,
            "games": len(sc), "a_wins": wins, "draws": sum(1 for x in sc if x == 0.5), "a_score": float(np.mean(sc)),
            "wilson95": [lo, hi], "a_plays_per_game": float(np.mean([r["plays"] for r in results])),
            "wins_vs_passive_share": sum(1 for r in won if r["opp_plays"] <= 3) / max(1, len(won)),
            "seconds": time.time() - t0}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--games", type=int, default=0)
    ap.add_argument("--decks", default="popular", choices=["popular", "hog26", "focus", "hog_vs_popular"])
    ap.add_argument("--decode", default="sample")
    ap.add_argument("--delay", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--per-worker", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    cfg = settings()
    res = head_to_head(a.a, a.b, a.games or cfg["abtest_games"], a.decks, a.device, a.workers, a.per_worker,
                       a.decode, a.delay, a.seed)
    atomic_write_json(Path(a.out), res)
    log(f"simeval result: {res}", "simeval")


if __name__ == "__main__":
    main()
