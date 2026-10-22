"""Stage 0: re-simulate every replayable IL_Replay match and write training shards.

    python -m abtrain.extract --workers 46

Work is split into (parquet part, chunk of 500 matches) tasks. A task deletes any partial
shards of its own, writes its shards, then a ``done`` marker with its statistics, so a restart
skips finished tasks and redoes only interrupted ones. ``index`` (run after all tasks) totals the
markers into ``data/index.json``: shard sizes, Hog sequences, deck popularity, skip reasons.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .common import RAW, SHARDS, atomic_write_json, log, read_json, settings
from .game import deck_key

CHUNK = 500
_W: dict = {}


def _init_worker():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from .game import Catalog, make_engine, make_parser

    eng = make_engine()
    _W["engine"] = eng
    _W["parser"] = make_parser(eng)
    _W["cat"] = Catalog.build(eng)


def _parts() -> list[Path]:
    return sorted((RAW / "replays").glob("*.parquet"))


def tasks() -> list[tuple[str, int, int]]:
    import pyarrow.parquet as pq

    out = []
    for p in _parts():
        n = pq.ParquetFile(p).metadata.num_rows
        for c in range((n + CHUNK - 1) // CHUNK):
            out.append((str(p), c, n))
    return out


def task_name(part: str, chunk: int) -> str:
    return f"{Path(part).stem.replace('part-', 'p')}_c{chunk:02d}"


def run_task(part: str, chunk: int, holdout_mod: int, per_shard: int, limit: int = 0) -> dict:
    import pyarrow.parquet as pq

    from .replay import MIN_STEPS, plan_match, replay
    from .shards import ShardWriter, pack_rows

    if "engine" not in _W:
        _init_worker()
    eng, parser, cat = _W["engine"], _W["parser"], _W["cat"]
    name = task_name(part, chunk)
    out_dir = SHARDS / name[:5]
    for old in out_dir.glob(f"{name}_*.npz"):
        old.unlink()
    table = pq.read_table(part, columns=["payload_json"]).slice(chunk * CHUNK, CHUNK)
    texts = table.column(0).to_pylist()
    writer = ShardWriter(out_dir, name, per_shard)
    skipped, decks, hog_decks = Counter(), Counter(), Counter()
    stats = Counter()
    t0 = time.time()
    for i, text in enumerate(texts):
        if limit and stats["matches"] >= limit:
            break
        try:
            payload = json.loads(text)
            plan = plan_match(payload, cat)
            if isinstance(plan, str):
                skipped[plan] += 1
                continue
            seed = (plan.match_hash * 2654435761 + 12345) % (2**63)
            bufs, info = replay(eng, parser, cat, plan, seed)
            if info["truncated"]:
                skipped["truncated: " + info["reason"]] += 1
            items = []
            hold = int(plan.match_hash % holdout_mod == 0)
            any_hog = int(any(plan.hog))
            for s in (0, 1):
                if len(bufs[s]) < MIN_STEPS:
                    continue
                packed = pack_rows(bufs[s].rows, bufs[s].lab)
                meta = [plan.match_hash, s, info["outcome"][s], info["truncated"], int(plan.hog[s]),
                        any_hog, info["outcome_source"], hold]
                dk = deck_key(plan.decks[s])
                items.append((packed, meta, dk))
                decks[dk] += 1
                if plan.hog[s]:
                    hog_decks[dk] += 1
                stats["steps"] += len(bufs[s])
                stats["seqs"] += 1
                stats["hog_steps"] += len(bufs[s]) if any_hog else 0
                stats["plays"] += int((packed["lab"][:, 0] >= 0).sum())
                stats["second_plays"] += int((packed["lab"][:, 3] >= 0).sum())
            if not items:
                skipped["too short"] += 1
                continue
            writer.add_match(items)
            stats["matches"] += 1
            stats["sim_agrees"] += info["sim_agrees"]
            stats["sim_ended"] += int(info["outcome_source"] == 0)
            for k in ("press_refused", "tile_fixed", "tile_lost", "deferred", "skipped_steps"):
                stats[k] += info[k]
        except Exception as exc:  # one bad match never stops a task
            skipped["error " + type(exc).__name__] += 1
            if skipped["error " + type(exc).__name__] <= 2:
                log(f"[{name}] match {i}: {exc!r}\n{traceback.format_exc()[-800:]}", "extract")
    writer.flush()
    shards = []
    for p in sorted(out_dir.glob(f"{name}_*.npz")):
        with np.load(p) as z:
            meta = z["seq_meta"]
            off = z["seq_off"]
        lens = np.diff(off)
        shards.append({
            "path": p.relative_to(SHARDS).as_posix(), "steps": int(lens.sum()), "seqs": int(len(lens)),
            "holdout_steps": int(lens[meta[:, 7] == 1].sum()),
            "hog_steps": int(lens[meta[:, 5] == 1].sum()),
        })
    return {"task": name, "seconds": time.time() - t0, "stats": dict(stats), "skipped": dict(skipped),
            "decks": dict(decks), "hog_decks": dict(hog_decks), "shards": shards}


def _done_path(name: str) -> Path:
    return SHARDS / "done" / f"{name}.json"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=settings()["extract_workers"])
    ap.add_argument("--limit-tasks", type=int, default=settings()["extract_limit_tasks"], help="testing only")
    ap.add_argument("--limit-matches", type=int, default=settings()["extract_limit_matches"], help="testing only")
    a = ap.parse_args(argv)
    cfg = settings()
    (SHARDS / "done").mkdir(parents=True, exist_ok=True)
    every = tasks()
    if a.limit_tasks:
        every = every[: a.limit_tasks]
    todo = [t for t in every if not _done_path(task_name(t[0], t[1])).exists()]
    log(f"extract: {len(todo)} tasks to do with {a.workers} workers", "extract")
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init_worker) as ex:
        futs = {ex.submit(run_task, p, c, cfg["holdout_mod"], cfg["matches_per_shard"], a.limit_matches): (p, c)
                for p, c, _ in todo}
        for fut in as_completed(futs):
            p, c = futs[fut]
            res = fut.result()  # a crashed worker raises here; the supervisor restarts the stage
            atomic_write_json(_done_path(task_name(p, c)), res)
            done += 1
            st = res["stats"]
            log(f"extract: {done}/{len(todo)} {res['task']} matches={st.get('matches', 0)} "
                f"steps={st.get('steps', 0)} {res['seconds']:.0f}s, elapsed {(time.time() - t0) / 3600:.2f}h",
                "extract")
    index()
    (SHARDS / ".complete").write_text(str(len(every)))


def index() -> dict:
    """Total the task markers into ``data/index.json``."""
    shards, decks, hog_decks, skipped, stats = [], Counter(), Counter(), Counter(), Counter()
    for p in sorted((SHARDS / "done").glob("*.json")):
        r = read_json(p, {})
        shards += r.get("shards", [])
        decks.update(r.get("decks", {}))
        hog_decks.update(r.get("hog_decks", {}))
        skipped.update(r.get("skipped", {}))
        stats.update(r.get("stats", {}))
    idx = {
        "shards": shards, "stats": dict(stats), "skipped": dict(skipped.most_common()),
        "decks": dict(decks.most_common(5000)), "hog_decks": dict(hog_decks.most_common(50)),
        "total_steps": int(sum(s["steps"] for s in shards)),
        "holdout_steps": int(sum(s["holdout_steps"] for s in shards)),
        "hog_steps": int(sum(s["hog_steps"] for s in shards)),
    }
    atomic_write_json(SHARDS.parent / "index.json", idx)
    log(f"index: {len(shards)} shards, {idx['total_steps']} steps, {stats.get('matches', 0)} matches, "
        f"hog steps {idx['hog_steps']}, skipped {sum(skipped.values())}", "extract")
    return idx


if __name__ == "__main__":
    main()
