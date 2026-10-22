"""V_ref for advantage-weighted BC, and the stage-1 calibration check.

    torchrun --nproc_per_node 4 -m abtrain.vref --model RUNS/bc1/final.pt

Runs the frozen stage-1 win-prob head over every sequence of every shard, start to end with the
LSTM state carried exactly, and stores V = P(win) - P(loss) per decision as a float16 sidecar
(``RUNS/vref/<shard>.npy``). A sidecar that exists is skipped, so a restart resumes.

``--calib`` instead measures the stage-1 value on held-out matches (Brier score, reliability,
confidence early vs late) and writes ``RUNS/bc1/calib.json`` with a pass/fail verdict.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np
import torch

from .bc import WindowStream, data_root, validate, vref_path
from .common import RUNS, SHARDS, atomic_write_bytes, atomic_write_json, log, read_json, settings, torch_load
from .model import build_model, to_device_batch
from .shards import collate, load_shard

CHUNK = 64


@torch.no_grad()
def shard_values(model, st: dict, dev, amp: dict, budget: int) -> np.ndarray:
    off = st["seq_off"]
    n_seq = len(off) - 1
    lens = np.diff(off)
    out = np.zeros(int(off[-1]), np.float16)
    # sequences in groups small enough for one forward per chunk
    max_len_steps = CHUNK * 64
    groups, cur = [], []
    for s in np.argsort(-lens):
        cur.append(int(s))
        if len(cur) * CHUNK >= max_len_steps:
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    for g in groups:
        starts0 = off[g].astype(np.int64)
        ends = off[np.array(g) + 1].astype(np.int64)
        state = model.initial_state(len(g), dev)
        n_chunks = int(math.ceil((ends - starts0).max() / CHUNK))
        for k in range(n_chunks):
            starts = starts0 + k * CHUNK
            live = starts < ends
            b = collate(st, np.minimum(starts, ends - 1), CHUNK, ends, starts)
            b = to_device_batch(b, dev)
            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                enc, new_state = model.trunk(b, state)
                fb = {"priv_ids": b["priv_ids"].reshape(-1, b["priv_ids"].shape[-1]),
                      "priv_f": b["priv_f"].reshape(-1, b["priv_f"].shape[-1])}
                _, v = model.value_out(enc, fb)
            v = v.float().reshape(len(g), CHUNK).cpu().numpy()
            valid = b["valid"].cpu().numpy() & live[:, None]
            for i in range(len(g)):
                if not live[i]:
                    continue
                n = int(valid[i].sum())
                out[starts[i] : starts[i] + n] = v[i, :n]
            state = new_state
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--calib", action="store_true")
    ap.add_argument("--calib-batches", type=int, default=60)
    ap.add_argument("--calib-out", default="")
    a = ap.parse_args(argv)
    cfg = settings()
    # independent processes (no process group): each rank owns every world-th shard
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    use_cuda = torch.cuda.is_available() and not os.environ.get("AB_FORCE_CPU")
    dev = torch.device(f"cuda:{local}" if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.set_device(dev)
    model = build_model().to(dev)
    ck = torch_load(a.model)
    model.load_state_dict(ck["model"] if "model" in ck else ck)
    model.eval()
    amp = dict(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda")
    root = data_root()
    idx = read_json(SHARDS.parent / "index.json", {})
    rels = [s["path"] for s in idx.get("shards", [])]
    if a.calib:
        if rank == 0:
            vds = WindowStream(rels, root, 0, 1, cfg["bc_batch_per_gpu"], cfg["bc_window"], cfg["bc_burn_in"],
                               "val", "all", seed=11)
            m = validate(model, vds, dev, a.calib_batches, cfg, amp)
            ok = (m.get("brier3", 9) < cfg["calib_brier3_max"] and m.get("calib_gap", 9) < cfg["calib_gap_max"]
                  and m.get("conf_last30s", 0) > m.get("conf_first10s", 1) + cfg["calib_conf_gain"])
            m["pass"] = bool(ok)
            atomic_write_json(Path(a.calib_out) if a.calib_out else RUNS / "bc1" / "calib.json", m)
            log(f"calibration: {m}", "vref")
        return
    mine = rels[rank::world]
    done = 0
    for rel in mine:
        p = vref_path(rel)
        if p.exists():
            continue
        try:
            st = load_shard(root / rel)
            v = shard_values(model, st, dev, amp, cfg["bc_token_budget"])
        except Exception as exc:
            log(f"vref failed for {rel}: {exc!r}", "vref")
            raise
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + f".tmp{os.getpid()}.npy")
        np.save(tmp, v)
        os.replace(tmp, p)
        done += 1
        if done % 50 == 0:
            log(f"vref rank {rank}: {done}/{len(mine)} shards", "vref")
    log(f"vref rank {rank}: finished {len(mine)} shards", "vref")


if __name__ == "__main__":
    main()
