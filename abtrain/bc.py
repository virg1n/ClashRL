"""Supervised stages on all 4 GPUs (torchrun, DDP):

  bc1      stage 1: plain BC + win/draw/loss head on every non-held-out human sequence
  awbc_A   stage 2 for A: 0.3 plain + 0.7 advantage-weighted BC, advantages from V_ref sidecars
  awbc_B   stage 2 for B: same loss, Hog 2.6 matches (both seats) + 30% general windows,
           warm-started from A's stage 2, early stopping on held-out Hog NLL
  standin  FirstLight stand-in: policy-only BC on FirstLight games recorded through flport

    torchrun --nproc_per_node 4 -m abtrain.bc --mode bc1 --out RUNS/bc1

Resumes from ``<out>/latest.pt``. Writes ``final.pt`` when done, ``metrics.jsonl`` and a log.
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as Fn

from .common import (LOGS, RUNS, SHARDS, SHM, append_jsonl, log, read_json, settings, torch_load,
                     torch_save_atomic)
from .model import build_model, param_count, to_device_batch
from .shards import collate, load_shard

VALUE_MODULES = ("value", "v_res", "priv_mlp", "priv_lin", "priv_slot")


# ---------------------------------------------------------------------------- data


def data_root() -> Path:
    """Shards are read from the /dev/shm copy when staged, else from the network home."""
    staged = SHM / "shards"
    return staged if (staged / ".complete").exists() else SHARDS


def vref_path(rel: str) -> Path:
    return RUNS / "vref" / (rel.replace("/", "__") + ".npy")


def advantages(store: dict, vref: np.ndarray, n: int) -> np.ndarray:
    """A_t = V(t+n) - V(t) inside each sequence. Past the end of a game that really ended, V is
    the final outcome; a truncated replay (simulator diverged) has no look-ahead for its last n
    steps, which get NaN (plain, unweighted BC)."""
    adv = np.full(len(store["lab"]), np.nan, np.float32)
    off = store["seq_off"]
    meta = store["seq_meta"]
    for s in range(len(off) - 1):
        a, b = int(off[s]), int(off[s + 1])
        v = vref[a:b].astype(np.float32)
        if meta[s, 3]:  # truncated
            if b - a > n:
                adv[a : b - n] = v[n:] - v[:-n]
            continue
        ext = np.concatenate([v, np.full(n, float(meta[s, 2]), np.float32)])
        adv[a:b] = ext[n : n + (b - a)] - v
    return adv


class WindowStream(torch.utils.data.IterableDataset):
    """Infinite stream of [B, L] batches from this rank's share of shards.

    kind: "train" (non-held-out), "val" (held-out matches only).
    Selection: "all", "hog" (sequences of matches with a Hog 2.6 seat), or "hogmix" (hog windows,
    plus ``general_frac`` of each batch from other sequences)."""

    def __init__(self, shards: list[str], root: Path, rank: int, world: int, batch: int, window: int,
                 burn_in: int, kind: str, select: str = "all", general_frac: float = 0.0,
                 with_adv: bool = False, adv_n: int = 24, seed: int = 0, feature_drop: float = 0.1,
                 general_shards: list[str] | None = None):
        super().__init__()
        self.shards = shards
        self.general_shards = general_shards or []
        self.root, self.rank, self.world = root, rank, world
        self.batch, self.window, self.burn = batch, window, burn_in
        self.kind, self.select, self.general_frac = kind, select, general_frac
        self.with_adv, self.adv_n, self.seed = with_adv, adv_n, seed
        self.feature_drop = feature_drop if kind == "train" else 0.0

    def _load(self, rel: str) -> dict | None:
        try:
            st = load_shard(self.root / rel)
            st["outcome"] = np.repeat(st["seq_meta"][:, 2], np.diff(st["seq_off"])).astype(np.int8)
            if self.with_adv:
                vp = vref_path(rel)
                v = np.load(vp)
                if len(v) != len(st["lab"]):
                    raise ValueError(f"vref length mismatch for {rel}")
                st["adv"] = advantages(st, v, self.adv_n)
            return st
        except Exception as exc:
            log(f"skipping shard {rel}: {exc!r}", "bc")
            return None

    def _windows(self, st: dict, rng: np.random.Generator, select: str) -> list[tuple]:
        meta, off = st["seq_meta"], st["seq_off"]
        loss_len = self.window - self.burn
        out = []
        for s in range(len(off) - 1):
            hold = meta[s, 7] == 1
            if (self.kind == "train") == bool(hold):
                continue
            if select == "hog" and meta[s, 5] != 1:
                continue
            if select == "general" and meta[s, 5] == 1:
                continue
            a, b = int(off[s]), int(off[s + 1])
            o = a + int(rng.integers(0, loss_len))
            if o > a:  # cover the opening too, with a shorter loss span that stops where the next starts
                out.append((a, b, a, o))
            for ls in range(o, b, loss_len):
                out.append((max(a, ls - self.burn), b, ls, min(b, ls + loss_len)))
        return out

    def _batches(self, pool: list, store_of: list, rng):
        rng.shuffle(pool)
        for i in range(0, len(pool) - self.batch + 1, self.batch):
            chunk = pool[i : i + self.batch]
            yield self._collate(chunk, store_of, rng)

    def _collate(self, chunk, stores, rng):
        # group by store to gather with one collate per store, then concatenate
        parts = []
        for sid in sorted({c[0] for c in chunk}):
            sel = [c for c in chunk if c[0] == sid]
            st = stores[sid]
            starts = np.array([c[1] for c in sel], np.int64)
            ends = np.array([c[2] for c in sel], np.int64)
            lfrom = np.array([c[3] for c in sel], np.int64)
            lto = np.array([c[4] for c in sel], np.int64)
            extra = {"outcome": st["outcome"]}
            if self.with_adv:
                extra["adv"] = st["adv"]
            parts.append(collate(st, starts, self.window, ends, lfrom, rng, self.feature_drop, extra, lto))
        out = {}
        emax = max(p["ent_f"].shape[2] for p in parts)
        for k in parts[0]:
            arrs = []
            for p in parts:
                x = p[k]
                if k.startswith("ent_") and x.shape[2] < emax:
                    pad = [(0, 0)] * x.ndim
                    pad[2] = (0, emax - x.shape[2])
                    x = np.pad(x, pad)
                arrs.append(x)
            out[k] = np.concatenate(arrs)
        return out

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        mine = self.shards[self.rank :: self.world][wid::nw] or self.shards[(self.rank * nw + wid) % max(1, len(self.shards)) :][:1]
        gen = self.general_shards[self.rank :: self.world][wid::nw] or self.general_shards
        epoch = 0
        while True:
            yielded = 0
            rng = np.random.default_rng((self.seed * 1000003 + self.rank * 101 + wid) * 7919 + epoch)
            order = list(mine)
            rng.shuffle(order)
            for i in range(0, len(order), 2):
                stores, pool = [], []
                for rel in order[i : i + 2]:
                    st = self._load(rel)
                    if st is None:
                        continue
                    sel = "hog" if self.select in ("hog", "hogmix") else "all"
                    ws = self._windows(st, rng, sel)
                    pool += [(len(stores), *w) for w in ws]
                    stores.append(st)
                if self.select == "hogmix" and self.general_frac > 0 and gen and pool:
                    n_gen = int(len(pool) * self.general_frac / (1 - self.general_frac))
                    rel = gen[int(rng.integers(0, len(gen)))]
                    st = self._load(rel)
                    if st is not None:
                        ws = self._windows(st, rng, "general")
                        rng.shuffle(ws)
                        pool += [(len(stores), *w) for w in ws[:n_gen]]
                        stores.append(st)
                if not pool:
                    continue
                for batch in self._batches(pool, stores, rng):
                    yielded += 1
                    yield batch
            if not yielded:  # nothing selectable in this worker's shards: draw from all of them
                if len(mine) >= len(self.shards):
                    return
                mine = list(self.shards)
            epoch += 1


def held_out_hog_shards(root: Path, rels: list[str]) -> list[str]:
    """Shards with at least one held-out sequence of a Hog 2.6 match (reads only seq_meta)."""
    out = []
    for rel in rels:
        try:
            with np.load(root / rel) as z:
                meta = z["seq_meta"]
            if ((meta[:, 7] == 1) & (meta[:, 5] == 1)).any():
                out.append(rel)
        except Exception as exc:
            log(f"skipping shard {rel}: {exc!r}", "bc")
    return out or rels[:1]


def make_loader(ds, workers: int):
    return torch.utils.data.DataLoader(ds, batch_size=None, num_workers=workers, pin_memory=True,
                                       prefetch_factor=4 if workers else None, persistent_workers=bool(workers))


# ---------------------------------------------------------------------------- loss


def bc_losses(out: dict, b: dict, pos_weight: float, weights: torch.Tensor | None = None) -> dict:
    """Per-part BC losses averaged over their applicable steps (optionally advantage-weighted)."""
    loss_m = b["loss"].reshape(-1)
    w = torch.ones_like(out["logp"]) if weights is None else weights.reshape(-1)
    can = out["can"] & loss_m
    act = out["act"] & loss_m
    gate_l = -torch.where(out["act"], pos_weight * out["lp_gate"], out["lp_gate"])
    play_l = -(out["lp_c1"] + out["lp_t1"] + out["lp_d1"] + out["lp_k"] + out["lp_c2"] + out["lp_t2"] + out["lp_d2"])
    wc = (w * can).sum().clamp_min(1e-6)
    wa = (w * act).sum().clamp_min(1e-6)
    gate = (gate_l * w * can).sum() / wc
    play = (play_l * w * act).sum() / wa
    return {"gate": gate, "play": play, "total": gate + play}


def value_loss(out: dict, b: dict, every: int, gen: torch.Generator | None = None) -> torch.Tensor:
    loss_m = b["loss"].reshape(-1)
    target = b["outcome_cls"].reshape(-1)
    r = torch.rand(loss_m.shape, device=loss_m.device, generator=gen)
    m = loss_m & (r < 1.0 / every)
    ce = Fn.cross_entropy(out["v_logits"], target, reduction="none")
    return (ce * m).sum() / m.sum().clamp_min(1.0)


# ---------------------------------------------------------------------------- training


def _setup_ddp():
    use_cuda = torch.cuda.is_available() and not os.environ.get("AB_FORCE_CPU")
    if "RANK" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        backend = os.environ.get("AB_DIST_BACKEND") or ("nccl" if use_cuda else "gloo")
        dist.init_process_group(backend, timeout=__import__("datetime").timedelta(minutes=30))
        rank, world = dist.get_rank(), dist.get_world_size()
        local = int(os.environ.get("LOCAL_RANK", 0))
    else:
        rank, world, local = 0, 1, 0
    dev = torch.device(f"cuda:{local}" if use_cuda else "cpu")
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
    return rank, world, dev


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["bc1", "awbc_A", "awbc_B", "standin"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--init", default="", help="checkpoint to start from (model weights)")
    ap.add_argument("--shards", default="", help="standin: directory of recorded shards")
    ap.add_argument("--passes", type=float, default=0.0)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--lr", type=float, default=0.0)
    ap.add_argument("--lr-end", type=float, default=0.0)
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--workers", type=int, default=-1)
    ap.add_argument("--val-every-min", type=float, default=25.0)
    ap.add_argument("--val-batches", type=int, default=24)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--token-budget", type=int, default=0, help="encoder tokens per forward (memory cap)")
    ap.add_argument("--plain", action="store_true",
                    help="stage 2 without advantage weights (fallback when the stage-1 value is uncalibrated)")
    a = ap.parse_args(argv)
    cfg = settings()
    rank, world, dev = _setup_ddp()
    out = Path(a.out)
    logname = f"bc_{a.mode}"
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
    say = (lambda m: log(m, logname)) if rank == 0 else (lambda m: None)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ---- data selection
    if a.mode == "standin":
        root = Path(a.shards)
        rels = sorted(p.relative_to(root).as_posix() for p in root.rglob("*.npz") if not p.name.startswith("."))
        total_steps = 0
        for r in rels:
            with np.load(root / r) as z:
                total_steps += int(z["seq_off"][-1])
        hog_rels, gen_rels = rels, []
    else:
        root = data_root()
        idx = read_json(SHARDS.parent / "index.json", {})
        shards = idx.get("shards", [])
        rels = [s["path"] for s in shards]
        total_steps = int(sum(s["steps"] - s["holdout_steps"] for s in shards))
        hog_rels = [s["path"] for s in shards if s["hog_steps"] > 0]
        gen_rels = [s["path"] for s in shards if s["hog_steps"] < s["steps"]]
        if a.mode == "awbc_B":  # 70% of each batch is Hog windows
            total_steps = int(sum(s["hog_steps"] for s in shards) * 0.98 / (1 - cfg["awbc_B_general_frac"]))
    if not rels:
        raise SystemExit("no shards found")
    batch = a.batch or cfg["bc_batch_per_gpu"]
    budget = a.token_budget or cfg["bc_token_budget"]
    window, burn = cfg["bc_window"], cfg["bc_burn_in"]
    if a.mode == "standin":
        window, burn = 64, 16
    workers = cfg["bc_workers_per_gpu"] if a.workers < 0 else a.workers
    with_adv = a.mode in ("awbc_A", "awbc_B") and not a.plain
    early_stop = a.mode in ("awbc_B", "standin")  # keep the best held-out policy NLL
    select = {"bc1": "all", "awbc_A": "all", "awbc_B": "hogmix", "standin": "all"}[a.mode]
    train_rels = hog_rels if a.mode == "awbc_B" else rels
    ds = WindowStream(train_rels, root, rank, world, batch, window, burn, "train", select,
                      cfg["awbc_B_general_frac"] if a.mode == "awbc_B" else 0.0, with_adv, cfg["awbc_n"],
                      a.seed, general_shards=gen_rels if a.mode == "awbc_B" else None)
    val_rels = rels
    if a.mode == "awbc_B":  # only shards that hold held-out Hog matches (rare): a validation pass
        val_rels = held_out_hog_shards(root, hog_rels)  # over all Hog shards outlasts the hang limit
        say(f"{len(val_rels)} of {len(hog_rels)} Hog shards contain held-out Hog matches")
    vds = WindowStream(val_rels, root, rank, world, batch, window, burn, "val",
                       "hog" if a.mode == "awbc_B" else "all", 0.0, False, cfg["awbc_n"], a.seed + 7)
    loader = make_loader(ds, workers)

    passes = a.passes or {"bc1": cfg["bc1_passes"], "awbc_A": cfg["awbc_passes_A"],
                          "awbc_B": float(cfg["awbc_B_max_passes"]), "standin": 3.0}[a.mode]
    steps_per_pass = max(1, total_steps // (world * batch * (window - burn)))
    total = int(passes * steps_per_pass)
    if a.mode == "standin":
        total = min(total, cfg["standin_steps"]) if total > 0 else cfg["standin_steps"]
    if a.max_steps or cfg["bc_max_steps"]:
        total = min(total, a.max_steps or cfg["bc_max_steps"])
    lr0 = a.lr or {"bc1": cfg["bc1_lr"], "standin": cfg["standin_lr"]}.get(a.mode, cfg["awbc_lr"])
    lr1 = a.lr_end or {"bc1": cfg["bc1_lr_end"], "standin": cfg["standin_lr"]}.get(a.mode, cfg["awbc_lr_end"])
    warm = cfg["bc_warmup_steps"] if a.mode == "bc1" else min(200, total // 20)
    say(f"mode {a.mode}: {len(train_rels)} shards, {total_steps} train steps, {steps_per_pass} optimizer "
        f"steps per pass, total {total}, world {world}, batch {batch}x{window}, lr {lr0}->{lr1}")

    # ---- model
    model = build_model().to(dev)
    if rank == 0:
        say(f"params {param_count(model) / 1e6:.2f}M")
    start_step, best_val, val_bad, value_coef = 0, float("inf"), 0, cfg["value_coef"]
    if a.mode == "standin":
        value_coef = 0.0
    best_value_state = None
    hog_best, hog_bad = float("inf"), 0
    opt = torch.optim.AdamW(model.parameters(), lr=lr0, weight_decay=cfg["bc_weight_decay"],
                            fused=dev.type == "cuda")
    latest = out / "latest.pt"
    if latest.exists():
        ck = torch_load(latest)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_step = int(ck["step"])
        best_val, val_bad, value_coef = ck.get("best_val", best_val), ck.get("val_bad", 0), ck.get("value_coef", value_coef)
        hog_best, hog_bad = ck.get("hog_best", hog_best), ck.get("hog_bad", 0)
        best_value_state = ck.get("best_value_state")
        say(f"resumed at step {start_step}")
        ds.seed = a.seed + 1000 * (ck.get("restarts", 0) + 1)
        restarts = ck.get("restarts", 0) + 1
    else:
        restarts = 0
        if a.init:
            ck = torch_load(a.init)
            model.load_state_dict(ck["model"] if "model" in ck else ck)
            say(f"initialised from {a.init}")
    if start_step >= total:
        if rank == 0 and not (out / "final.pt").exists():
            torch_save_atomic({"model": model.state_dict(), "mode": a.mode, "step": start_step}, out / "final.pt")
        say("already complete")
        return
    ddp = model
    if world > 1:
        ddp = torch.nn.parallel.DistributedDataParallel(model, device_ids=[dev.index] if dev.type == "cuda" else None,
                                                        find_unused_parameters=True)
    amp = dict(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda")

    def lr_at(step):
        if step < warm:
            return lr0 * (step + 1) / warm
        p = min(1.0, (step - warm) / max(1, total - warm))
        return lr1 + 0.5 * (lr0 - lr1) * (1 + math.cos(math.pi * p))

    def save(step, final=False):
        if rank != 0:
            return
        ck = {"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "mode": a.mode,
              "best_val": best_val, "val_bad": val_bad, "value_coef": value_coef, "restarts": restarts,
              "best_value_state": best_value_state, "hog_best": hog_best, "hog_bad": hog_bad}
        torch_save_atomic(ck, latest)
        if final:
            torch_save_atomic({"model": model.state_dict(), "mode": a.mode, "step": step}, out / "final.pt")

    gen = torch.Generator(device=dev)
    gen.manual_seed(a.seed * 31 + rank)
    t_ck = t_val = time.time()
    t0 = time.time()
    it = iter(loader)
    step = start_step
    acc = {}
    nonfinite = 0
    while step < total:
        try:
            b = next(it)
        except StopIteration:
            b = None
        # every rank must agree on stopping, validating and checkpointing, or DDP deadlocks
        val_due = time.time() - t_val > a.val_every_min * 60 or step + 1 >= total
        ck_due = time.time() - t_ck > cfg["checkpoint_minutes"] * 60
        ctl = torch.tensor([float(b is None), float(val_due), float(ck_due)], device=dev)
        if world > 1:
            dist.all_reduce(ctl, op=dist.ReduceOp.MAX)
            dist.broadcast(ctl[1:], 0)  # rank 0's clock decides validation and checkpoints
        if ctl[0] > 0:
            say("training data exhausted (no selectable windows left); stopping this stage early")
            break
        val_due, ck_due = bool(ctl[1] > 0), bool(ctl[2] > 0)
        b = to_device_batch(b, dev)
        b["outcome_cls"] = outcome_targets(b)
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        opt.zero_grad(set_to_none=True)
        parts = micro_splits(b, budget)
        for mi, sub in enumerate(parts):
            share = sub["glob"].shape[0] / b["glob"].shape[0]
            sync = mi == len(parts) - 1
            ctx = ddp.no_sync() if (world > 1 and not sync) else _nullctx()
            with ctx:
                with torch.autocast(**amp):
                    o = _ddp_eval(ddp, sub)
                    plain = bc_losses(o, sub, cfg["gate_pos_weight"])
                    loss = plain["total"]
                    if with_adv:
                        adv = sub["adv"].reshape(-1).float()
                        lm = sub["loss"].reshape(-1)
                        ok = lm & torch.isfinite(adv)  # steps with a look-ahead
                        adv = torch.nan_to_num(adv)
                        nv = ok.sum().clamp_min(1)
                        mu = (adv * ok).sum() / nv
                        sd = (((adv - mu) ** 2) * ok).sum().div(nv).sqrt()
                        w = torch.clamp(torch.exp(((adv - mu) / (sd + 1e-6)) / cfg["awbc_beta"]), 0, cfg["awbc_wmax"])
                        w = torch.where(ok, w, torch.where(lm, torch.ones_like(w), torch.zeros_like(w)))
                        weighted = bc_losses(o, sub, cfg["gate_pos_weight"], w)
                        loss = cfg["awbc_plain_frac"] * plain["total"] + (1 - cfg["awbc_plain_frac"]) * weighted["total"]
                        ess = (w.sum() ** 2 / (w.pow(2).sum() + 1e-9)) / lm.sum().clamp_min(1)
                        _acc(acc, "ess", ess.item())
                    if value_coef > 0:
                        vl = value_loss(o, sub, cfg["value_subsample"], gen)
                        loss = loss + value_coef * vl
                        _acc(acc, "value_ce", vl.item())
                    # a non-finite loss still runs backward (DDP needs every rank in the
                    # all-reduce); the shared non-finite gradient norm then skips the update
                (loss * share).backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["bc_grad_clip"])
        if not torch.isfinite(gn):
            nonfinite += 1
            say(f"non-finite gradient at step {step} ({nonfinite}), update skipped")
            opt.zero_grad(set_to_none=True)
            if nonfinite > 50:
                raise RuntimeError("too many non-finite updates")
            continue
        opt.step()
        b = sub
        step += 1
        _acc(acc, "loss", loss.item())
        _acc(acc, "gate", plain["gate"].item())
        _acc(acc, "play", plain["play"].item())
        _acc(acc, "grad_norm", float(gn))
        lm = b["loss"].reshape(-1)
        can = o["can"] & lm
        _acc(acc, "gate_p", (torch.sigmoid(o["gate_logit"]) * can).sum().item() / max(1, can.sum().item()))
        _acc(acc, "gate_y", (o["act"] & lm).sum().item() / max(1, can.sum().item()))
        if step % 100 == 0 and rank == 0:
            el = time.time() - t0
            rate = (step - start_step) * world * batch * window / max(el, 1e-6)
            rec = {"step": step, "total": total, "lr": lr_at(step), "rows_per_s": rate, "value_coef": value_coef,
                   **{k: v[0] / max(1, v[1]) for k, v in acc.items()}, "time": time.time()}
            append_jsonl(out / "metrics.jsonl", rec)
            say(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items() if k != "time"))
            acc = {}
        if val_due:
            t_val = time.time()
            vm = validate(model, vds, dev, a.val_batches, {**cfg, "bc_token_budget": budget}, amp)
            if rank == 0:
                append_jsonl(out / "val.jsonl", {"step": step, **vm, "time": time.time()})
                say("val " + " ".join(f"{k}={v:.4g}" for k, v in vm.items()))
                # value head early stop: freeze it once held-out value loss turns up
                if value_coef > 0 and vm.get("value_ce") is not None:
                    if vm["value_ce"] < best_val - 1e-4:
                        best_val, val_bad = vm["value_ce"], 0
                        best_value_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                                            if k.split(".")[0] in VALUE_MODULES}
                    elif vm["value_ce"] > best_val * 1.01:
                        val_bad += 1
            stop_hog = False
            if rank == 0 and early_stop and "policy_nll" in vm:  # no held-out windows: train the full budget
                nll = vm["policy_nll"]
                if nll < hog_best - 1e-4:
                    hog_best, hog_bad = nll, 0
                    torch_save_atomic({"model": model.state_dict(), "mode": a.mode, "step": step}, out / "best.pt")
                else:
                    hog_bad += 1
                stop_hog = hog_bad >= 3
            flags = torch.tensor([val_bad, int(stop_hog)], device=dev)
            if world > 1:
                dist.broadcast(flags, 0)
            if int(flags[0]) >= 2 and value_coef > 0:
                value_coef = 0.0
                if rank == 0:
                    say("value head stopped early (held-out value loss rising); restoring its best weights")
                if best_value_state is not None:
                    sd = model.state_dict()
                    sd.update({k: v.to(dev) for k, v in best_value_state.items()})
                    model.load_state_dict(sd)
                if world > 1:
                    for p in model.parameters():
                        dist.broadcast(p.data, 0)
            if int(flags[1]):
                say("held-out policy NLL stopped improving: early stop")
                break
        if ck_due:
            t_ck = time.time()
            save(step)
    if early_stop and rank == 0 and (out / "best.pt").exists():
        best = torch_load(out / "best.pt")
        model.load_state_dict(best["model"])
        say(f"using the best held-out checkpoint, from step {best['step']}")
    save(step, final=True)
    say(f"done at step {step}")
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


class _nullctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def micro_splits(b: dict, budget: int) -> list[dict]:
    """Split a batch along B so each part stays under ``budget`` encoder tokens; entity
    padding is trimmed per part. Long windows stay whole (the LSTM runs through them)."""
    bsz, t, e = b["ent_f"].shape[:3]
    n = max(1, min(bsz, math.ceil(bsz * t * (20 + e) / max(1, budget))))
    if n == 1:
        return [b]
    out = []
    for i in range(n):
        sl = slice(i * bsz // n, (i + 1) * bsz // n)
        sub = {k: v[sl] for k, v in b.items()}
        # the highest occupied slot (dropped tokens leave holes, so a count would cut real ones)
        occ = sub["ent_mask"].reshape(-1, e).any(0)
        em = int(occ.nonzero().max().item()) + 1 if bool(occ.any()) else 1
        for k in ("ent_ids", "ent_f", "ent_mask"):
            sub[k] = sub[k][:, :, :em]
        out.append(sub)
    return out


def _ddp_eval(ddp, b):
    """Run ``evaluate`` through DDP's forward hook so gradients are synchronised."""
    if isinstance(ddp, torch.nn.parallel.DistributedDataParallel):
        return ddp(b)
    return ddp.evaluate(b)


def outcome_targets(b: dict) -> torch.Tensor:
    """Win/draw/loss class per step (0 win, 1 draw, 2 loss) from the ``outcome`` field."""
    o = b["outcome"].long()
    return torch.where(o > 0, 0, torch.where(o < 0, 2, 1))


def _acc(acc: dict, k: str, v: float):
    s, n = acc.get(k, (0.0, 0))
    acc[k] = (s + v, n + 1)


@torch.no_grad()
def validate(model, vds, dev, n_batches: int, cfg: dict, amp: dict) -> dict:
    model.eval()
    it = iter(make_loader(vds, 0))
    tot: dict = {}
    briers, conf_early, conf_late = [], [], []
    bins = np.zeros((10, 3))
    for _ in range(n_batches):
        try:
            bb = to_device_batch(next(it), dev)
        except StopIteration:
            break
        bb["outcome_cls"] = outcome_targets(bb)
        for b in micro_splits(bb, cfg["bc_token_budget"]):
            _val_part(model, b, amp, tot, briers, bins, conf_early, conf_late)
    model.train()
    return _val_summary(tot, briers, bins, conf_early, conf_late)


def _val_part(model, b, amp, tot, briers, bins, conf_early, conf_late):
    if True:
        with torch.autocast(**amp):
            o = model.evaluate(b)
        lm = b["loss"].reshape(-1)
        can = o["can"] & lm
        act = o["act"] & lm
        _acc(tot, "gate_nll", (-(o["lp_gate"] * can).sum() / can.sum().clamp_min(1)).item())
        play = -(o["lp_c1"] + o["lp_t1"] + o["lp_d1"] + o["lp_k"] + o["lp_c2"] + o["lp_t2"] + o["lp_d2"])
        _acc(tot, "play_nll", ((play * act).sum() / act.sum().clamp_min(1)).item())
        _acc(tot, "policy_nll", (-(o["logp"] * lm).sum() / lm.sum().clamp_min(1)).item())
        _acc(tot, "card_nll", (-(o["lp_c1"] * act).sum() / act.sum().clamp_min(1)).item())
        _acc(tot, "tile_nll", (-(o["lp_t1"] * o["has_t1"] * lm).sum() / (o["has_t1"] & lm).sum().clamp_min(1)).item())
        ce = Fn.cross_entropy(o["v_logits"], b["outcome_cls"].reshape(-1), reduction="none")
        _acc(tot, "value_ce", ((ce * lm).sum() / lm.sum().clamp_min(1)).item())
        p = o["v_logits"].float().softmax(-1)
        y = Fn.one_hot(b["outcome_cls"].reshape(-1), 3).float()
        br = ((p - y) ** 2).sum(-1)
        briers.append(br[lm].cpu().numpy())
        pw = p[:, 0][lm].cpu().numpy()
        yw = (b["outcome_cls"].reshape(-1) == 0)[lm].cpu().numpy()
        for k in range(10):
            sel = (pw >= k / 10) & (pw < (k + 1) / 10 + (k == 9))
            bins[k] += [sel.sum(), pw[sel].sum(), yw[sel].sum()]
        tsec = (b["glob"][..., 2].reshape(-1) * 180.0)[lm].cpu().numpy()
        rem = (b["glob"][..., 3].reshape(-1) * 180.0)[lm].cpu().numpy()
        conf = (p[:, 0] - p[:, 2]).abs()[lm].cpu().numpy()
        conf_early.append(conf[tsec < 10])
        conf_late.append(conf[(rem < 30) & (tsec > 60)])


def _val_summary(tot, briers, bins, conf_early, conf_late) -> dict:
    res = {k: v[0] / max(1, v[1]) for k, v in tot.items()}
    if briers:
        res["brier3"] = float(np.concatenate(briers).mean())
    ce_ = np.concatenate(conf_early) if conf_early else np.zeros(0)
    cl_ = np.concatenate(conf_late) if conf_late else np.zeros(0)
    res["conf_first10s"] = float(ce_.mean()) if len(ce_) else float("nan")
    res["conf_last30s"] = float(cl_.mean()) if len(cl_) else float("nan")
    rel = [(int(n), float(ps / n), float(ys / n)) for n, ps, ys in bins if n > 0]
    res["calib_gap"] = float(sum(n * abs(pp - yy) for n, pp, yy in rel) / max(1, sum(n for n, _, _ in rel)))
    return res


if __name__ == "__main__":
    main()
