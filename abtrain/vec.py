"""CPU worker processes that each run several battles; the parent batches inference on GPU.

Protocol (one round trip per decision for every battle):
  parent -> worker: [(lab_seat0, lab_seat1, next_spec), ...] for the worker's battles
  worker -> parent: [(obs_pair | None, rewards, done, info), ...]
A battle that ends is reset at once with ``next_spec``; its first observation comes back in the
same reply (``info['reset'] = True``). Observations travel with packed tile masks.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import traceback

import numpy as np

from .game import HAND, NT

_CTX = mp.get_context("spawn")


_F16 = ("ent_f", "glob", "cardf", "priv_f")


def pack_obs(o):
    """Tile masks packed to bits; float inputs rounded to float16 exactly as rollouts store
    them, so acting and training see identical inputs."""
    o.tiles = np.packbits(o.tiles, axis=-1)
    for k in _F16:
        setattr(o, k, getattr(o, k).astype(np.float16))
    return o


def unpack_obs(o):
    if o.tiles.dtype == np.uint8 and o.tiles.shape[-1] != NT:
        o.tiles = np.unpackbits(o.tiles, axis=-1, count=NT).astype(bool)
    for k in _F16:
        setattr(o, k, getattr(o, k).astype(np.float32))
    return o


def _worker(conn, n_envs: int, rcfg: dict, cores: list[int] | None):
    os.environ["OMP_NUM_THREADS"] = "1"
    try:
        if cores and hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, cores)
    except OSError:
        pass
    from .env import Battle, RewardCfg

    try:
        battles = [Battle(rcfg=RewardCfg(**rcfg)) for _ in range(n_envs)]
        cat = battles[0].cat
        for b in battles[1:]:
            b.cat = cat
        conn.send(("ready", None))
        while True:
            msg, payload = conn.recv()
            if msg == "close":
                break
            if msg == "reset":
                out = []
                for i, spec in payload:
                    obs = battles[i].reset(spec)
                    out.append([pack_obs(o) for o in obs])
                conn.send(("ok", out))
            elif msg == "step":
                out = []
                for i, (lab0, lab1, nxt) in enumerate(payload):
                    obs, rew, done, info = battles[i].step([lab0, lab1])
                    if done:
                        info["meta"] = battles[i].spec.meta
                        obs = battles[i].reset(nxt)
                        info["reset"] = True
                    out.append(([pack_obs(o) for o in obs], rew, done, info))
                conn.send(("ok", out))
    except Exception:
        try:
            conn.send(("error", traceback.format_exc()))
        except Exception:
            pass
        raise


class VecBattles:
    """``n_workers`` x ``per_worker`` battles. Battle ``k`` lives on worker ``k // per_worker``."""

    def __init__(self, n_workers: int, per_worker: int, rcfg: dict, cores: list[int] | None = None):
        self.n_workers, self.per_worker = n_workers, per_worker
        self.n = n_workers * per_worker
        self.conns, self.procs = [], []
        for w in range(n_workers):
            parent, child = _CTX.Pipe()
            mine = [cores[w % len(cores)]] if cores else None
            p = _CTX.Process(target=_worker, args=(child, per_worker, rcfg, mine), daemon=True)
            p.start()
            self.conns.append(parent)
            self.procs.append(p)
        for c in self.conns:
            self._recv(c)

    def _recv(self, c):
        if not c.poll(600):
            raise RuntimeError("env worker did not answer in 10 minutes")
        tag, payload = c.recv()
        if tag == "error":
            raise RuntimeError("env worker failed:\n" + payload)
        return payload

    def reset(self, specs: list) -> list:
        for w, c in enumerate(self.conns):
            c.send(("reset", [(i, specs[w * self.per_worker + i]) for i in range(self.per_worker)]))
        out = []
        for c in self.conns:
            out += [[unpack_obs(o) for o in pair] for pair in self._recv(c)]
        return out

    def step(self, labs: list, next_specs: list) -> list:
        """labs[k] = (lab0, lab1); returns per battle (obs_pair, rewards, done, info)."""
        for w, c in enumerate(self.conns):
            sl = range(w * self.per_worker, (w + 1) * self.per_worker)
            c.send(("step", [(labs[k][0], labs[k][1], next_specs[k]) for k in sl]))
        out = []
        for c in self.conns:
            for obs, rew, done, info in self._recv(c):
                out.append(([unpack_obs(o) for o in obs], rew, done, info))
        return out

    def close(self):
        for c in self.conns:
            try:
                c.send(("close", None))
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=5)
            if p.is_alive():
                p.kill()


def batch_obs(obs_list: list, device) -> dict:
    """List of Obs -> dict of [N, ...] tensors for ``AtlasNet.act`` (tiles unpacked)."""
    import torch

    n = len(obs_list)
    emax = max(1, max(len(o.ent_ids) for o in obs_list))
    ent_ids = np.zeros((n, emax, 3), np.int64)
    ent_f = np.zeros((n, emax, obs_list[0].ent_f.shape[-1] if len(obs_list[0].ent_f.shape) > 1 else 46), np.float32)
    ent_mask = np.zeros((n, emax), bool)
    for i, o in enumerate(obs_list):
        e = len(o.ent_ids)
        if e:
            ent_ids[i, :e] = o.ent_ids
            ent_f[i, :e] = o.ent_f
            ent_mask[i, :e] = True
    arr = {
        "glob": np.stack([o.glob for o in obs_list]).astype(np.float32),
        "cards": np.stack([o.cards for o in obs_list]).astype(np.int64),
        "cardf": np.stack([o.cardf for o in obs_list]).astype(np.float32),
        "priv_ids": np.stack([o.priv_ids for o in obs_list]).astype(np.int64),
        "priv_f": np.stack([o.priv_f for o in obs_list]).astype(np.float32),
        "cand": np.stack([o.cand for o in obs_list]),
        "ccost": np.stack([o.cand_cost for o in obs_list]).astype(np.float32),
        "elixir": np.array([o.elixir for o in obs_list], np.float32),
        "tiles": np.stack([o.tiles for o in obs_list]),
        "ent_ids": ent_ids, "ent_f": ent_f, "ent_mask": ent_mask,
    }
    return {k: torch.from_numpy(v).to(device, non_blocking=True) for k, v in arr.items()}
