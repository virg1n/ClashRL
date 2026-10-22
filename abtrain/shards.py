"""Shard format, writer, and the window collate used by every trainer.

A shard (``*.npz``, compressed) holds whole matches, both seats, one row per 250 ms decision:

  seq_off [S+1] i64            step offsets of each sequence (one seat of one match)
  seq_meta [S, 8] i32          match_hash, seat, outcome (+1/0/-1), truncated, is_hog (this seat),
                               any_hog (either seat), outcome_source (0 sim, 1 real), holdout
  decks [S] str                the seat's deck key (name:form|...)
  glob [T, G] f16, cards [T, 16] i16, cardf [T, 16, CF] f16, priv_ids [T, 13] i16,
  priv_f [T, PF] f16, cand [T] u8 (bit k = candidate k legal), ccost [T, 7] u8,
  ent_off [T+1] i64, ent_ids [E, 3] i16, ent_f [E, F] f16,
  elx [T] i16 (exact own elixir in milli-elixir, for the second-play affordability rule),
  lab [T, 6] i16 (c1, t1, d1, c2, t2, d2), tm [T, 2] i32 (row in ``tmask`` or -1),
  tmask [M, 72] u8 (packed 576-tile legality of the labelled candidate)

PPO rollouts use the same step fields in memory, plus their own (logp, value, reward, ...).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .game import CARD_ROWS, CF, F, G, HAND, MAX_ENT, NCAND, NT, PF, PRIV_IDS

STEP_KEYS = ("glob", "cards", "cardf", "priv_ids", "priv_f", "cand", "ccost", "elx", "lab", "tm")
_CAND_BITS = (1 << np.arange(NCAND)).astype(np.uint8)


def pack_rows(rows, labs) -> dict:
    """Obs list + label list of one sequence -> step arrays (entities with local offsets)."""
    t = len(rows)
    glob = np.stack([r.glob for r in rows]).astype(np.float16)
    cards = np.stack([r.cards for r in rows]).astype(np.int16)
    cardf = np.stack([r.cardf for r in rows]).astype(np.float16)
    priv_ids = np.stack([r.priv_ids for r in rows]).astype(np.int16)
    priv_f = np.stack([r.priv_f for r in rows]).astype(np.float16)
    cand = np.stack([(r.cand.astype(np.uint8) * _CAND_BITS).sum() for r in rows]).astype(np.uint8)
    ccost = np.clip(np.stack([r.cand_cost for r in rows]), 0, 255).astype(np.uint8)
    elx = np.array([round(r.elixir * 1000) for r in rows], np.int16)
    ent_n = np.array([min(len(r.ent_ids), MAX_ENT) for r in rows], np.int64)
    ent_off = np.concatenate([[0], np.cumsum(ent_n)]).astype(np.int64)
    ent_ids = np.concatenate([r.ent_ids[:MAX_ENT] for r in rows]).astype(np.int16).reshape(-1, 3)
    ent_f = np.concatenate([r.ent_f[:MAX_ENT] for r in rows]).astype(np.float16).reshape(-1, F)
    lab = np.asarray(labs, np.int16).reshape(t, 6)
    tm = np.full((t, 2), -1, np.int32)
    masks = []
    for i, (r, lb) in enumerate(zip(rows, lab)):
        for j in (0, 1):
            c = int(lb[3 * j])
            if 0 <= c < HAND:
                m = r.tiles[c].copy()  # the real legal tiles; collation checks the label against them
                tm[i, j] = len(masks)
                masks.append(np.packbits(m))
    tmask = np.stack(masks).astype(np.uint8) if masks else np.zeros((0, NT // 8), np.uint8)
    return dict(glob=glob, cards=cards, cardf=cardf, priv_ids=priv_ids, priv_f=priv_f, cand=cand,
                ccost=ccost, elx=elx, ent_off=ent_off, ent_ids=ent_ids, ent_f=ent_f, lab=lab, tm=tm, tmask=tmask)


def concat_sequences(seqs: list[dict]) -> dict:
    """Concatenate packed sequences into one store with global offsets."""
    out = {}
    lens = [len(s["lab"]) for s in seqs]
    out["seq_off"] = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    for k in ("glob", "cards", "cardf", "priv_ids", "priv_f", "cand", "ccost", "elx", "lab"):
        out[k] = np.concatenate([s[k] for s in seqs])
    ent_n = np.concatenate([np.diff(s["ent_off"]) for s in seqs])
    out["ent_off"] = np.concatenate([[0], np.cumsum(ent_n)]).astype(np.int64)
    out["ent_ids"] = np.concatenate([s["ent_ids"] for s in seqs])
    out["ent_f"] = np.concatenate([s["ent_f"] for s in seqs])
    tms, base = [], 0
    for s in seqs:
        tm = s["tm"].copy()
        tm[tm >= 0] += base
        tms.append(tm)
        base += len(s["tmask"])
    out["tm"] = np.concatenate(tms)
    out["tmask"] = np.concatenate([s["tmask"] for s in seqs]) if base else np.zeros((0, NT // 8), np.uint8)
    return out


class ShardWriter:
    def __init__(self, out_dir: Path, prefix: str, per_shard: int):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.prefix, self.per_shard = prefix, per_shard
        self.n = 0
        self._reset()

    def _reset(self):
        self.seqs, self.meta, self.decks, self.matches = [], [], [], 0

    def add_match(self, items: list[tuple[dict, list[int], str]]) -> None:
        for packed, meta, deck in items:
            self.seqs.append(packed)
            self.meta.append(meta)
            self.decks.append(deck)
        self.matches += 1
        if self.matches >= self.per_shard:
            self.flush()

    def flush(self) -> Path | None:
        if not self.seqs:
            return None
        store = concat_sequences(self.seqs)
        store["seq_meta"] = np.asarray(self.meta, np.int32).reshape(-1, 8)
        store["decks"] = np.asarray(self.decks)
        name = self.dir / f"{self.prefix}_{self.n:03d}.npz"
        tmp = self.dir / f".tmp_{self.prefix}_{self.n:03d}_{os.getpid()}.npz"
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, **store)
        os.replace(tmp, name)
        self.n += 1
        self._reset()
        return name


def load_shard(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def unpack_cand(bits: np.ndarray) -> np.ndarray:
    return (bits[..., None] & _CAND_BITS) > 0


def collate(store: dict, starts: np.ndarray, length: int, seq_end: np.ndarray, loss_from: np.ndarray,
            rng: np.random.Generator | None = None, feature_drop: float = 0.0,
            extra: dict | None = None, loss_to: np.ndarray | None = None) -> dict:
    """Windows ``[starts[i], starts[i] + length)`` (clipped at ``seq_end[i]``) -> [B, L] arrays.

    ``loss`` marks steps >= ``loss_from[i]`` that exist. ``extra`` maps name -> per-step array
    gathered the same way (advantages, rewards, ...). ``feature_drop`` hides projectile tokens
    and target links at random (the live bridge can miss them)."""
    b = len(starts)
    ar = np.arange(length)
    idx = starts[:, None] + ar[None]
    valid = idx < seq_end[:, None]
    idx = np.where(valid, idx, starts[:, None])
    out = {
        "valid": valid,
        "loss": valid & (idx >= loss_from[:, None]) & (idx < (loss_to if loss_to is not None else seq_end)[:, None]),
        "glob": store["glob"][idx].astype(np.float32),
        "cards": store["cards"][idx].astype(np.int64),
        "cardf": store["cardf"][idx].astype(np.float32),
        "priv_ids": store["priv_ids"][idx].astype(np.int64),
        "priv_f": store["priv_f"][idx].astype(np.float32),
        "cand": unpack_cand(store["cand"][idx]),
        "ccost": store["ccost"][idx].astype(np.float32),
        "elixir": (store["elx"][idx].astype(np.float32) / 1000.0 if "elx" in store  # exact
                   # older shards: float16 elixir, lowered by its largest rounding error
                   else store["glob"][idx][..., 0].astype(np.float32) * 10.0 - 0.0025),
        "lab": store["lab"][idx].astype(np.int64),
    }
    lab = out["lab"]
    # steps marked SKIP (-2) by extraction (a play the action space cannot express at the window
    # start) carry no policy target
    skip = lab[..., 0] == -2
    if skip.any():
        out["loss"] &= ~skip
        lab[skip] = -1
    for j, key in ((0, "tm1"), (1, "tm2")):
        rows = store["tm"][idx, j]
        m = np.zeros((b, length, NT), bool)
        has = rows >= 0
        if has.any():
            m[has] = np.unpackbits(store["tmask"][rows[has]], axis=-1, count=NT).astype(bool)
        out[key] = m
    # labels outside the policy's action support (extraction checks this; older or foreign
    # shards may not) are excluded from the loss, never made legal: index bounds, allowed
    # candidates and tiles, the second play distinct, ordered and affordable with the first paid
    bad = invalid_labels(lab, out["cand"], out["ccost"], out["elixir"], out["tm1"], out["tm2"])
    if bad.any():
        out["loss"] &= ~bad
        lab[bad] = -1
    out["invalid"] = bad
    lab[~valid] = -1
    out["lab"] = lab
    # entities
    e0 = store["ent_off"][idx]
    en = np.minimum(store["ent_off"][idx + 1] - e0, MAX_ENT)
    en[~valid] = 0
    emax = max(1, int(en.max()))
    flat_n = en.ravel()
    total = int(flat_n.sum())
    ent_ids = np.zeros((b * length, emax, 3), np.int64)
    ent_f = np.zeros((b * length, emax, F), np.float32)
    ent_mask = np.zeros((b * length, emax), bool)
    if total:
        step_of = np.repeat(np.arange(b * length), flat_n)
        first = np.repeat(np.cumsum(flat_n) - flat_n, flat_n)
        j = np.arange(total) - first
        src = np.repeat(e0.ravel(), flat_n) + j
        ent_ids[step_of, j] = store["ent_ids"][src]
        ent_f[step_of, j] = store["ent_f"][src]
        ent_mask[step_of, j] = True
        if feature_drop > 0 and rng is not None:
            kinds = ent_ids[step_of, j, 2]
            drop = (kinds == 5) & (rng.random(total) < feature_drop)
            ent_mask[step_of[drop], j[drop]] = False
            cut = rng.random(total) < feature_drop
            for col in (16, 17, 18, 19, 20, 39, 40, 41, 45):
                ent_f[step_of[cut], j[cut], col] = 0.0
    out["ent_ids"] = ent_ids.reshape(b, length, emax, 3)
    out["ent_f"] = ent_f.reshape(b, length, emax, F)
    out["ent_mask"] = ent_mask.reshape(b, length, emax)
    if extra:
        for k, arr in extra.items():
            out[k] = arr[idx]
    return out


def invalid_labels(lab: np.ndarray, cand: np.ndarray, ccost: np.ndarray, elixir: np.ndarray,
                   tm1: np.ndarray | None = None, tm2: np.ndarray | None = None) -> np.ndarray:
    """[..., 6] labels -> bool mask of the steps the policy could not have produced. ``tm1`` /
    ``tm2`` are the legal-tile masks of the labelled first / second candidate."""
    from .game import NCAND, NDELAY, SECOND_PLAY_SLACK

    c1, t1, d1, c2, t2, d2 = (lab[..., k] for k in range(6))
    has1, has2 = c1 >= 0, c2 >= 0
    take = lambda a, c: np.take_along_axis(a, np.clip(c, 0, a.shape[-1] - 1)[..., None], -1)[..., 0]  # noqa: E731

    def one(c, t, d, tm):
        out = (c >= NCAND) | (d < 0) | (d >= NDELAY) | ~take(cand, c)
        card = c < HAND
        out |= card & ((t < 0) | (t >= NT))  # a card needs a tile, a button press none
        out |= ~card & (t >= 0)
        if tm is not None:
            out |= card & (t >= 0) & (t < NT) & ~take(tm, t)
        return out

    bad = has1 & one(c1, t1, d1, tm1)
    left = elixir.astype(np.float32) - take(ccost, c1)
    bad2 = one(c2, t2, d2, tm2) | (c2 == c1) | (d2 < d1) | (take(ccost, c2) > left + SECOND_PLAY_SLACK) | ~has1
    return bad | (has2 & bad2) | (c1 < -1) | (c2 < -1)


def shard_list(root: Path) -> list[Path]:
    return sorted(p for p in Path(root).rglob("*.npz") if not p.name.startswith("."))
