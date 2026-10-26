"""Opponent league and deck sampling for A and B (shares from ab-training.md, Da 2026-10-15).

A, phase 2 (from the best A, Da 2026-10-26; shares in ``a_mix``): 20% the real FirstLight
   General on its focus decks, 10% the real FirstLight Hog 2 on Hog 2.6 (both in FirstLight env
   workers, ``flbattle``), 50% self-play and PFSP over past snapshots (A's, B's, promoted),
   20% A's stage-2 BC model and the snapshot A loses to most on the decks A loses to most.
   The FirstLight share is set by how many env workers run FirstLight (``ppo_fl_workers``);
   the simulator workers split the rest in proportion. Against FirstLight, A's deck is a
   popular deck whose cards FirstLight can name.
A, phase 1: 40% self-play, 40% PFSP over frozen opponents (A's and B's snapshots, the BC models),
   20% hard (FirstLight stand-ins, and the snapshot A loses to most on the decks A loses to most).
   A's own deck: popularity-weighted real deck from the 220k matches. Opponent decks: 65% by
   popularity, 30% FirstLight's General benchmark decks, 5% the decks A currently loses to most.
B: always Hog 2.6 (hero Musketeer, evo Cannon and Skeletons). 10% Hog mirror self-play, 70% A's
   checkpoints and B's past selves on popularity decks, 20% hard (FL-Hog stand-in in the mirror,
   FL-General stand-in on its decks, decks B loses to most).
PFSP weight (1 - win rate)^2 with a 20% uniform floor. Win rates are exponential averages.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np

from .common import FL_REPO, LEAGUE, RUNS, SHARDS, log, read_json
from .game import HOG26, deck_key, parse_deck_key

EMA = 0.03


def focus_decks(cat) -> list[str]:
    """FirstLight's 20 focus decks (its General benchmark pool), as our deck keys."""
    path = FL_REPO / "native_runner" / "training" / "v4" / "focus_decks_20.json"
    if not path.exists():
        path = Path(__file__).with_name("focus_decks_20.json")
    try:
        from royaleimitate.replays import CARD_NAMES, base_key
        from .game import sanitize_forms
        from .replay import _form

        rows = json.loads(path.read_text())["decks"]
        out = []
        for r in rows:
            deck = [(CARD_NAMES.get(base_key(k)), _form(k)) for k in r["card_keys"]]
            if all(n in cat.ids for n, _ in deck) and len({n for n, _ in deck}) == 8:
                out.append(deck_key(sanitize_forms(cat, deck)))
        return out
    except Exception as exc:
        log(f"FirstLight focus decks unavailable: {exc!r}")
        return []


def prune_snapshots(run: str, keep_latest: int, keep_best: int) -> list[str]:
    """Keep the newest ``keep_latest`` snapshots of ``run`` plus the ``keep_best`` with the best
    FirstLight result (promotion games when played, else the screen's primary suite); delete the
    rest. Returns the deleted file names."""
    snaps = sorted((LEAGUE / run).glob("snap_*.pt"))
    if len(snaps) <= keep_latest + keep_best:
        return []
    scores: dict[str, float] = {}
    try:
        with open(RUNS / "eval" / f"{run}.jsonl", encoding="utf-8") as fh:
            for line in fh:
                d = json.loads(line)
                prim = d.get("primary") or (d.get("suites") or [None])[0]
                if "incomplete" in d or prim not in d:
                    continue
                score = (d.get("promotion") or d[prim])["win_rate"]
                name = Path(d["path"]).name
                scores[name] = max(score, scores.get(name, 0.0))
    except OSError:
        pass
    keep = set(snaps[-keep_latest:])
    ranked = sorted((s for s in snaps if s not in keep and s.name in scores), key=lambda s: -scores[s.name])
    keep.update(ranked[:keep_best])
    gone = []
    for s in snaps:
        if s not in keep:
            s.unlink(missing_ok=True)
            gone.append(s.name)
    return gone


class DeckBook:
    def __init__(self, cat, rng: np.random.Generator, top: int = 3000):
        idx = read_json(SHARDS.parent / "index.json", {}) or {}
        decks = idx.get("decks", {})
        keys = [k for k in list(decks)[:top] if all(n in cat.ids for n, _ in parse_deck_key(k))]
        if not keys:
            keys = [deck_key(HOG26)]
            w = np.ones(1)
        else:
            w = np.array([decks[k] for k in keys], np.float64)
        self.pop_keys, self.pop_w = keys, w / w.sum()
        self.focus = focus_decks(cat) or keys[:20]
        self.hog = deck_key(HOG26)
        self.cat_ids = cat.ids
        self.rng = rng
        self.stats: dict[str, list[float]] = {}  # deck -> [ema win rate, games]

    def popular(self) -> str:
        return self.pop_keys[int(self.rng.choice(len(self.pop_keys), p=self.pop_w))]

    def popular_known(self, known) -> str:
        """A popularity-weighted deck whose cards are all in ``known`` (RoyaleSim card ids),
        e.g. the cards FirstLight can name."""
        from .game import parse_deck_key

        d = self.popular()
        for _ in range(200):
            if all(self.cat_ids[n] in known for n, _ in parse_deck_key(d)):
                return d
            d = self.popular()
        return self.hog

    def focus_deck(self) -> str:
        return self.focus[int(self.rng.integers(len(self.focus)))]

    def hard(self) -> str:
        cands = [(v[0], k) for k, v in self.stats.items() if v[1] >= 20]
        if len(cands) < 5:
            return self.popular()
        cands.sort()
        worst = [k for _, k in cands[:10]]
        return worst[int(self.rng.integers(len(worst)))]

    def report(self, opp_deck: str, score: float):
        s = self.stats.setdefault(opp_deck, [0.5, 0])
        s[0] += max(EMA, 1.0 / (s[1] + 1)) * (score - s[0])
        s[1] += 1

    def dealt(self, key: str) -> list:
        """A deck key -> 8 (name, form) in a random deal order."""
        deck = parse_deck_key(key)
        order = self.rng.permutation(8)
        return [deck[i] for i in order]


class League:
    def __init__(self, run: str, cat, seed: int, mix: dict | None = None):
        self.run = run
        self.mix = mix  # A phase 2 shares (config ``a_mix``); None = phase 1
        self.fl_known: frozenset | None = None
        self.rng = np.random.default_rng(seed)
        self.decks = DeckBook(cat, self.rng)
        self.opps: dict[str, dict] = {}  # id -> {path, kind, wr, games}: the active pool
        self.last_scan = 0.0
        self.state_path = RUNS / run / "league.json"
        saved = read_json(self.state_path, {}) or {}
        self.decks.stats = saved.get("decks", {})
        self._saved_opps = saved.get("opps", {})
        self.refresh(force=True)

    # ---- pool on disk
    def refresh(self, force: bool = False):
        if not force and time.time() - self.last_scan < 300:
            return
        self.last_scan = time.time()
        found = {}
        for run in ("A", "B"):
            for p in sorted((LEAGUE / run).glob("snap_*.pt")):
                found[f"snap:{run}:{p.stem[5:]}"] = (p, "snap", run)
        for p in sorted((LEAGUE / "standins").glob("*.pt")):
            found[f"standin:{p.stem}"] = (p, "standin", None)
        for p in sorted((LEAGUE / "bc").glob("*.pt")):
            found[f"bc:{p.stem}"] = (p, "bc", None)
        for p in sorted((LEAGUE / "promoted").glob("*.pt")):
            found[f"promoted:{p.stem}"] = (p, "promoted", None)
        # bounded pool of past selves per run: the newest 12, 12 spread over the older history,
        # and any older one we still lose to (win rate <= 0.3)
        keep = set(k for k in found if not k.startswith("snap:"))
        for run in ("A", "B"):
            snaps = sorted(k for k in found if k.startswith(f"snap:{run}:"))
            old = snaps[:-12]
            keep.update(snaps[-12:])
            if old:
                keep.update(old[int(i)] for i in np.linspace(0, len(old) - 1, min(12, len(old))))
            keep.update(k for k in old if self._stat(k)["wr"] <= 0.3 and self._stat(k)["games"] >= 10)
        for k in list(self.opps):
            if k not in keep:
                self._saved_opps[k] = {"wr": self.opps[k]["wr"], "games": self.opps[k]["games"]}
                del self.opps[k]
        for k in keep:
            if k not in self.opps:
                p, kind, run = found[k]
                prev = self._stat(k)
                self.opps[k] = {"path": str(p), "kind": kind, "run": run, "wr": prev["wr"], "games": prev["games"]}

    def _stat(self, k: str) -> dict:
        if k in self.opps:
            return self.opps[k]
        prev = self._saved_opps.get(k, {})
        return {"wr": prev.get("wr", 0.5), "games": prev.get("games", 0)}

    def save(self):
        from .common import atomic_write_json

        opps = dict(self._saved_opps)
        opps.update({k: {"wr": v["wr"], "games": v["games"]} for k, v in self.opps.items()})
        atomic_write_json(self.state_path, {"decks": self.decks.stats, "opps": opps})

    def _pfsp(self, keys: list[str]) -> str | None:
        if not keys:
            return None
        if self.rng.random() < 0.2:
            return keys[int(self.rng.integers(len(keys)))]
        w = np.array([(1.0 - self.opps[k]["wr"]) ** 2 + 1e-3 for k in keys])
        return keys[int(self.rng.choice(len(keys), p=w / w.sum()))]

    def _hardest(self, keys: list[str]) -> str | None:
        rated = [(self.opps[k]["wr"], k) for k in keys if self.opps[k]["games"] >= 10]
        if not rated:
            return self._pfsp(keys)
        rated.sort()
        return rated[0][1]

    def next_match(self, fl: bool = False) -> dict:
        """-> {"opp": id or "self", "decks": [key seat-learner, key opponent], "learner_seat": 0/1}.
        ``fl``: a match for a FirstLight env worker (opponent ``fl:general`` / ``fl:hog2``)."""
        self.refresh()
        if self.run == "A" and self.mix is not None:
            return self._next_a2(fl)
        r = self.rng.random()
        standins = [k for k in self.opps if k.startswith("standin:")]
        stand_gen = [k for k in standins if "general" in k]
        stand_hog = [k for k in standins if "hog" in k]
        d = self.decks
        if self.run == "A":
            frozen = [k for k in self.opps if self.opps[k]["kind"] in ("snap", "bc", "promoted")]
            own = d.popular()
            if r < 0.4 or not frozen:
                opp = "self"
            elif r < 0.8:
                opp = self._pfsp(frozen)
            else:
                pick = self.rng.random()
                if stand_gen and pick < 0.5:
                    return self._m(stand_gen[0], own, d.focus_deck())
                if stand_hog and pick < 0.75:
                    return self._m(stand_hog[0], own, d.hog)
                return self._m(self._hardest(frozen), own, d.hard())
            q = self.rng.random()
            opp_deck = d.popular() if q < 0.65 else d.focus_deck() if q < 0.95 else d.hard()
            return self._m(opp, own, opp_deck)
        # B: Hog 2.6 always
        own = d.hog
        pool = [k for k in self.opps if (self.opps[k]["kind"] in ("snap", "promoted"))
                or k in ("bc:A2", "bc:A1")]
        if r < 0.1 or not pool:
            return self._m("self", own, d.hog)
        if r < 0.8:
            return self._m(self._pfsp(pool), own, d.popular())
        pick = self.rng.random()
        if stand_hog and pick < 0.4:
            return self._m(stand_hog[0], own, d.hog)
        if stand_gen and pick < 0.8:
            return self._m(stand_gen[0], own, d.focus_deck())
        return self._m(self._hardest(pool), own, d.hard())

    def _next_a2(self, fl: bool) -> dict:
        mix, d = self.mix, self.decks
        if fl:
            if self.fl_known is None:
                from .flmatch import _fl_known_cards

                self.fl_known = _fl_known_cards()
            own = d.popular_known(self.fl_known)
            g, h = mix["fl_general"], mix["fl_hog2"]
            if self.rng.random() < g / max(1e-9, g + h):
                return self._m("fl:general", own, d.focus_deck())
            return self._m("fl:hog2", own, d.hog)
        past = [k for k in self.opps if self.opps[k]["kind"] in ("snap", "promoted")]
        bc = [k for k in ("bc:A2",) if k in self.opps]
        own = d.popular()
        w = np.array([mix["self"], mix["past"] if past else 0.0, mix["bc"] if bc else 0.0,
                      mix["hard"] if past else 0.0])
        pick = int(self.rng.choice(4, p=w / w.sum()))
        if pick == 3:
            return self._m(self._hardest(past), own, d.hard())
        opp = ("self", self._pfsp(past) if past else None, bc[0] if bc else None)[pick]
        q = self.rng.random()
        return self._m(opp, own, d.popular() if q < 0.65 else d.focus_deck() if q < 0.95 else d.hard())

    def _m(self, opp: str | None, own: str, opp_deck: str) -> dict:
        if opp is None:
            opp = "self"
        return {"opp": opp, "decks": [own, opp_deck], "learner_seat": int(self.rng.integers(2))}

    def report(self, opp: str, opp_deck: str, score: float):
        """score from the learner's side: 1 win, 0.5 draw, 0 loss."""
        if opp in self.opps:
            o = self.opps[opp]
            o["wr"] += max(EMA, 1.0 / (o["games"] + 1)) * (score - o["wr"])
            o["games"] += 1
        self.decks.report(opp_deck, score)

    def summary(self) -> dict:
        return {k: (round(v["wr"], 3), v["games"]) for k, v in sorted(self.opps.items())}
