"""Matches with the real FirstLight models (through ``flport``), tick by tick in RoyaleSim.

Three uses:
  eval      our checkpoint vs FirstLight General (or Hog 2) on an eval suite
  baseline  FirstLight Hog 2 vs FirstLight General under the same protocol (the bar to beat)
  record    FirstLight vs FirstLight games written as training shards for the stand-ins

    python -m abtrain.flmatch eval --ckpt RUNS/ppo_A/final.pt --suite hog26 --games 40 --out x.json
    python -m abtrain.flmatch baseline --games 100 --out RUNS/eval/fl_baseline.json
    python -m abtrain.flmatch record --workers 20 --games 1200

Eval protocol (FirstLight's own, ``docs/training_history.md``): argmax FirstLight, no forced
first action, no command delay, seats alternate in pairs with the same seed and decks.
Suites: ``hog26`` = both seats on the Hog 2.6 deck with its forms (FL Hog 2 scored 88% vs
General here in FirstLight's engine); ``mixed`` = both seats on the same FirstLight focus deck
(FL Hog 2 scored 59% vs General on mixed decks).

Recording: the FirstLight seat's plays inside each 250 ms window become our labels, exactly
as human replays do (card slot at the window start, tile, delay tick, second play). Half of
the recorded games use FirstLight's argmax as benchmarked, half sample with a forced first
play after 15 s, so the stand-ins are not only passive.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .common import FL_PORT, FL_REPO, FL_SHARDS, atomic_write_json, log, read_json, settings, wilson

FL_CKPTS = {
    "hog2": "checkpoints/2_6hog_expert/hog26-specialist2.pt",
    "hog1": "checkpoints/2_6hog_expert/hog26-specialist1.pt",
    "general": "checkpoints/General/checkpoint-step-00000460.pt",
    "il": "checkpoints/IL/checkpoint-step-00029396.pt",
}
MAX_TICKS = int(os.environ.get("AB_FL_MAX_TICKS", 7200))  # lower only for quick tests
FORCE_TICK = 300  # FirstLight's historical fallback: one forced first action after 15 s
_CTX = mp.get_context("spawn")
_W: dict = {}


def fl_ready() -> bool:
    return all((FL_REPO / p).exists() for p in FL_CKPTS.values()) and (FL_PORT / "flport").exists()


def _paths():
    for p in (str(FL_REPO), str(FL_PORT)):
        if p not in sys.path:
            sys.path.insert(0, p)


# ----------------------------------------------------------------------------- worker side


def _init(counter, gpus: list[int]):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    _paths()
    import torch

    torch.set_num_threads(1)
    with counter.get_lock():
        i = counter.value
        counter.value += 1
    if gpus and torch.cuda.is_available():
        _W["device"] = f"cuda:{gpus[i % len(gpus)]}"
    else:
        _W["device"] = "cpu"
    from flport.catalog import Catalog as FLCatalog
    from native_runner.training.v4.factory import production_semantic_bundle

    from .game import Catalog, make_engine, make_parser

    eng = make_engine()
    _W["engine"] = eng
    _W["parser"] = make_parser(eng)
    _W["cat"] = Catalog.build(eng)
    _W["flcat"] = FLCatalog.build(eng)
    _W["bundle"] = production_semantic_bundle()
    _W["ours"] = {}


def _ours(path: str):
    if path not in _W["ours"]:
        from .model import load_policy

        if len(_W["ours"]) >= 4:
            _W["ours"].clear()
        _W["ours"][path] = load_policy(path, _W["device"])
    return _W["ours"][path]


def _fl_policy(name: str):
    from flport.agent import load

    path = FL_CKPTS.get(name, name)
    p = Path(path)
    if not p.is_absolute():
        p = FL_REPO / p
    return load(str(p), _W["device"])


class OurSeat:
    """One of our checkpoints on one seat: decides every 5 ticks, sends each play at its delay tick."""

    def __init__(self, team: int, model, view, decode: str):
        import torch

        from .model import DECODES

        self.team, self.model, self.view = team, model, view
        self.dec = DECODES[decode]
        self.dev = next(model.parameters()).device
        self.state = model.initial_state(1, self.dev)
        self.queue: list[tuple[int, object]] = []
        self.stats = {"plays": 0, "first_play_tick": None, "curve": []}  # curve: (tick, P(win), P(loss))
        self.torch = torch

    def act(self, tick: int, state) -> list:
        from .game import DT, HAND, command, legal_masks
        from .vec import batch_obs

        if tick % DT == 0:
            cand, tiles = legal_masks(_W["parser"], state, self.team)
            o = self.view.build(state, cand, tiles)
            for k in ("ent_f", "glob", "cardf", "priv_f"):  # the float16 rounding training saw
                setattr(o, k, getattr(o, k).astype(np.float16).astype(np.float32))
            with self.torch.no_grad(), self.torch.autocast(device_type=self.dev.type, dtype=self.torch.bfloat16,
                                                           enabled=self.dev.type == "cuda"):
                lab, _lp, _v, vl, self.state = self.model.act(batch_obs([o], self.dev), self.state, **self.dec)
            pr = vl[0].float().softmax(-1).tolist()
            self.stats["curve"].append((tick, pr[0], pr[2], [int(p.crowns) for p in state.players]))
            lab = lab[0].tolist()
            for j in (0, 1):
                c, t, d = lab[3 * j : 3 * j + 3]
                if c < 0 or (c < HAND and t < 0):
                    continue
                self.queue.append((tick + max(0, min(int(d), DT - 1)), command(self.team, c, t)))
        out = [c for due, c in self.queue if due <= tick]
        self.queue = [(due, c) for due, c in self.queue if due > tick]
        return out

    def end(self):
        pass


class FLSeat:
    """flport's FirstLightSeat, optionally with FirstLight's forced first action."""

    def __init__(self, team: int, name: str, scene, sample: bool, force_first: bool):
        from flport.agent import FirstLightSeat

        self.seat = FirstLightSeat(team, _fl_policy(name), scene, sample=sample, device=_W["device"])
        # FirstLight's batch-1 CUDA graph is shared per model and device inside a process; one
        # failed capture (seen with torch 2.14 / CUDA 13) leaves it half-built and every later
        # decision raises "V4 CUDA Graph storage is incomplete". The plain forward pass makes the
        # same argmax decisions.
        self.seat.session._cuda_graph = None
        self.team = team
        self.force_first = force_first
        self.stats = self.seat.stats
        if force_first:
            orig = self.seat.session.decide
            self._tick = 0

            def decide(observation, force_act=False):
                force = self.seat.stats["plays"] == 0 and self._tick >= FORCE_TICK
                return orig(observation, force_act=force)

            self.seat.session.decide = decide

    def act(self, tick: int, state) -> list:
        self._tick = tick
        return self.seat.act(tick)

    def end(self):
        self.seat.end()


def play_game(task: dict) -> dict:
    """task: seed, decks [key0, key1], sides [side0, side1] with side = {"kind": "ours", "path",
    "decode"} or {"kind": "fl", "name", "sample", "force_first", "record": tag or None}."""
    from flport.observation import Scene
    from royalegym.protocol import HAND_SIZE, DeployStatus, MatchSetup, ShuffleMode

    from .game import DT, HAND, SeatView, label_supported, legal_masks, nearest_legal_tile, own_tile, parse_deck_key
    from .replay import SeqBuf

    eng, parser, cat = _W["engine"], _W["parser"], _W["cat"]
    seed = int(task["seed"])
    r = np.random.default_rng(seed)
    decks, forms = [], []
    for t in (0, 1):
        dk = parse_deck_key(task["decks"][t])
        dk = [dk[i] for i in r.permutation(8)]
        decks.append([cat.ids[n] for n, _ in dk])
        forms.append([f for _, f in dk])
    if hasattr(eng, "set_command_delay_ticks"):
        eng.set_command_delay_ticks(0)
    setup = MatchSetup(decks=decks, forms=forms, shuffle=ShuffleMode.NONE)
    eng.reset(seed, setup)
    rec_trace = None
    if task.get("trace"):  # a RoyaleGym trace for royaleviser / render_video.py, one frame per tick
        from royalegym.replay import ReplayRecorder

        rec_trace = ReplayRecorder(frame_every_tick=True)
        rec_trace.begin(eng, seed, setup)
    state = eng.state()
    scene = Scene(_W["flcat"], _W["bundle"], decks=decks, forms=forms, arena=eng.arena(), episode_id=f"ab-{seed}")
    scene.ingest(state)
    views = [None, None]
    seats = []
    recs = {}
    for t in (0, 1):
        sd = task["sides"][t]
        if sd["kind"] == "ours" or sd.get("record"):
            views[t] = SeatView(cat, parser)
            views[t].reset(t, decks[t], forms[t], decks[1 - t])
        if sd["kind"] == "ours":
            seats.append(OurSeat(t, _ours(sd["path"]), views[t], sd.get("decode", "t05")))
        else:
            seats.append(FLSeat(t, sd["name"], scene, bool(sd.get("sample")), bool(sd.get("force_first"))))
            if sd.get("record"):
                recs[t] = {"buf": SeqBuf(), "win": None}
    rejected = [0, 0]
    t0 = time.time()
    while not state.game_over and state.tick < MAX_TICKS:
        tick = int(state.tick)
        if tick % DT == 0:
            for t, rec in recs.items():
                cand, tiles = legal_masks(parser, state, t)
                rec["win"] = {"obs": views[t].build(state, cand, tiles), "tiles": tiles, "tick": tick,
                              "hand": list(state.players[t].hand), "cmds": []}
        commands = []
        for seat in seats:
            commands.extend(seat.act(tick, state))
        evo_ready = [{int(x[0]) for x in state.players[t].evo if len(x) > 2 and x[2]} for t in (0, 1)]
        hands = [list(state.players[t].hand) for t in (0, 1)]
        results = eng.step(commands, 1)
        if rec_trace is not None:
            rec_trace.record_frame(eng)
            rec_trace.record_step(tick, 1, commands, results)
        played = []
        for cmd, res in zip(commands, results):
            if res.status != DeployStatus.OK:
                rejected[cmd.team] += 1
                continue
            if cmd.hand_slot >= HAND_SIZE:
                played.append((res.tick, cmd.team, -1 - (cmd.hand_slot - HAND_SIZE), 0, 0, 0))
                card = -1
            else:
                card = hands[cmd.team][cmd.hand_slot]
                form = scene.form_of(cmd.team, card)
                code = 2 if form == 2 else (1 if card in evo_ready[cmd.team] else 0)
                played.append((res.tick, cmd.team, card, code, res.x, res.y))
                for v in views:
                    if v is not None:
                        v.note_play(cmd.team, card)
                if isinstance(seats[cmd.team], OurSeat):
                    st = seats[cmd.team].stats
                    st["plays"] += 1
                    if st["first_play_tick"] is None:
                        st["first_play_tick"] = tick
            rec = recs.get(cmd.team)
            if rec is not None and rec["win"] is not None:
                rec["win"]["cmds"].append((tick, int(cmd.hand_slot), card, int(cmd.x), int(cmd.y)))
        state = eng.state()
        scene.ingest(state, played)
        if (tick + 1) % DT == 0 or state.game_over:
            for t, rec in recs.items():
                w = rec["win"]
                if w is None:
                    continue
                lab = [-1] * 6
                n = 0
                used = set()
                skip = False
                for ctick, slot, card, x, y in sorted(w["cmds"])[:2]:
                    if slot >= HAND:
                        c, tl = slot, -1
                    else:
                        tl = -1
                        if card in w["hand"]:
                            c = w["hand"].index(card)
                            tl = nearest_legal_tile(w["tiles"][c], own_tile(t, x, y))
                        if tl < 0:  # not expressible at the window start: keep it out of the loss
                            skip = True
                            break
                    if c in used:
                        continue
                    used.add(c)
                    lab[3 * n : 3 * n + 3] = [c, tl, max(0, min(ctick - w["tick"], DT - 1))]
                    n += 1
                rec["buf"].rows.append(w["obs"])
                skip = skip or not label_supported(w["obs"], lab)
                rec["buf"].lab.append([-2, -1, -1, -1, -1, -1] if skip else lab)
                rec["win"] = None
    for seat in seats:
        seat.end()
    w = int(state.winner) if state.game_over else 2
    plays = []
    first = []
    for t, seat in enumerate(seats):
        if isinstance(seat, OurSeat):
            plays.append(seat.stats["plays"])
            first.append(seat.stats["first_play_tick"])
        else:
            plays.append(seat.stats["plays"] + seat.stats.get("abilities", 0))
            first.append(seat.stats.get("first_play_tick"))
    if rec_trace is not None:
        from royalegym.replay import save_trace

        rec_trace.end(eng)
        Path(task["trace"]).parent.mkdir(parents=True, exist_ok=True)
        save_trace(rec_trace.trace, task["trace"])
    curves = {t: seat.stats["curve"] for t, seat in enumerate(seats) if isinstance(seat, OurSeat)}
    out = {"seed": seed, "winner": w, "ticks": int(state.tick), "crowns": [state.players[0].crowns, state.players[1].crowns],
           "curves": curves,
           "plays": plays, "first_play": first, "rejected": rejected, "seconds": time.time() - t0,
           "meta": task.get("meta", {})}
    if recs:
        out["recorded"] = {t: rec["buf"] for t, rec in recs.items()}
    return out


def _safe_game(task: dict) -> dict:
    try:
        return play_game(task)
    except Exception as exc:  # one broken game never stops an evaluation
        return {"error": repr(exc), "trace": traceback.format_exc()[-1500:], "meta": task.get("meta", {})}


# ----------------------------------------------------------------------------- pool


class FLPool:
    """Worker processes with the engine, flport and the FirstLight models loaded once."""

    def __init__(self, workers: int, gpus: list[int] | None = None):
        counter = _CTX.Value("i", 0)
        self.ex = ProcessPoolExecutor(max_workers=workers, mp_context=_CTX, initializer=_init,
                                      initargs=(counter, list(gpus or [])))
        self.workers = workers

    def run(self, tasks: list[dict], fn=None, progress=None, retries: int = 2) -> list[dict]:
        """Every task's result. A game that failed is played again with the same task (seed,
        seats, decks) up to ``retries`` times; one still failing comes back as an error, so a
        summary never silently drops it from the denominator."""
        out: dict[int, dict] = {}
        pending = list(range(len(tasks)))
        for attempt in range(retries + 1):
            futs = {self.ex.submit(fn or _safe_game, tasks[i]): i for i in pending}
            for f in as_completed(futs):
                out[futs[f]] = f.result()
                if progress:
                    progress(len(out), len(tasks))
            pending = [i for i in pending if "error" in out[i]]
            if not pending:
                break
        return [out[i] for i in range(len(tasks))]

    def close(self):
        self.ex.shutdown(wait=False, cancel_futures=True)


def visible_gpus() -> list[int]:
    try:
        import torch

        return list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    except Exception:
        return []


# ----------------------------------------------------------------------------- suites


# suite -> (candidate's decks, opponent's decks, opponent FirstLight model)
#   hog26         Hog 2.6 mirror                                      vs FL General
#   mixed         the same FirstLight focus deck on both sides        vs FL General
#   pop_vs_focus  a popular human deck  vs a FirstLight focus deck    vs FL General
#   pop_vs_hog2   a popular human deck  vs Hog 2.6                    vs FL Hog 2 (the Hog specialist)
#   hog_vs_focus  Hog 2.6               vs a FirstLight focus deck    vs FL General
SUITE_OPP = {"pop_vs_hog2": "hog2"}


def suite_opponent(suite: str) -> str:
    return SUITE_OPP.get(suite, "general")


_FL_CARDS: frozenset | None = None


def _fl_known_cards() -> frozenset:
    """RoyaleSim card ids FirstLight can name (its seats track both decks)."""
    global _FL_CARDS
    if _FL_CARDS is None:
        _paths()
        from flport.catalog import Catalog as FLCatalog

        from .game import make_engine

        _FL_CARDS = frozenset(FLCatalog.build(make_engine()).fl_card_by_rs)
    return _FL_CARDS


def suite_decks(suite: str, n_pairs: int, seed: int) -> list[tuple[str, str]]:
    """(candidate deck, opponent deck) per seat-swapped pair."""
    from .game import HOG26, Catalog, deck_key, make_engine, parse_deck_key
    from .league import DeckBook, focus_decks

    hog = deck_key(HOG26)
    if suite == "hog26":
        return [(hog, hog)] * n_pairs
    rng = np.random.default_rng(seed)
    cat = Catalog.build(make_engine())
    focus = focus_decks(cat)
    if not focus:
        raise RuntimeError("FirstLight focus decks are not available")

    def focus_at(i):  # every focus deck once before any repeats
        return focus[int(rng.integers(len(focus)))] if i >= len(focus) else focus[(seed + i) % len(focus)]

    if suite == "mixed":
        return [(focus_at(i), focus_at(i)) for i in range(n_pairs)]
    if suite == "hog_vs_focus":
        return [(hog, focus_at(i)) for i in range(n_pairs)]
    if suite in ("pop_vs_focus", "pop_vs_hog2"):
        known = _fl_known_cards()
        book = DeckBook(cat, rng)
        out = []
        for i in range(n_pairs):
            for _ in range(200):  # popularity-weighted human deck FirstLight can also name
                d = book.popular()
                dk = parse_deck_key(d)
                # FirstLight's public tracker raises "public ability source changed card identity"
                # when one side plays two hero cards (e.g. Hero Musketeer + Hero Ice Golem)
                if all(cat.ids[n] in known for n, _ in dk) and sum(f == 2 for _, f in dk) <= 1:
                    break
            out.append((d, focus_at(i) if suite == "pop_vs_focus" else hog))
        return out
    raise ValueError(f"unknown suite {suite}")


def eval_tasks(cand: dict, opp: dict, suite: str, games: int, seed: int) -> list[dict]:
    """Seat-swapped pairs: the candidate side ``cand`` vs ``opp``."""
    pairs = max(1, games // 2)
    tasks = []
    for i, (dc, do) in enumerate(suite_decks(suite, pairs, seed)):
        s = seed * 100003 + i
        for ct in (0, 1):
            sides, dks = [None, None], [None, None]
            sides[ct], sides[1 - ct] = cand, opp
            dks[ct], dks[1 - ct] = dc, do
            tasks.append({"seed": s, "decks": dks, "sides": sides, "meta": {"cand_team": ct, "pair": i, "suite": suite}})
    return tasks


def summarize(results: list[dict]) -> dict:
    ok = [r for r in results if "error" not in r]
    scores, opp_plays_won, plays, first = [], [], [], []
    for r in ok:
        ct = r["meta"]["cand_team"]
        w = r["winner"]
        s = 1.0 if w == ct else 0.0 if w in (0, 1) else 0.5
        scores.append(s)
        plays.append(r["plays"][ct])
        if r["first_play"][ct] is not None:
            first.append(r["first_play"][ct] * 0.05)
        if s == 1.0:
            opp_plays_won.append(r["plays"][1 - ct])
    n = len(scores)
    wins = sum(1 for s in scores if s == 1.0)
    lo, hi = wilson(wins, n)
    return {"games": n, "errors": len(results) - n, "wins": wins, "draws": sum(1 for s in scores if s == 0.5),
            "win_rate": wins / max(1, n), "score": float(np.mean(scores)) if scores else 0.0, "wilson95": [lo, hi],
            "plays_per_game": float(np.mean(plays)) if plays else 0.0,
            "first_play_s": float(np.mean(first)) if first else None,
            "wins_vs_passive_share": sum(1 for p in opp_plays_won if p <= 3) / max(1, wins),
            "seconds_per_game": float(np.mean([r["seconds"] for r in ok])) if ok else None,
            "error_sample": next((r["error"] for r in results if "error" in r), None)}


def ours(path: str, decode: str = "t05") -> dict:
    return {"kind": "ours", "path": str(path), "decode": decode}


def fl(name: str, sample: bool = False, force_first: bool = False, record: str | None = None) -> dict:
    return {"kind": "fl", "name": name, "sample": sample, "force_first": force_first, "record": record}


# ----------------------------------------------------------------------------- recording


RECORD_MIX = (  # (share, side 0, deck 0, side 1, deck 1) — seats are swapped every other game
    (0.40, "hog2", "hog26", "general", "hog26"),
    (0.35, "general", "focus", "general", "focus"),
    (0.25, "hog2", "hog26", "general", "focus"),
)


def _record_chunk(args) -> dict:
    chunk, games, base_seed, holdout_mod, attempt = args
    from .game import HOG26, deck_key, is_hog26, parse_deck_key
    from .league import focus_decks
    from .shards import ShardWriter, pack_rows

    try:
        rng = np.random.default_rng(base_seed + chunk * 7919 + attempt)  # fresh games on a retry
        focus = focus_decks(_W["cat"]) or [deck_key(HOG26)]
        u = (chunk * 0.6180339887) % 1.0  # spreads matchups evenly over chunks
        acc = 0.0
        for share, n0, d0, n1, d1 in RECORD_MIX:
            acc += share
            if u < acc:
                break
        writers = {tag: ShardWriter(FL_SHARDS / tag, f"c{chunk:05d}_a{attempt:02d}", 10 ** 6)
                   for tag in ("general", "hog2")}
        stats = {"games": 0, "asked": games, "steps": {"general": 0, "hog2": 0}, "errors": 0, "seconds": 0.0}
        for g in range(games):
            seed = int(rng.integers(2**62))
            deck = [deck_key(HOG26) if d0 == "hog26" else focus[int(rng.integers(len(focus)))],
                    deck_key(HOG26) if d1 == "hog26" else focus[int(rng.integers(len(focus)))]]
            names = [n0, n1]
            if g % 2:
                deck, names = deck[::-1], names[::-1]
            sample = bool(g % 4 >= 2)  # half argmax as benchmarked, half sampled with a forced first play
            sides = [fl(names[t], sample=sample, force_first=sample, record=names[t]) for t in (0, 1)]
            res = _safe_game({"seed": seed, "decks": deck, "sides": sides})
            if "error" in res:
                stats["errors"] += 1
                if stats["errors"] <= 2:
                    log(f"record chunk {chunk}: {res['error']}\n{res['trace']}", "flrecord")
                continue
            stats["games"] += 1
            stats["seconds"] += res["seconds"]
            w = res["winner"]
            ghash = zlib.crc32(f"fl{chunk}_{g}_{seed}".encode()) & 0x7FFFFFFF
            hold = int(ghash % holdout_mod == 0)
            for t, buf in res["recorded"].items():
                if len(buf) < 40:
                    continue
                outcome = 1 if w == t else -1 if w in (0, 1) else 0
                hog = int(is_hog26(n for n, _ in parse_deck_key(deck[t])))
                meta = [ghash, t, outcome, 0, hog, hog, 0, hold]
                tag = names[t]
                writers[tag].add_match([(pack_rows(buf.rows, buf.lab), meta, deck[t])])
                stats["steps"][tag] += len(buf)
        for wr in writers.values():
            wr.flush()
        return {"chunk": chunk, **stats}
    except Exception:
        return {"chunk": chunk, "fatal": traceback.format_exc()[-2000:]}


RECORD_PER_CHUNK = 6


def _chunk_parts(c: int) -> list[dict]:
    return [read_json(p, {}) or {} for p in sorted((FL_SHARDS / "parts").glob(f"c{c:05d}_a*.json"))]


def record_complete() -> bool:
    """Every chunk has its full quota of finished games (counted from its part markers) and
    both FirstLight models have data."""
    games = settings()["fl_record_max_games"]
    n_chunks = (games + RECORD_PER_CHUNK - 1) // RECORD_PER_CHUNK
    full = all(sum(p.get("games", 0) for p in _chunk_parts(c)) >= RECORD_PER_CHUNK for c in range(n_chunks))
    return full and all(any((FL_SHARDS / tag).glob("*.npz")) for tag in ("general", "hog2"))


def record(workers: int, games: int, gpus: list[int], per_chunk: int = RECORD_PER_CHUNK, seed: int = 777) -> None:
    """Chunks of ``per_chunk`` FirstLight games. Each attempt of a chunk writes its own shards
    and, when it finishes, a part marker with its count of finished games; the next attempt
    plays only the missing games, with fresh seeds. An attempt without a marker (interrupted)
    has its shards removed. Raises (so the stage retries) until every quota is filled."""
    parts_dir = FL_SHARDS / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    n_chunks = (games + per_chunk - 1) // per_chunk
    tries = read_json(FL_SHARDS / "attempts.json", {}) or {}
    todo = []
    for c in range(n_chunks):
        marked = {p.stem for p in parts_dir.glob(f"c{c:05d}_a*.json")}
        for tag in ("general", "hog2"):
            for p in (FL_SHARDS / tag).glob(f"c{c:05d}_a*_*.npz"):
                if p.name.rsplit("_", 1)[0] not in marked:
                    p.unlink()
        missing = per_chunk - sum(p.get("games", 0) for p in _chunk_parts(c))
        if missing > 0:
            tries[str(c)] = tries.get(str(c), 0) + 1
            todo.append((c, missing, tries[str(c)]))
    atomic_write_json(FL_SHARDS / "attempts.json", tries)
    log(f"recording FirstLight games: {len(todo)}/{n_chunks} chunks to fill, {workers} workers", "flrecord")
    pool = FLPool(workers, gpus)
    try:
        futs = {pool.ex.submit(_record_chunk, (c, n, seed, 10, att)): (c, att) for c, n, att in todo}
        for i, f in enumerate(as_completed(futs)):
            c, att = futs[f]
            res = f.result()
            if "fatal" in res:
                log(f"record chunk {c} attempt {att} failed:\n{res['fatal']}", "flrecord")
                continue
            atomic_write_json(parts_dir / f"c{c:05d}_a{att:02d}.json", res)
            if i % 10 == 0 or res["games"] < res.get("asked", res["games"]):
                log(f"recorded chunk {c} attempt {att} ({i + 1}/{len(todo)}): {res['games']} games finished, "
                    f"{res['errors']} failed, {res['seconds'] / max(1, res['games']):.0f}s/game", "flrecord")
    finally:
        pool.close()
    log(f"recording complete: {record_complete()}", "flrecord")
    if not record_complete():
        raise SystemExit("recording incomplete (games still missing, or a FirstLight model without data); retry")


# ----------------------------------------------------------------------------- CLI


def _deck_arg(text: str) -> str:
    """hog26, focus:N (FirstLight focus deck N), popular:N (N-th most played human deck) or a deck key."""
    from .game import HOG26, Catalog, deck_key, make_engine, parse_deck_key
    from .league import focus_decks

    if text == "hog26":
        return deck_key(HOG26)
    kind, _, num = text.partition(":")
    if kind == "focus":
        return focus_decks(Catalog.build(make_engine()))[int(num)]
    if kind == "popular":
        from .league import DeckBook

        cat = Catalog.build(make_engine())
        known = _fl_known_cards()
        keys = [k for k in DeckBook(cat, np.random.default_rng(0)).pop_keys
                if all(cat.ids[n] in known for n, _ in parse_deck_key(k))]
        return keys[int(num)]
    return text


def trace_battle(a, gpus: list[int]) -> None:
    """One recorded battle: blue = team 0 (bottom), red = team 1 (top)."""
    def side(spec):
        return fl(spec[3:]) if spec.startswith("fl:") else ours(spec, a.decode)

    _init(_CTX.Value("i", 0), gpus)
    decks = [_deck_arg(a.blue_deck), _deck_arg(a.red_deck)]
    res = play_game({"seed": a.seed, "decks": decks, "sides": [side(a.blue), side(a.red)], "trace": a.out})
    names = [a.blue, a.red]
    w = res["winner"]
    info = {"blue": names[0], "red": names[1], "blue_deck": decks[0], "red_deck": decks[1], "seed": a.seed,
            "winner": ("blue" if w == 0 else "red" if w == 1 else "draw"), "crowns": res["crowns"],
            "ticks": res["ticks"], "plays": res["plays"], "trace": a.out}
    # render_video.py --eval: P(blue wins) per tick, from blue's own win-probability head when
    # blue is one of ours, else 1 - red's estimate
    def blue_p(team, p, q):  # P(blue wins) among decisive outcomes, from one head's (win, loss)
        pb = p / max(1e-6, p + q)
        return pb if team == 0 else 1.0 - pb

    curve = None
    heads = {t: {c[0]: c for c in res["curves"][t]} for t in res["curves"]}
    if heads:
        ticks = sorted(set().union(*[set(h) for h in heads.values()]))
        curve = []
        for tick in ticks:
            vals = [blue_p(t, h[tick][1], h[tick][2]) for t, h in heads.items() if tick in h]
            crowns = next(h[tick][3] for h in heads.values() if tick in h)
            curve.append({"tick": tick, "p_blue": round(sum(vals) / len(vals), 4), "crowns": crowns})
    out = Path(a.out)
    out.with_suffix(".json").write_text(json.dumps(info, indent=1))
    if curve:
        out.with_name(out.stem + "_eval.json").write_text(json.dumps({"eval": curve}))
    print(json.dumps(info))


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eval")
    e.add_argument("--ckpt", required=True)
    e.add_argument("--opp", default="")
    e.add_argument("--suite", default="hog26", choices=["hog26", "mixed", "pop_vs_focus", "pop_vs_hog2", "hog_vs_focus"])
    e.add_argument("--games", type=int, default=40)
    e.add_argument("--decode", default="t05")
    e.add_argument("--seed", type=int, default=5)
    e.add_argument("--workers", type=int, default=0)
    e.add_argument("--out", required=True)
    b = sub.add_parser("baseline")
    b.add_argument("--games", type=int, default=0)
    b.add_argument("--suites", default="mixed,hog_vs_focus")
    b.add_argument("--workers", type=int, default=0)
    b.add_argument("--out", required=True)
    t = sub.add_parser("trace", help="record one battle for a video (render with render_video.py)")
    t.add_argument("--blue", required=True, help="checkpoint path, or fl:general / fl:hog2")
    t.add_argument("--red", required=True)
    t.add_argument("--blue-deck", default="hog26", help="deck key, or hog26 / focus:N / popular:N")
    t.add_argument("--red-deck", default="hog26")
    t.add_argument("--decode", default="argmax")
    t.add_argument("--seed", type=int, default=1)
    t.add_argument("--out", required=True, help=".msgpack trace; a .json with the result and P(win) curve goes next to it")
    r = sub.add_parser("record")
    r.add_argument("--workers", type=int, default=0)
    r.add_argument("--games", type=int, default=0)
    a = ap.parse_args(argv)
    cfg = settings()
    gpus = visible_gpus()
    if a.cmd == "trace":
        trace_battle(a, gpus)
        return
    if a.cmd == "record":
        record(a.workers or cfg["fl_record_workers"], a.games or cfg["fl_record_max_games"], gpus)
        return
    workers = a.workers or cfg["fl_eval_workers"]
    pool = FLPool(workers, gpus)
    try:
        if a.cmd == "eval":
            tasks = eval_tasks(ours(a.ckpt, a.decode), fl(a.opp or suite_opponent(a.suite)), a.suite, a.games, a.seed)
            res = summarize(pool.run(tasks))
            res.update({"ckpt": a.ckpt, "opp": a.opp or suite_opponent(a.suite), "suite": a.suite, "decode": a.decode})
        else:
            n = a.games or cfg["fl_baseline_games"]
            res = read_json(Path(a.out), {}) or {}
            for suite in a.suites.split(","):
                if isinstance(res.get(suite), dict) and not res[suite].get("errors"):
                    continue  # already measured
                tasks = eval_tasks(fl("hog2"), fl("general"), suite, n, 4242)
                res[suite] = summarize(pool.run(tasks, progress=lambda i, t: i % 10 == 0 and log(
                    f"baseline {suite}: {i}/{t}", "fleval")))
            res["protocol"] = "FL Hog 2 (argmax) vs FL General (argmax), seat-swapped pairs, no delay"
        bad = [r for r in ([res] if a.cmd == "eval" else [v for v in res.values() if isinstance(v, dict)])
               if r["errors"]]
        if bad:  # an incomplete evaluation is not a result: fail so the stage is retried
            raise SystemExit(f"{a.cmd}: too many failed games: {bad[0]['errors']} errors, e.g. {bad[0]['error_sample']}")
        atomic_write_json(Path(a.out), res)
        log(f"{a.cmd}: {res}", "fleval")
    finally:
        pool.close()


if __name__ == "__main__":
    main()
