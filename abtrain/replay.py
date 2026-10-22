"""Human IL_Replay matches re-simulated tick by tick in RoyaleSim, labelled in our action format.

Each match is replayed with its decks and card forms (evolutions, heroes), every card play is
injected at its exact tick and point, and ability presses at their tick. Both seats are observed
every 250 ms (5 ticks). Plays sent inside a 5-tick window become that window's label:
(candidate, tile, delay bin) for the first, and again for a second play; a third play in one
window moves to the next window.

Divergence: the first card play the engine refuses (or a card not in hand) ends the match there
for both seats; the prefix is kept with ``truncated = 1``. A refused ability press is only
removed from its label. The deal order is the unshuffled order under which every play was in
hand (``royaleimitate.replays.deal_order``).

Outcome per seat: the simulator's winner when the battle ended in the simulator; otherwise
(truncated, or the real game ended first) the real recorded result.
"""

from __future__ import annotations

import zlib
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .game import (DT, HAND, NBTN, NT, Catalog, SeatView, is_hog26, label_supported, nearest_legal_tile, own_tile,
                   legal_masks, sanitize_forms)

MIN_STEPS = 60
SKIP = -2  # label marker: this step stays out of the policy loss (see shards.collate)
LOCKOUT = 90  # the engine refuses every deploy before tick 90
_SIDES = ("team", "opponent")


@dataclass
class Plan:
    tag: str
    decks: list[list[tuple[str, int]]]  # per seat, dealt order, (name, form)
    plays: list[list[tuple[int, str, int, int]]]  # per seat: (tick, name, x, y) engine units
    presses: list[list[tuple[int, list[str]]]]  # per seat: (tick, candidate card names)
    outcome: tuple[int, int]
    end_tick: int | None
    hog: tuple[bool, bool]

    @property
    def match_hash(self) -> int:
        return zlib.crc32(self.tag.encode()) & 0x7FFFFFFF


def _form(key: str) -> int:
    k = key.lower()
    if k.endswith("-hero"):
        return 2
    if "-ev" in k:
        return 1
    return 0


def plan_match(payload: dict, cat: Catalog) -> Plan | str:
    from royaleimitate.replays import CARD_NAMES, base_key, deal_order

    battle = payload["battle"]
    decks: list[list[tuple[str, int]]] = []
    for side in _SIDES:
        players = battle[side]["players"]
        if len(players) != 1:
            return "not one player a side"
        pl = players[0]
        tower = (pl.get("tower_card") or {}).get("card_key", "tower-princess")
        if tower != "tower-princess":
            return "tower troop not princess"
        deck = []
        for card in pl["deck"]:
            name = CARD_NAMES.get(base_key(card["card_key"]))
            if name is None or name not in cat.ids:
                return "card not in the catalogue"
            deck.append((name, _form(card["card_key"])))
        if len(deck) != 8 or len({n for n, _ in deck}) != 8:
            return "not an eight-card deck"
        decks.append(sanitize_forms(cat, deck))
    plays: list[list[tuple[int, str, int, int]]] = [[], []]
    presses: list[list[tuple[int, list[str]]]] = [[], []]
    events = sorted(payload["events"], key=lambda e: (e["replay_tick_20hz"], e["source_index"]))
    for ev in events:
        seat = _SIDES.index(ev["side"]) if ev.get("side") in _SIDES else None
        if seat is None:
            return "event without a side"
        tick = int(ev["replay_tick_20hz"])
        if ev["kind"] == "play_card":
            name = CARD_NAMES.get(base_key(ev["card_key"] or ""))
            if name not in {n for n, _ in decks[seat]}:
                return "play of a card not in the deck"
            co = ev.get("coordinates") or {}
            units = co.get("native_world_units")
            if not units:
                return "play without coordinates"
            plays[seat].append((tick, name, int(units["x"]) * 18, int(units["y"]) * 18))
        elif ev["kind"] == "activate_ability":
            cands = [CARD_NAMES.get(base_key(k)) for k in (ev.get("ability_source_candidates") or [])]
            presses[seat].append((tick, [c for c in cands if c]))
    dealt = []
    for seat in (0, 1):
        names = [n for n, _ in decks[seat]]
        order = deal_order(names, [p[1] for p in plays[seat]])
        if order is None:
            return "no deal order fits the plays"
        forms = dict(decks[seat])
        dealt.append([(n, forms[n]) for n in order])
    res = battle.get("result")
    outcome = {"victory": (1, -1), "defeat": (-1, 1)}.get(res, (0, 0))
    secs = (((payload.get("replay") or {}).get("duration") or {}).get("display_seconds"))
    end_tick = int(secs) * 20 + 20 if secs else None
    hog = (is_hog26(n for n, _ in dealt[0]), is_hog26(n for n, _ in dealt[1]))
    return Plan(payload["source"]["replay_tag"], dealt, plays, presses, outcome, end_tick, hog)


@dataclass
class SeqBuf:
    """One seat's decisions, as numpy rows, plus the label rows."""

    rows: list = field(default_factory=list)  # Obs per step
    lab: list = field(default_factory=list)  # [c1, t1, d1, c2, t2, d2]

    def __len__(self) -> int:
        return len(self.rows)

    def cut(self, n: int) -> None:
        del self.rows[n:]
        del self.lab[n:]


class _Cmd:
    __slots__ = ("send", "seat", "kind", "cand", "x", "y", "step", "slot", "name")

    def __init__(self, send, seat, kind, cand, x, y, step, slot, name):
        self.send, self.seat, self.kind, self.cand = send, seat, kind, cand
        self.x, self.y, self.step, self.slot, self.name = x, y, step, slot, name


def replay(engine, parser, cat: Catalog, plan: Plan, seed: int) -> tuple[list[SeqBuf], dict]:
    """Returns ([SeqBuf seat 0, SeqBuf seat 1], info). ``info`` has outcome, truncated, reason."""
    from royalegym.protocol import DeployCommand, DeployStatus, MatchSetup, ShuffleMode

    decks = [[cat.ids[n] for n, _ in d] for d in plan.decks]
    forms = [[f for _, f in d] for d in plan.decks]
    try:
        engine.reset(seed, MatchSetup(decks=decks, forms=forms, shuffle=ShuffleMode.NONE))
    except Exception:
        forms = [[0] * 8, [0] * 8]
        engine.reset(seed, MatchSetup(decks=decks, forms=forms, shuffle=ShuffleMode.NONE))
    views = [SeatView(cat, parser), SeatView(cat, parser)]
    for s in (0, 1):
        views[s].reset(s, decks[s], forms[s], decks[1 - s])
    # upcoming actions per seat: (send_tick, kind, payload); a play is sent one tick before it
    # appears, never before the opening lockout
    queues = []
    for s in (0, 1):
        q = [(max(t - 1, LOCKOUT), 0, (cat.ids[n], x, y)) for t, n, x, y in plan.plays[s]]
        q += [(max(t - 1, LOCKOUT), 1, tuple(cat.ids[c] for c in cands if c in cat.ids))
              for t, cands in plan.presses[s]]
        q.sort(key=lambda r: (r[0], r[1]))
        queues.append(deque(q))
    bufs = [SeqBuf(), SeqBuf()]
    info = {"truncated": 0, "reason": "", "press_refused": 0, "tile_fixed": 0, "tile_lost": 0,
            "deferred": 0, "skipped_steps": 0}
    state = engine.state()
    end_tick = plan.end_tick
    stop = False
    while not state.game_over and not stop:
        tick = int(state.tick)
        if end_tick is not None and tick >= end_tick:
            break
        if tick % DT:
            # only reached if a step left us off-grid, which step() below never does
            engine.step([], DT - tick % DT)
            state = engine.state()
            continue
        step_idx = len(bufs[0])
        window: list[_Cmd] = []
        for s in (0, 1):
            cand_mask, tiles = legal_masks(parser, state, s)
            obs = views[s].build(state, cand_mask, tiles)
            me = state.players[s]
            hand = list(me.hand)
            lab = [-1, -1, -1, -1, -1, -1]
            q = queues[s]
            taken = 0
            skip = False
            used_slots: set[int] = set()
            deferred = []
            while q and q[0][0] < tick + DT:
                send, kind, payload = q.popleft()
                send = max(send, tick)
                if taken >= 2:
                    deferred.append((tick + DT, kind, payload))
                    info["deferred"] += 1
                    continue
                if kind == 0:
                    cid, x, y = payload
                    if cid not in hand:
                        if taken == 1:  # drawn after the first play: try it next window
                            deferred.append((tick + DT, kind, payload))
                            info["deferred"] += 1
                            continue
                        stop = True
                        info["reason"] = "card not in hand"
                        break
                    slot = hand.index(cid)
                    if slot in used_slots:
                        deferred.append((tick + DT, kind, payload))
                        info["deferred"] += 1
                        continue
                    t = own_tile(s, x, y)
                    t2 = nearest_legal_tile(tiles[slot], t)
                    if t2 != t:
                        info["tile_fixed" if t2 >= 0 else "tile_lost"] += 1
                    if t2 < 0:
                        skip = True  # no legal tile at the window start: the policy could not do this
                    lab[3 * taken : 3 * taken + 3] = [slot, t2, send - tick]
                    used_slots.add(slot)
                    window.append(_Cmd(send, s, 0, slot, x, y, step_idx, taken, cid))
                    taken += 1
                else:
                    btn = -1
                    for b, row in enumerate(me.abilities[:NBTN]):
                        card = int(row[3]) if len(row) > 3 else -1
                        if card in payload:
                            btn = b
                            break
                    if btn < 0 and len(me.abilities) == 1:
                        btn = 0
                    if btn < 0 or not cand_mask[HAND + btn] or (HAND + btn) in used_slots:
                        info["press_refused"] += 1
                        continue
                    lab[3 * taken : 3 * taken + 3] = [HAND + btn, -1, send - tick]
                    used_slots.add(HAND + btn)
                    window.append(_Cmd(send, s, 1, HAND + btn, 0, 0, step_idx, taken, -1))
                    taken += 1
            if stop:
                break
            for d in reversed(deferred):
                q.appendleft(d)
            if len(q) > 1 and deferred:
                queues[s] = deque(sorted(q, key=lambda r: (r[0], r[1])))
            bufs[s].rows.append(obs)
            skip = skip or not label_supported(obs, lab)
            info["skipped_steps"] += int(skip)
            bufs[s].lab.append([SKIP, -1, -1, -1, -1, -1] if skip else lab)
        if stop:
            for b in bufs:
                b.cut(step_idx)
            info["truncated"] = 1
            break
        # run the window tick by tick, sending each command at its tick
        window.sort(key=lambda c: (c.send, c.seat, c.slot))
        for k in range(DT):
            now = tick + k
            cmds = [c for c in window if c.send == now]
            wire = []
            for c in cmds:
                if c.kind == 0:
                    wire.append(DeployCommand(team=c.seat, hand_slot=c.cand, x=c.x, y=c.y))
                else:
                    wire.append(DeployCommand(team=c.seat, hand_slot=c.cand, x=0, y=0))
            results = engine.step(wire, 1)
            for c, r in zip(cmds, results):
                if r.status == DeployStatus.OK:
                    if c.kind == 0:
                        for v in views:
                            v.note_play(c.seat, c.name)
                    continue
                if c.kind == 1:  # a refused press only loses its label
                    info["press_refused"] += 1
                    lab = bufs[c.seat].lab[c.step]
                    if lab[0] == SKIP:
                        continue
                    if c.slot == 0:
                        lab[0:3] = lab[3:6]
                        lab[3:6] = [-1, -1, -1]
                    else:
                        lab[3:6] = [-1, -1, -1]
                    continue
                stop = True
                info["reason"] = f"play refused: {DeployStatus(r.status).name}"
                break
            if stop:
                break
        if stop:
            for b in bufs:
                b.cut(step_idx)
            info["truncated"] = 1
            break
        state = engine.state()
    if state.game_over and not info["truncated"]:
        w = int(state.winner)
        outcome = (1, -1) if w == 0 else (-1, 1) if w == 1 else (0, 0)
        info["outcome_source"] = 0
    else:
        outcome = plan.outcome
        info["outcome_source"] = 1
    info["outcome"] = outcome
    info["sim_agrees"] = int(state.game_over and outcome == plan.outcome)
    return bufs, info
