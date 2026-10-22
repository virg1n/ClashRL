"""The game layer shared by extraction, PPO and evaluation: engine factory, card catalogue,
decks with forms, the per-seat observation, legality masks and the action codec.

Frames: every policy input is in the acting seat's own frame (seat 1 is turned half a turn),
tiles are 18 x 32, tile index = y * 18 + x, y = 0 is the seat's own back line.

Action (FirstLight V4 style, one decision per 250 ms = 5 engine ticks):
  gate (wait / act) -> candidate (4 hand slots + 3 ability buttons) -> tile (576, cards only)
  -> delay bin (0..4 ticks into the window) -> continue? -> second candidate / tile / delay
  (second delay >= first). ``Play = (cand, tile, delay)``; tile is -1 for a button press.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field

import numpy as np

NX, NY = 18, 32
NT = NX * NY
TILE = 18000
W_SUB, H_SUB = NX * TILE, NY * TILE
DT = 5  # engine ticks per decision (250 ms)
HAND = 4
NBTN = 3
NCAND = HAND + NBTN
NDELAY = 5
CARD_VOCAB = 160  # engine catalogue has 145 cards; ids are shifted by one (0 = none)
UNIT_VOCAB = 256  # engine has 237 unit types
N_KINDS = 6
KIND_TROOP, KIND_BUILDING, KIND_KING, KIND_PRINCESS, KIND_SPELL, KIND_PROJECTILE = range(6)
CLIENT = "15.535.29"
LIVE_DELAY_TICKS = 21

ENT_NAMES = (
    "x", "y", "hp_frac", "log_hp", "log_max_hp", "shield_frac", "own", "flying", "deploying",
    "deploy_s", "stunned", "knockback", "ph_unknown", "ph_idle", "ph_windup", "ph_fire",
    "has_target", "tgt_tower", "tgt_building", "tgt_dx", "tgt_dy", "vx", "vy", "age", "radius",
    "charge", "st_underground", "st_invisible", "st_hidden", "st_evolved", "st_hero", "st_clone",
    "st_windup", "st_ability", "king_active", "aim_dx", "aim_dy", "delay_s", "splash", "homing",
    "ptgt_dx", "ptgt_dy", "motion", "flown", "n_buffs", "is_target_link",
)
F = len(ENT_NAMES)
GLOB_NAMES = (
    "elixir", "elixir_frac", "time", "remaining", "overtime", "rate", "own_crowns", "opp_crowns",
    "own_king_active", "opp_king_active", "own_t0", "own_t1", "own_t2", "opp_t0", "opp_t1",
    "opp_t2", "opp_elixir_est", "n_own", "n_opp", "pending_cost", "n_pending", "cmd_delay",
    "btn0_has", "btn0_avail", "btn0_spent", "btn0_cost", "btn1_has", "btn1_avail", "btn1_cost",
    "btn2_has", "btn2_avail", "btn2_cost", "lockout", "pending_min_left", "opp_cards_seen",
)
G = len(GLOB_NAMES)
CARD_ROWS = 4 + 1 + 3 + 8  # hand, next, own cycle, opponent revealed
CF = 10  # cost, affordable, hand, next, cycle, opp, evo form, hero form, evo ready, pending
PRIV_IDS = 4 + 1 + 8  # opponent hand, next, deck
PF = 4  # opponent exact elixir, opponent pending cost, opponent button available, elixir balance
MAX_ENT = 128
# A second play must be affordable at the window start with the first one paid. Elixir is kept
# exact (milli-elixir) for this check, so the slack is only float32 noise.
SECOND_PLAY_SLACK = 1e-4


def make_engine(delay: int | tuple[int, int] = 0):
    from royalegym.rust_engine import RustEngine, client_overrides

    eng = RustEngine(calibration_overrides=client_overrides(CLIENT))
    if delay:
        eng.set_command_delay_ticks(delay)
    return eng


def make_parser(engine):
    from royalegym.action import TileActionParser

    p = TileActionParser(ability_buttons=True)
    p.bind(engine)
    assert p.nx == NX and p.ny == NY, (p.nx, p.ny)
    return p


@dataclass
class Catalog:
    """Card table of the bound engine, plus which forms each card supports."""

    names: list[str]
    ids: dict[str, int]
    cost: np.ndarray
    count: np.ndarray  # units one play puts down (>= 1)
    spell: np.ndarray
    evo_ok: frozenset[str]
    hero_ok: frozenset[str]
    unit_names: list[str] = field(default_factory=list)

    @classmethod
    def build(cls, engine) -> "Catalog":
        from royalegym.protocol import card_is_spell

        cards = list(engine.cards())
        assert len(cards) < CARD_VOCAB - 1, len(cards)
        names = [c.name for c in cards]
        ids = {c.name: c.card_id for c in cards}
        cost = np.zeros(CARD_VOCAB, np.float32)
        count = np.ones(CARD_VOCAB, np.float32)
        spell = np.zeros(CARD_VOCAB, bool)
        for c in cards:
            cost[c.card_id] = c.elixir
            count[c.card_id] = max(1, int(c.count or 1))
            spell[c.card_id] = card_is_spell(c)
        evo_ok = frozenset(c.name for c in cards if c.evo_cycle)
        units = engine.unit_types() or []
        assert len(units) < UNIT_VOCAB - 1, len(units)
        return cls(names, ids, cost, count, spell, evo_ok, _hero_forms(tuple(names)), units)


@functools.lru_cache(maxsize=1)
def _hero_forms(names: tuple[str, ...]) -> frozenset[str]:
    """Cards the engine accepts in their hero form, found by asking it."""
    from royalegym.protocol import MatchSetup, ShuffleMode

    eng = make_engine()
    ids = {n: i for i, n in enumerate(n for n in names)}
    filler = [n for n in ("Knight", "Archer", "Goblins", "Giant", "Pekka", "Minions", "Balloon", "Witch")]
    ok = set()
    for n in names:
        rest = [ids[f] for f in filler if f != n][:7]
        try:
            eng.reset(1, MatchSetup(decks=[[ids[n]] + rest] * 2, forms=[[2] + [0] * 7] * 2, shuffle=ShuffleMode.NONE))
            ok.add(n)
        except Exception:
            pass
    return frozenset(ok)


# --- decks ---------------------------------------------------------------------------------

# FirstLight's Hog 2.6 specialist deck with its forms (hero Musketeer, evo Cannon and Skeletons).
HOG26 = (("HogRider", 0), ("Musketeer", 2), ("Cannon", 1), ("Fireball", 0), ("Log", 0),
         ("Skeletons", 1), ("IceGolemite", 0), ("IceSpirits", 0))
HOG26_SET = frozenset(n for n, _ in HOG26)


def deck_key(deck: list[tuple[str, int]] | tuple) -> str:
    """Order-free identity of a deck with forms, e.g. ``Cannon:1|Fireball:0|...``."""
    return "|".join(sorted(f"{n}:{f}" for n, f in deck))


def parse_deck_key(key: str) -> list[tuple[str, int]]:
    out = []
    for part in key.split("|"):
        n, f = part.rsplit(":", 1)
        out.append((n, int(f)))
    return out


def is_hog26(names) -> bool:
    return frozenset(names) == HOG26_SET


def sanitize_forms(cat: Catalog, deck: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """Drop forms the engine cannot play (they become the base card)."""
    out = []
    for n, f in deck:
        if f == 1 and n not in cat.evo_ok:
            f = 0
        if f == 2 and n not in cat.hero_ok:
            f = 0
        out.append((n, f))
    return out


# --- per-seat observation ------------------------------------------------------------------

_STATUS_BITS = (1, 2, 4, 8, 16, 32, 64, 128)
_UNVALIDATED_STATUS = (32,)  # st_windup
ELIXIR_TICKS = 56  # ticks per elixir at 1x (2.8 s)


@dataclass
class Obs:
    glob: np.ndarray  # [G] f32
    cards: np.ndarray  # [CARD_ROWS] i16, card id + 1 (0 = none)
    cardf: np.ndarray  # [CARD_ROWS, CF] f32
    ent_ids: np.ndarray  # [E, 3] i16: unit type + 1, card id + 1, kind
    ent_f: np.ndarray  # [E, F] f32
    priv_ids: np.ndarray  # [PRIV_IDS] i16, card id + 1
    priv_f: np.ndarray  # [PF] f32
    cand: np.ndarray  # [NCAND] bool, legal candidates
    tiles: np.ndarray  # [HAND, NT] bool, legal tiles per hand slot
    cand_cost: np.ndarray  # [NCAND] f32 elixir cost of each candidate now
    elixir: float = 0.0  # exact own elixir (milli-elixir / 1000) for the affordability rule


def legal_masks(parser, state, team) -> tuple[np.ndarray, np.ndarray]:
    m = np.asarray(parser.action_mask(state, team), dtype=bool)
    tiles = m[1 : 1 + HAND * NT].reshape(HAND, NT)
    cand = np.zeros(NCAND, bool)
    cand[:HAND] = tiles.any(axis=1)
    nb = min(NBTN, len(m) - 1 - HAND * NT)
    if nb > 0:
        cand[HAND : HAND + nb] = m[1 + HAND * NT : 1 + HAND * NT + nb]
    return cand, tiles


def elixir_balance(cat: Catalog, state, team: int) -> float:
    """B's potential: (own elixir + board worth) - (enemy's), in elixir. Towers excluded."""
    worth = [0.0, 0.0]
    for e in state.entities:
        k = int(e.kind)
        if k >= 2 or e.card_id < 0:
            continue
        worth[e.team] += float(cat.cost[e.card_id]) / float(cat.count[e.card_id]) * (e.hp / max(1, e.max_hp))
    me, op = state.players[team], state.players[1 - team]
    return (me.elixir_milli / 1000.0 + worth[team]) - (op.elixir_milli / 1000.0 + worth[1 - team])


class SeatView:
    """Builds one seat's observation every decision. Stateful per (battle, seat): entity
    velocities and ages, the own cycle, opponent cards revealed by plays and an estimate of
    the opponent's elixir from those plays. Only public information enters the policy part;
    ``priv_*`` (opponent hand, next card, deck and exact elixir) feed the value branch only."""

    def __init__(self, cat: Catalog, parser):
        self.cat = cat
        self.parser = parser
        self.seat = 0

    def reset(self, seat: int, own_deck: list[int], own_forms: list[int], opp_deck: list[int],
              start_elixir: float = 6.0, delay_ticks: int = 0) -> None:
        self.seat = seat
        self.queue = list(own_deck)  # own deal order; a played card goes to the back
        self.forms = {c: f for c, f in zip(own_deck, own_forms)}
        self.opp_deck = list(opp_deck)
        self.prev: dict[int, tuple[float, float, int]] = {}
        self.first: dict[int, int] = {}
        self.opp_seen: list[int] = []
        self.opp_est = float(start_elixir)
        self.last_tick = 0
        self.delay_ticks = int(delay_ticks)

    # public events: a play that RAN (its unit/spell appeared), seen by both seats
    def note_play(self, team: int, card_id: int) -> None:
        if card_id < 0:
            return
        if team == self.seat:
            if card_id in self.queue:
                self.queue.remove(card_id)
                self.queue.append(card_id)
        else:
            if card_id not in self.opp_seen and len(self.opp_seen) < 8:
                self.opp_seen.append(card_id)
            self.opp_est = max(0.0, self.opp_est - float(self.cat.cost[card_id]))

    def build(self, st, cand: np.ndarray, tiles: np.ndarray) -> Obs:
        s = self.seat
        flip = s == 1
        tick = int(st.tick)
        rate = float(st.elixir_rate)
        dticks = max(0, tick - self.last_tick)
        self.last_tick = tick
        self.opp_est = min(10.0, self.opp_est + dticks * rate / ELIXIR_TICKS)
        me, op = st.players[s], st.players[1 - s]
        ents = st.entities
        pos = {e.uid: e for e in ents}
        n_e = len(ents) + len(st.spells) + len(st.projectiles)
        ids = np.zeros((n_e, 3), np.int16)
        f = np.zeros((n_e, F), np.float32)
        n_own = n_opp = 0
        i = 0
        for e in ents:
            x, y = (W_SUB - e.x, H_SUB - e.y) if flip else (e.x, e.y)
            k = int(e.kind)
            kind = (KIND_TROOP, KIND_BUILDING, KIND_KING, KIND_PRINCESS)[k] if k < 4 else KIND_TROOP
            own = e.team == s
            if kind < 2:
                if own:
                    n_own += 1
                else:
                    n_opp += 1
            r = f[i]
            r[0], r[1] = x / W_SUB, y / H_SUB
            mh = max(1, e.max_hp)
            r[2] = e.hp / mh
            r[3] = np.log1p(max(0, e.hp)) / 9.0
            r[4] = np.log1p(mh) / 9.0
            r[5] = (e.shield or 0) / mh
            r[6] = own
            r[7] = e.flying
            r[8] = e.deploy_ticks > 0  # visible deploy state; the countdown itself is not validated live
            r[10] = e.stun_ticks > 0
            r[11] = e.knockback_ticks > 0
            # attack phase, charge progress and the windup status bit are sim-only per-unit
            # timers the live bridge does not validate: always "unknown" for the policy
            r[12] = 1.0
            t = pos.get(e.target_uid) if e.target_uid >= 0 else None
            if t is not None:
                tx, ty = (W_SUB - t.x, H_SUB - t.y) if flip else (t.x, t.y)
                r[16] = 1.0
                r[17] = int(t.kind) >= 2
                r[18] = int(t.kind) == 1
                r[19], r[20] = (tx - x) / W_SUB * 4.0, (ty - y) / H_SUB * 4.0
                r[45] = 1.0
            p = self.prev.get(e.uid)
            if p is not None and tick > p[2]:
                inv = 1.0 / ((tick - p[2]) * 0.05) / TILE / 5.0
                r[21], r[22] = (x - p[0]) * inv, (y - p[1]) * inv
            self.prev[e.uid] = (x, y, tick)
            r[23] = (tick - self.first.setdefault(e.uid, tick)) * 0.05 / 30.0
            r[24] = e.radius / TILE
            sf = e.status_flags
            if sf >= 0:
                for b, bit in enumerate(_STATUS_BITS):
                    r[26 + b] = bool(sf & bit) and (26 + b) not in _UNVALIDATED_STATUS
            if kind == KIND_KING:
                r[34] = st.players[e.team].king_active
            r[44] = min(len(e.buffs), 4) / 4.0
            ids[i, 0] = e.unit_type + 1 if 0 <= e.unit_type < UNIT_VOCAB - 1 else 0
            ids[i, 1] = e.card_id + 1 if e.card_id >= 0 else 0
            ids[i, 2] = kind
            i += 1
        for sp in st.spells:
            x, y = (W_SUB - sp.x, H_SUB - sp.y) if flip else (sp.x, sp.y)
            ax, ay = (W_SUB - sp.aim_x, H_SUB - sp.aim_y) if flip else (sp.aim_x, sp.aim_y)
            r = f[i]
            r[0], r[1], r[6] = x / W_SUB, y / H_SUB, sp.team == s
            r[35], r[36] = (ax - x) / W_SUB, (ay - y) / H_SUB
            r[37] = sp.delay_ticks * 0.05
            r[42] = int(sp.motion) / 7.0
            r[43] = max(0, sp.ticks_flown) / 20.0
            ids[i, 1] = sp.card_id + 1 if sp.card_id >= 0 else 0
            ids[i, 2] = KIND_SPELL
            i += 1
        for pr in st.projectiles:
            x, y = (W_SUB - pr.x, H_SUB - pr.y) if flip else (pr.x, pr.y)
            ax, ay = (W_SUB - pr.aim_x, H_SUB - pr.aim_y) if flip else (pr.aim_x, pr.aim_y)
            r = f[i]
            r[0], r[1], r[6] = x / W_SUB, y / H_SUB, pr.team == s
            r[35], r[36] = (ax - x) / W_SUB, (ay - y) / H_SUB
            r[38] = pr.splash / TILE
            t = pos.get(pr.target_uid) if pr.target_uid >= 0 else None
            if t is not None:
                tx, ty = (W_SUB - t.x, H_SUB - t.y) if flip else (t.x, t.y)
                r[39] = 1.0
                r[40], r[41] = (tx - x) / W_SUB * 4.0, (ty - y) / H_SUB * 4.0
                r[45] = 1.0
            ids[i, 1] = pr.firer_card_id + 1 if pr.firer_card_id >= 0 else 0
            ids[i, 2] = KIND_PROJECTILE
            i += 1
        if len(self.prev) > 4096:  # forget long-dead units
            live = {e.uid for e in ents}
            self.prev = {u: v for u, v in self.prev.items() if u in live}

        g = np.zeros(G, np.float32)
        el = me.elixir_milli / 1000.0
        g[0], g[1] = el / 10.0, el - int(el)
        g[2] = tick * 0.05 / 180.0
        end = st.regular_ticks + (st.overtime_ticks if st.overtime else 0)
        g[3] = max(0, end - tick) * 0.05 / 180.0
        g[4] = st.overtime
        g[5] = rate / 3.0
        g[6], g[7] = me.crowns / 3.0, op.crowns / 3.0
        g[8], g[9] = me.king_active, op.king_active
        for j in range(3):
            g[10 + j] = me.tower_hp[j] / max(1, me.tower_max_hp[j]) if j < len(me.tower_hp) else 0.0
            g[13 + j] = op.tower_hp[j] / max(1, op.tower_max_hp[j]) if j < len(op.tower_hp) else 0.0
        g[16] = self.opp_est / 10.0
        g[17], g[18] = n_own / 20.0, n_opp / 20.0
        g[19] = me.pending_cost / 10.0
        g[20] = len(me.pending) / 2.0
        g[21] = self.delay_ticks / 25.0
        btn_cost = np.zeros(NBTN, np.float32)
        for b, row in enumerate(me.abilities[:NBTN]):
            avail, spent, cost = int(row[0]), int(row[1]), int(row[2])
            btn_cost[b] = cost
            if b == 0:
                g[22], g[23], g[24], g[25] = 1.0, avail, spent, cost / 10.0
            else:
                base = 26 if b == 1 else 29
                g[base], g[base + 1], g[base + 2] = 1.0, avail, cost / 10.0
        g[32] = tick < 90
        if me.pending:
            g[33] = min(int(r[4]) for r in me.pending) / 25.0
        g[34] = len(self.opp_seen) / 8.0

        cards = np.zeros(CARD_ROWS, np.int16)
        cardf = np.zeros((CARD_ROWS, CF), np.float32)
        hand = list(me.hand)
        costs = list(me.hand_costs) if me.hand_costs else [-2] * HAND
        evo_ready = {int(r[0]) for r in me.evo if len(r) > 2 and r[2]}
        pend_cards = {int(r[1]) for r in me.pending if int(r[0]) == 0}
        cand_cost = np.zeros(NCAND, np.float32)
        for j, c in enumerate(hand[:HAND]):
            if c < 0:
                continue
            cost = float(costs[j]) if costs[j] >= 0 else float(self.cat.cost[c])
            cand_cost[j] = cost
            cards[j] = c + 1
            row = cardf[j]
            row[0], row[1], row[2] = cost / 10.0, cost <= el, 1.0
            fm = self.forms.get(c, 0)
            row[6], row[7], row[8], row[9] = fm == 1, fm == 2, c in evo_ready, c in pend_cards
        cand_cost[HAND:] = btn_cost
        if me.next_card >= 0:
            cards[4] = me.next_card + 1
            cardf[4, 0], cardf[4, 3] = self.cat.cost[me.next_card] / 10.0, 1.0
            fm = self.forms.get(me.next_card, 0)
            cardf[4, 6], cardf[4, 7] = fm == 1, fm == 2
        cyc = [c for c in self.queue if c not in hand and c != me.next_card][:3]
        for j, c in enumerate(cyc):
            cards[5 + j] = c + 1
            cardf[5 + j, 0], cardf[5 + j, 4] = self.cat.cost[c] / 10.0, 1.0
            fm = self.forms.get(c, 0)
            cardf[5 + j, 6], cardf[5 + j, 7] = fm == 1, fm == 2
        for j, c in enumerate(self.opp_seen):
            cards[8 + j] = c + 1
            cardf[8 + j, 0], cardf[8 + j, 5] = self.cat.cost[c] / 10.0, 1.0

        priv_ids = np.zeros(PRIV_IDS, np.int16)
        for j, c in enumerate(list(op.hand)[:HAND]):
            priv_ids[j] = c + 1 if c >= 0 else 0
        priv_ids[4] = op.next_card + 1 if op.next_card >= 0 else 0
        for j, c in enumerate(self.opp_deck[:8]):
            priv_ids[5 + j] = c + 1
        priv_f = np.zeros(PF, np.float32)
        priv_f[0] = op.elixir_milli / 10000.0
        priv_f[1] = op.pending_cost / 10.0
        priv_f[2] = float(any(int(r[0]) for r in op.abilities))
        priv_f[3] = elixir_balance(self.cat, st, s) / 20.0
        # one cap everywhere (acting, rollouts, shards, evaluation): units first, then spells,
        # then projectiles, in the order built above
        n = min(i, MAX_ENT)
        return Obs(g, cards, cardf, ids[:n], f[:n], priv_ids, priv_f, cand, tiles, cand_cost,
                   me.elixir_milli / 1000.0)


def label_supported(o: Obs, lab) -> bool:
    """True when the policy could produce this label from observation ``o`` (the decision at the
    window start): legal candidates and tiles, a distinct and affordable second play, ordered
    delays. Anything else is kept out of the policy loss."""
    c1, t1, d1, c2, t2, d2 = (int(x) for x in lab)
    if c1 < 0:
        return True
    if not o.cand[c1] or (c1 < HAND and (t1 < 0 or not o.tiles[c1][t1])):
        return False
    if c2 < 0:
        return True
    if c2 == c1 or not o.cand[c2] or d2 < d1 or (c2 < HAND and (t2 < 0 or not o.tiles[c2][t2])):
        return False
    cost = np.clip(o.cand_cost, 0, 255).astype(np.uint8).astype(np.float32)  # as stored
    return bool(cost[c2] <= np.float32(o.elixir) - cost[c1] + SECOND_PLAY_SLACK)


# --- action codec --------------------------------------------------------------------------


def own_tile(team: int, x_engine: int, y_engine: int) -> int:
    x, y = (W_SUB - x_engine, H_SUB - y_engine) if team == 1 else (x_engine, y_engine)
    tx = min(max(int(x // TILE), 0), NX - 1)
    ty = min(max(int(y // TILE), 0), NY - 1)
    return ty * NX + tx


def tile_center_engine(team: int, tile: int) -> tuple[int, int]:
    tx, ty = tile % NX, tile // NX
    x, y = tx * TILE + TILE // 2, ty * TILE + TILE // 2
    return (W_SUB - x, H_SUB - y) if team == 1 else (x, y)


def command(team: int, cand: int, tile: int):
    from royalegym.protocol import DeployCommand

    if cand >= HAND:
        return DeployCommand(team=team, hand_slot=HAND + (cand - HAND), x=0, y=0)
    x, y = tile_center_engine(team, tile)
    return DeployCommand(team=team, hand_slot=cand, x=x, y=y)


def nearest_legal_tile(mask: np.ndarray, tile: int, radius: int = 2) -> int:
    """``tile`` if legal, else the nearest legal tile within ``radius`` (Chebyshev), else -1."""
    if 0 <= tile < NT and mask[tile]:
        return tile
    tx, ty = tile % NX, tile // NX
    best, bd = -1, 1e9
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            x, y = tx + dx, ty + dy
            if 0 <= x < NX and 0 <= y < NY and mask[y * NX + x]:
                d = dx * dx + dy * dy
                if d < bd:
                    best, bd = y * NX + x, d
    return best


def potential(state, team: int) -> float:
    """Phi: crown difference + tower HP-loss fractions (enemy +, own -, king x3)."""
    me, op = state.players[team], state.players[1 - team]
    phi = float(me.crowns - op.crowns)
    for j in range(min(3, len(me.tower_hp))):
        w = 3.0 if j == 0 else 1.0
        phi -= w * (1.0 - me.tower_hp[j] / max(1, me.tower_max_hp[j]))
        phi += w * (1.0 - op.tower_hp[j] / max(1, op.tower_max_hp[j]))
    return phi
