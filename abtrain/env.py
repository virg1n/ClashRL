"""One battle at decision level, for PPO, sim evaluation and the FirstLight bridge.

``Battle.step([lab_seat0, lab_seat1])`` runs one 250 ms window: each seat's first and second
play are sent at their delay tick inside the window (under the engine's command delay, if
any), the engine runs tick by tick, and both seats get rewards and their next observation.

Reward per seat (FirstLight's base, ``abtrain/README.md``):
  terminal +1 / -1 / 0;  0.05 x (gamma Phi' - Phi), Phi = crowns + tower HP-loss fractions
  (king x3), Phi' = 0 at the end;  two-clock elixir overflow penalty (personal, linear past a
  grace; unilateral, quadratic while the opponent is not also full), capped per step;
  B only: c_E x (gamma E' - E), E = own elixir + board worth - the enemy's, E' = 0 at the end.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .game import (DT, ELIXIR_TICKS, HAND, Catalog, Obs, SeatView, command, elixir_balance,
                   legal_masks, make_engine, make_parser, potential)

MAX_TICKS = 12000
_TGT_COLS = (16, 17, 18, 19, 20, 39, 40, 41, 45)


@dataclass
class SeatSpec:
    deck: list  # 8 (name, form) in dealt order
    elixir_shaping: bool = False


@dataclass
class BattleSpec:
    seats: tuple  # (SeatSpec, SeatSpec)
    delay: int = 0  # command delay in ticks, both seats
    seed: int = 0
    feature_drop: float = 0.0
    meta: dict = field(default_factory=dict)  # opaque, returned with the result


@dataclass
class RewardCfg:
    gamma: float = 0.9997
    shaping: float = 0.05
    personal: float = 0.001
    unilateral: float = 0.0005
    grace: float = 1.0
    cap: float = 0.1
    elixir_coef: float = 0.01


class Battle:
    def __init__(self, cat: Catalog | None = None, rcfg: RewardCfg | None = None):
        self.engine = make_engine()
        self.parser = make_parser(self.engine)
        self.cat = cat or Catalog.build(self.engine)
        self.rcfg = rcfg or RewardCfg()
        self.views = [SeatView(self.cat, self.parser), SeatView(self.cat, self.parser)]
        self.rng = np.random.default_rng(0)
        self.spec: BattleSpec | None = None

    def reset(self, spec: BattleSpec) -> list[Obs]:
        from royalegym.protocol import MatchSetup, ShuffleMode

        self.spec = spec
        self.rng = np.random.default_rng(spec.seed)
        if hasattr(self.engine, "set_command_delay_ticks"):
            self.engine.set_command_delay_ticks(int(spec.delay))
        decks = [[self.cat.ids[n] for n, _ in s.deck] for s in spec.seats]
        forms = [[f for _, f in s.deck] for s in spec.seats]
        self.engine.reset(int(spec.seed) & (2**63 - 1), MatchSetup(decks=decks, forms=forms, shuffle=ShuffleMode.NONE))
        for s in (0, 1):
            self.views[s].reset(s, decks[s], forms[s], decks[1 - s], delay_ticks=spec.delay)
        self.state = self.engine.state()
        self.phi = [potential(self.state, s) for s in (0, 1)]
        self.ebal = [elixir_balance(self.cat, self.state, s) for s in (0, 1)]
        self.waste = [0.0, 0.0]
        self.uni = [0.0, 0.0]
        self.done = False
        self.stats = {"plays": [0, 0], "first_play": [-1, -1], "refused": [0, 0], "overflow": [0.0, 0.0],
                      "elixir_trade": [0.0, 0.0], "steps": 0}
        return self._observe()

    def _observe(self) -> list[Obs]:
        out = []
        for s in (0, 1):
            cand, tiles = legal_masks(self.parser, self.state, s)
            o = self.views[s].build(self.state, cand, tiles)
            if self.spec.feature_drop > 0 and len(o.ent_ids):
                p = self.spec.feature_drop
                keep = ~((o.ent_ids[:, 2] == 5) & (self.rng.random(len(o.ent_ids)) < p))
                o.ent_ids, o.ent_f = o.ent_ids[keep], o.ent_f[keep]
                cut = self.rng.random(len(o.ent_f)) < p
                if cut.any():
                    o.ent_f[np.ix_(cut, _TGT_COLS)] = 0.0
            out.append(o)
        return out

    def step(self, labs) -> tuple[list[Obs] | None, list[float], bool, dict]:
        """labs: per seat (c1, t1, d1, c2, t2, d2), -1 = none. Returns (obs, rewards, done, info)."""
        from royalegym.protocol import DeployStatus

        assert not self.done
        st0 = self.state
        tick0 = int(st0.tick)
        sends: list[tuple[int, int, object]] = []
        for s in (0, 1):
            lab = [int(x) for x in labs[s]]
            for j in (0, 1):
                c, t, d = lab[3 * j : 3 * j + 3]
                if c < 0:
                    continue
                if c < HAND and t < 0:
                    continue
                sends.append((tick0 + max(0, min(d, DT - 1)), s, command(s, c, t)))
        sends.sort(key=lambda x: (x[0], x[1]))
        delayed = self.engine.command_delay_ticks != (0, 0) if hasattr(self.engine, "command_delay_ticks") else False
        for k in range(DT):
            now = tick0 + k
            cmds = [c for t, _, c in sends if t == now]
            hands = [list(self.state.players[s].hand) for s in (0, 1)] if cmds else None
            results = self.engine.step(cmds, 1)
            if delayed:
                for _t, team, kind, what, reason in (self.engine.step_commands_run() or []):
                    if reason == 0 and kind == "deploy":
                        self._ran(team, what)
            for c, r in zip(cmds, results):
                if r.status != DeployStatus.OK:
                    self.stats["refused"][c.team] += 1
                    continue
                if not delayed and c.hand_slot < HAND:
                    self._ran(c.team, hands[c.team][c.hand_slot])
            if cmds or k == DT - 1:
                self.state = self.engine.state()
            if self.state.game_over:
                break
        st = self.state
        self.stats["steps"] += 1
        g = self.rcfg.gamma
        done = bool(st.game_over) or int(st.tick) >= MAX_TICKS
        rewards = [0.0, 0.0]
        dticks = int(st.tick) - tick0
        full = [st.players[s].elixir_milli >= 10000 and st0.players[s].elixir_milli >= 10000 for s in (0, 1)]
        for s in (0, 1):
            phi = 0.0 if done else potential(st, s)
            r = self.rcfg.shaping * (g * phi - self.phi[s])
            self.phi[s] = phi
            # elixir overflow, FirstLight's two clocks
            wasted = dticks * float(st.elixir_rate) / ELIXIR_TICKS if full[s] else 0.0
            pen = 0.0
            if full[s]:
                self.waste[s] += wasted
                if self.waste[s] > self.rcfg.grace:
                    pen += self.rcfg.personal * wasted
                if not full[1 - s]:
                    u0 = self.uni[s]
                    self.uni[s] += wasted
                    pen += self.rcfg.unilateral * (self.uni[s] ** 2 - u0 ** 2)
                else:
                    self.uni[s] = 0.0
            else:
                self.waste[s] = 0.0
                self.uni[s] = 0.0
            pen = min(pen, self.rcfg.cap)
            self.stats["overflow"][s] += wasted
            r -= pen
            e = 0.0 if done else elixir_balance(self.cat, st, s)
            if self.spec.seats[s].elixir_shaping:
                r += self.rcfg.elixir_coef * (g * e - self.ebal[s])
            self.stats["elixir_trade"][s] += (e - self.ebal[s]) if not done else 0.0
            self.ebal[s] = e
            rewards[s] = r
        info = {}
        if done:
            self.done = True
            w = int(st.winner) if st.game_over else 2
            res = (1, -1) if w == 0 else (-1, 1) if w == 1 else (0, 0)
            for s in (0, 1):
                rewards[s] += float(res[s])
            info = {"result": res, "ticks": int(st.tick), "crowns": (st.players[0].crowns, st.players[1].crowns),
                    **{k: (list(v) if isinstance(v, list) else v) for k, v in self.stats.items()}}
            return None, rewards, True, info
        return self._observe(), rewards, False, info

    def _ran(self, team: int, card_id: int) -> None:
        if card_id is None or card_id < 0:
            return
        for v in self.views:
            v.note_play(team, card_id)
        self.stats["plays"][team] += 1
        if self.stats["first_play"][team] < 0:
            self.stats["first_play"][team] = int(self.state.tick)
