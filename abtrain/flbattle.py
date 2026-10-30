"""A PPO battle against a real FirstLight model (through ``flport``), in an env worker.

Same decision-level interface as ``env.Battle``: ``step([lab0, lab1])`` runs one 250 ms window
and returns both seats' rewards. Only the learner's seat is ours; the FirstLight seat gets no
observation (``None``) and its label is ignored: FirstLight decides on its own every 5 ticks
inside the window from flport's scene, which ingests the state every tick, as in
``flmatch.play_game``. FirstLight plays argmax, as in its benchmark and our evaluation, without
command delay (its own protocol). The learner's seat may have one (``SeatSpec.delay``).

``spec.meta``: ``fl`` = the FirstLight model (``general`` / ``hog2``), ``learner_seat``.
"""

from __future__ import annotations

from .env import Battle, BattleSpec
from .game import DT, HAND


class FLBattle(Battle):
    def __init__(self, cat=None, rcfg=None, device: str = "cpu"):
        super().__init__(cat, rcfg)
        from .flmatch import fl_setup

        fl_setup(self.engine, device)
        self.fl_seat = None
        self.scene = None
        self.pending = {}  # (team, card) -> (form code, x, y) of a delayed play waiting to run

    def reset(self, spec: BattleSpec):
        from flport.observation import Scene

        from .flmatch import _W, FLSeat

        if spec.delay:
            raise ValueError("FirstLight battles run without command delay")
        self._end_fl()
        self.pending = {}
        learner = int(spec.meta["learner_seat"])
        fl_team = 1 - learner
        self.observed = (learner,)
        obs = super().reset(spec)
        decks = [[self.cat.ids[n] for n, _ in s.deck] for s in spec.seats]
        forms = [[f for _, f in s.deck] for s in spec.seats]
        self.scene = Scene(_W["flcat"], _W["bundle"], decks=decks, forms=forms, arena=self.engine.arena(),
                           episode_id=f"ppo-{spec.seed}")
        self.scene.ingest(self.state)
        self.fl_seat = FLSeat(fl_team, spec.meta["fl"], self.scene, sample=False, force_first=False)
        return obs

    def _window(self, tick0: int, sends: list) -> None:
        from royalegym.protocol import HAND_SIZE, DeployStatus

        for k in range(DT):
            now = tick0 + k
            st = self.state
            cmds = self._due(sends, now) + self.fl_seat.act(now, st)
            if cmds:
                hands = [list(st.players[s].hand) for s in (0, 1)]
                evo_ready = [{int(x[0]) for x in st.players[s].evo if len(x) > 2 and x[2]} for s in (0, 1)]
            results = self.engine.step(cmds, 1)
            played = []
            if any(self.delays):
                for run_tick, team, kind, what, reason in (self.engine.step_commands_run() or []):
                    if reason != 0:
                        self.pending.pop((team, what), None)
                    elif kind == "deploy" and (team, what) in self.pending:
                        code, x, y = self.pending.pop((team, what))
                        played.append((run_tick, team, what, code, x, y))
                        self._ran(team, what, run_tick)
                    elif kind == "ability":
                        played.append((run_tick, team, -1 - (what - HAND_SIZE), 0, 0, 0))
            for c, r in zip(cmds, results):
                if r.status != DeployStatus.OK:
                    self.stats["refused"][c.team] += 1
                    continue
                if c.hand_slot >= HAND_SIZE:  # a hero ability button
                    if not self.delays[c.team]:
                        played.append((r.tick, c.team, -1 - (c.hand_slot - HAND_SIZE), 0, 0, 0))
                    continue
                card = hands[c.team][c.hand_slot]
                code = 2 if self.scene.form_of(c.team, card) == 2 else (1 if card in evo_ready[c.team] else 0)
                if self.delays[c.team]:  # accepted; it runs (and is seen) later
                    self.pending[(c.team, card)] = (code, r.x, r.y)
                    continue
                played.append((r.tick, c.team, card, code, r.x, r.y))
                if c.hand_slot < HAND:
                    self._ran(c.team, card, now)
            self.state = self.engine.state()
            self.scene.ingest(self.state, played)
            if self.state.game_over:
                break

    def _rewards(self, st0, tick0: int):
        out = super()._rewards(st0, tick0)
        if out[2]:
            self._end_fl()
        return out

    def _end_fl(self) -> None:
        if self.fl_seat is not None:
            try:
                self.fl_seat.end()
            except Exception:
                pass
            self.fl_seat = None
