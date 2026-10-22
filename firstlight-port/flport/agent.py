"""A FirstLight V4 checkpoint playing one RoyaleSim seat."""

from __future__ import annotations

from dataclasses import dataclass, field

from royalegym.protocol import HAND_SIZE, DeployCommand

from native_runner.arena import cell_to_world
from native_runner.contracts import ActionKind
from native_runner.training.v4.expert import FIRST_POLICY_DECISION_TICK, POLICY_DECISION_TICKS
from native_runner.training.v4.factory import _public_tracker_v4
from native_runner.training.v4.policy_session import PolicySessionV4, load_policy_v4
from native_runner.training.v4.tensorizer import UniversalObservationTensorizerV4

from .observation import SCALE, Scene

CHECKPOINTS = {
    "hog2": "checkpoints/2_6hog_expert/hog26-specialist2.pt",
    "hog1": "checkpoints/2_6hog_expert/hog26-specialist1.pt",
    "general": "checkpoints/General/checkpoint-step-00000460.pt",
    "il": "checkpoints/IL/checkpoint-step-00029396.pt",
}

_LOADED: dict[str, object] = {}


def load(checkpoint: str, device: str = "cpu"):
    key = f"{checkpoint}@{device}"
    if key not in _LOADED:
        _LOADED[key] = load_policy_v4(checkpoint, device=device)
    return _LOADED[key]


def form_availability(forms: list[int]) -> list[int]:
    """RoyaleSim forms (0 base, 1 evolution, 2 hero) -> FirstLight role bits (1 evo, 2 hero)."""
    return [1 if f == 1 else 2 if f == 2 else 0 for f in forms]


@dataclass
class _Pending:
    due_tick: int  # the engine tick the command must be sent at (it runs during due_tick + 1)
    command: DeployCommand
    cost: float


@dataclass
class FirstLightSeat:
    """One recurrent FirstLight episode on one seat. ``act`` is called every engine tick and
    returns the commands to send to the engine this tick."""

    team: int
    policy: object
    scene: Scene
    sample: bool = False
    device: str = "cpu"
    pending: list[_Pending] = field(default_factory=list)
    stats: dict = field(default_factory=lambda: {"decisions": 0, "plays": 0, "abilities": 0, "errors": 0})

    def __post_init__(self) -> None:
        bundle = self.scene.bundle
        fl_decks = tuple(tuple(self.scene.fl_card(c) for c in deck) for deck in self.scene.decks)
        roles = tuple(
            {card: (bool(mask & 2), bool(mask & 1)) for card, mask in zip(fl_decks[o], form_availability(self.scene.forms[o]))}
            for o in (0, 1)
        )
        self.tensorizer = UniversalObservationTensorizerV4(
            actor_owner=self.team,
            card_catalog=bundle.card_catalog,
            ability_catalog=bundle.ability_catalog,
            entity_archetype_catalog=bundle.entity_archetype_catalog,
            effect_catalog=bundle.effect_catalog,
            card_specs=bundle.card_specs,
            deck=fl_decks[self.team],
            tracker=_public_tracker_v4(bundle, fl_decks, roles),
            deck_roles=roles[self.team],
            # FirstLight refuses anything it cannot name; here an unnamed object is
            # tensorized as UNKNOWN and counted in Scene.unknown_archetypes instead.
            reject_unknown_public_semantics=False,
        )
        self.session = PolicySessionV4(self.policy.model, self.tensorizer, device=self.device, sample=self.sample)
        self.started = False

    def reserved(self) -> float:
        return sum(p.cost for p in self.pending)

    def act(self, tick: int) -> list[DeployCommand]:
        out = []
        for p in [p for p in self.pending if p.due_tick <= tick]:
            self.pending.remove(p)
            out.append(p.command)
        if tick % POLICY_DECISION_TICKS != 0:
            return out
        observation = self.scene.observation(self.team, reserved_elixir=self.reserved())
        if not self.started:
            players = self.scene.state.players
            self.session.start_episode(
                observation, initial_elixir={o: players[o].elixir_milli / 1000.0 for o in (0, 1)}
            )
            self.started = True
        if tick < FIRST_POLICY_DECISION_TICK:
            self.tensorizer.tensorize(observation, validate=False)
            return out
        decision = self.session.decide(observation)
        self.stats["decisions"] += 1
        hero_buttons = {uid: button for button, (uid, _row, _card) in self.scene._hero_sources(self.team).items()}
        for action in decision.decoded.actions:
            due = tick + max(1, int(action.execute_offset_ticks or 1)) - 1
            if action.kind == ActionKind.PLAY_CARD:
                wx, wy = cell_to_world(action.target_grid)
                if action.subcell_offset is not None:
                    wx += int(round(action.subcell_offset[0] * 1000))
                    wy += int(round(action.subcell_offset[1] * 1000))
                wx, wy = min(max(wx, 0), 17_999), min(max(wy, 0), 31_999)
                command = DeployCommand(self.team, int(action.hand_slot), wx * SCALE, wy * SCALE)
                cost = float(action.metadata.get("policy_effective_cost", 0.0))
                self.stats["plays"] += 1
                self.stats.setdefault("first_play_tick", tick)
            elif action.kind == ActionKind.ACTIVATE_ABILITY:
                button = hero_buttons.get(int(action.source_entity))
                if button is None:
                    self.stats["errors"] += 1
                    continue
                command = DeployCommand(self.team, HAND_SIZE + button, 0, 0)
                cost = 0.0
                self.stats["abilities"] += 1
            else:
                continue
            if due <= tick:
                out.append(command)
            else:
                self.pending.append(_Pending(due, command, cost))
        return out

    def end(self) -> None:
        if self.started:
            self.session.end_episode()
