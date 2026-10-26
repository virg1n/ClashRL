"""RoyaleSim ``BattleState`` -> FirstLight ``ObservationV1``.

One ``Scene`` per battle. Feed it every engine tick with ``ingest`` (the state after
the tick, and what was played during it); ask it for either seat's FAIR observation
with ``observation``.

Frames. RoyaleSim's engine frame is absolute, team 0 at the bottom, 18000 subtiles per
tile. FirstLight's native frame is absolute, owner 0 at the bottom, 1000 units per
tile. So FirstLight = RoyaleSim / 18 on both axes and owner == team.

What RoyaleSim does not report, and how it is filled (see README):
  * causal groups: one deployment group per play, matched by (team, card, birth tick);
    later spawns of the same card are a spawn wave per birth tick.
  * combat events: spawn, death, damage and tower damage, derived from state diffs.
    Attack start/hit, target change, effect apply/remove and projectile events are
    not synthesized.
  * per-unit timers the native probe reads (cooldown left, attack phase time left,
    movement speed, charge progress, capture, relocation): left unset (None).
  * the own deck cycle: tracked from plays; the order of the three cards behind the
    next card is unknown until each is played once and is filled in deck order.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field
from typing import Iterable

from royalegym.protocol import (
    EMPTY_CARD,
    HAND_SIZE,
    STATUS_EVOLVED,
    STATUS_HERO,
    STATUS_INVISIBLE,
    STATUS_UNDERGROUND,
    BattleState,
    EntityKind,
    SpellMotion,
    ability_row,
    status_of,
)

from native_runner.arena import OccupiedFootprintV1, card_placement_mask, native_building_footprint
from native_runner.building_placement_profiles import building_placement_profile
from native_runner.contracts import (
    AbilityPhase,
    AbilityRuntimeStateV1,
    ActionKind,
    ActionMaskV1,
    AttackPhase,
    AttackStateV1,
    CausalGroupKind,
    CausalGroupRefV1,
    CombatEventKind,
    CombatEventV1,
    EffectStateV1,
    EntityStateV1,
    EventV1,
    EvolutionPhase,
    EvolutionRuntimeStateV1,
    ObservationTier,
    ObservationV1,
    PlayerStateV1,
    ProjectilePhase,
    ProjectileStateV1,
    TimeStateV1,
    TowerStateV1,
    VisibilityPhase,
    VisibilityStateV1,
    SemanticEvidenceLevel,
    SemanticProvenanceV1,
    ENTITY_RUNTIME_SEMANTIC_FIELDS,
    PLAYER_RUNTIME_SEMANTIC_FIELDS,
    TOWER_RUNTIME_SEMANTIC_FIELDS,
    EVENT_RUNTIME_SEMANTIC_FIELDS,
    ATTACK_STATE_FIELDS,
    PROJECTILE_STATE_FIELDS,
    EVOLUTION_RUNTIME_STATE_FIELDS,
    ABILITY_RUNTIME_STATE_FIELDS,
    EFFECT_STATE_FIELDS,
    VISIBILITY_STATE_FIELDS,
    COMBAT_EVENT_FIELDS,
)

from .catalog import Catalog, _fl_data_dir

SCALE = 18  # RoyaleSim subtiles per FirstLight native unit
TICK_MS = 50
GAMEPLAY_END_TICK = 6000  # FirstLight's model clock: five minutes of 50 ms ticks
PRINCESS_TOWER_TROOP_ID = 159_000_000
RULESET_ID = __import__("hashlib").sha256(b"royalesim-15.535.29").hexdigest()
SPELL_FLIGHT_MOTIONS = {int(SpellMotion.FLIGHT), int(SpellMotion.AIRBORNE)}
ROLLING = int(SpellMotion.ROLLING)
EVENT_KEEP_TICKS = 20
BOTH_SEE = {"0": True, "1": True}


def _present(value) -> bool:
    if value is None:
        return False
    if isinstance(value, (tuple, list, dict, set, frozenset)):
        return bool(value)
    return True


def _prov(fields, values: dict) -> SemanticProvenanceV1:
    """Provenance that claims exactly the fields this adapter filled (derived from RoyaleSim)."""
    return SemanticProvenanceV1(
        field_evidence={
            name: (SemanticEvidenceLevel.NATIVE_DERIVED if _present(values.get(name)) else SemanticEvidenceLevel.UNKNOWN)
            for name in fields
        }
    )


def _make(cls, slot: str, fields, **values):
    return cls(**values, **{slot: _prov(fields, values)})


def fl_xy(x: float, y: float) -> tuple[float, float]:
    return (float(x) / SCALE, float(y) / SCALE)


def _buff_gids() -> dict[str, int]:
    effects = json.load(gzip.open(_fl_data_dir() / "effects.json.gz"))
    return {str(name): int(gid) for name, gid in effects["global_id_registry"]["buff_entries"].items()}


@dataclass
class _Play:
    tick: int
    team: int
    rs_card: int
    form: int
    x: int
    y: int
    handle: str


@dataclass
class _SeatMemory:
    queue: list[int] = field(default_factory=list)  # RS card ids not in hand, next card first
    last_hand: tuple[int, ...] = ()
    revealed: set[int] = field(default_factory=set)  # FL ids this seat has publicly played


class Scene:
    def __init__(self, catalog: Catalog, bundle, *, decks, forms, arena, episode_id: str):
        self.catalog = catalog
        self.bundle = bundle
        self.specs = bundle.card_specs
        self.decks = [list(map(int, d)) for d in decks]  # RS ids
        self.forms = [list(map(int, f)) for f in forms]
        self.arena = arena
        self.episode_id = episode_id
        self.buff_gid = _buff_gids()
        self.state: BattleState | None = None
        self.tick = 0
        self.birth: dict[int, int] = {}
        self.last_pos: dict[int, tuple[int, int, int]] = {}
        self.last_hp: dict[int, tuple[int, int, int, int, bool]] = {}  # uid -> hp, team, rs card, unit type, tower
        self.group_of: dict[int, CausalGroupRefV1] = {}
        self.plays: list[_Play] = []
        self.events: list[EventV1] = []
        self.seat = [_SeatMemory(), _SeatMemory()]
        self.unknown_archetypes: dict[str, int] = {}
        self.unknown_cards: set[int] = set()
        self.tower_uid: dict[tuple[int, int], int] = {}
        self._mask_cache: dict = {}
        # projectiles and spells have no engine id: each (team, card) gets its own block of
        # synthetic ids, so an id never names two cards (FirstLight's public tracker binds
        # (owner, entity id) to a hero card and raises if that card changes)
        self.synth_block: dict[tuple[int, object], int] = {}

    # ------------------------------------------------------------------ helpers
    def fl_card(self, rs_card: int) -> int | None:
        if rs_card is None or rs_card < 0:
            return None
        fl = self.catalog.fl_card_by_rs.get(int(rs_card))
        if fl is None:
            self.unknown_cards.add(int(rs_card))
        return fl

    def form_of(self, team: int, rs_card: int) -> int:
        deck = self.decks[team]
        return self.forms[team][deck.index(rs_card)] if rs_card in deck else 0

    def _note_unknown(self, label: str) -> None:
        self.unknown_archetypes[label] = self.unknown_archetypes.get(label, 0) + 1

    # ------------------------------------------------------------------ ingest
    def ingest(self, state: BattleState, played: Iterable[tuple[int, int, int, int, int, int]] = ()) -> None:
        """``played``: (tick, team, rs_card, form_code, x, y) of every card play and button
        press (rs_card = -1 - button for a press) that RAN during the ticks just stepped."""

        self.tick = int(state.tick)
        for tick, team, rs_card, form, x, y in played:
            self._record_play(int(tick), int(team), int(rs_card), int(form), int(x), int(y), state)
        self._diff(state)
        self._update_cycles(state)
        self.state = state
        cutoff = self.tick - EVENT_KEEP_TICKS
        self.events = [event for event in self.events if event.tick > cutoff]

    def _record_play(self, tick, team, rs_card, form, x, y, state) -> None:
        if rs_card < 0:  # an ability button press
            button = -1 - rs_card
            rows = state.players[team].abilities
            card = ability_row(rows[button]).card_id if button < len(rows) else EMPTY_CARD
            fl = self.fl_card(card)
            hero = next(
                (
                    e.uid
                    for e in state.entities
                    if e.team == team and e.card_id == card and (status_of(e) or 0) & STATUS_HERO
                ),
                None,
            )
            self.events.append(
                _make(EventV1, "runtime_provenance", EVENT_RUNTIME_SEMANTIC_FIELDS,
                    tick=tick,
                    event_type="ability_activation",
                    owner=team,
                    entity_id=hero,
                    card_id=fl,
                    data={"fair_ability_activation_exact": True, "source_card_id": fl, "kind": "activate_ability"},
                )
            )
            return
        fl = self.fl_card(rs_card)
        handle = f"play:{team}:{tick}:{rs_card}"
        self.plays.append(_Play(tick, team, rs_card, form, x, y, handle))
        self.plays = self.plays[-64:]
        if fl is None:
            return
        self.seat[team].revealed.add(fl)
        self.events.append(
            _make(EventV1, "runtime_provenance", EVENT_RUNTIME_SEMANTIC_FIELDS,
                tick=tick,
                event_type="action_executed",
                owner=team,
                card_id=fl,
                position=fl_xy(x, y),
                data={"kind": "play_card", "form_code": int(form), "action_id": handle},
            )
        )

    def _group_for(self, uid: int, team: int, rs_card: int, birth: int) -> CausalGroupRefV1 | None:
        fl = self.fl_card(rs_card)
        for play in reversed(self.plays):
            if play.team == team and play.rs_card == rs_card and 0 <= birth - play.tick <= 2:
                return CausalGroupRefV1(kind=CausalGroupKind.DEPLOYMENT, handle=play.handle, source_card_id=fl)
        return CausalGroupRefV1(
            kind=CausalGroupKind.SPAWN_WAVE, handle=f"wave:{team}:{rs_card}:{birth}", source_card_id=fl
        )

    def _combat(self, kind, tick, owner, *, target=None, target_card=None, source_card=None, amount=None, pos=None):
        return _make(EventV1, "runtime_provenance", EVENT_RUNTIME_SEMANTIC_FIELDS,
            tick=tick,
            event_type=kind.value if hasattr(kind, "value") else str(kind),
            owner=owner,
            entity_id=target,
            card_id=source_card,
            position=pos,
            combat=_make(CombatEventV1, "provenance", COMBAT_EVENT_FIELDS,
                kind=kind if isinstance(kind, CombatEventKind) else CombatEventKind.UNKNOWN,
                target_entity=target,
                target_card_id=target_card,
                source_card_id=source_card,
                amount=amount,
                position=pos,
                visible_by_owner=BOTH_SEE,
            )
            if isinstance(kind, CombatEventKind)
            else None,
            data={"amount": amount} if amount is not None else {},
        )

    def _diff(self, state: BattleState) -> None:
        tick = int(state.tick)
        seen: set[int] = set()
        for e in state.entities:
            seen.add(e.uid)
            tower = e.kind in (EntityKind.KING_TOWER, EntityKind.PRINCESS_TOWER)
            if e.uid not in self.birth:
                self.birth[e.uid] = tick
                if not tower:
                    self.group_of[e.uid] = self._group_for(e.uid, e.team, e.card_id, tick)
                    self.events.append(
                        self._combat(
                            CombatEventKind.SPAWN, tick, e.team, target=e.uid, source_card=self.fl_card(e.card_id),
                            pos=fl_xy(e.x, e.y),
                        )
                    )
            previous = self.last_hp.get(e.uid)
            if previous is not None and e.hp < previous[0]:
                amount = float(previous[0] - e.hp)
                if tower:
                    self.events.append(
                        _make(EventV1, "runtime_provenance", EVENT_RUNTIME_SEMANTIC_FIELDS, tick=tick, event_type="tower_damage", owner=e.team, entity_id=e.uid,
                                position=fl_xy(e.x, e.y), data={"amount": amount})
                    )
                else:
                    self.events.append(
                        self._combat(
                            CombatEventKind.DAMAGE, tick, e.team, target=e.uid,
                            target_card=self.fl_card(e.card_id), amount=amount, pos=fl_xy(e.x, e.y),
                        )
                    )
            self.last_hp[e.uid] = (e.hp, e.team, e.card_id, e.unit_type, tower)
        for uid in [uid for uid in self.last_hp if uid not in seen]:
            hp, team, card, _unit, tower = self.last_hp.pop(uid)
            pos = self.last_pos.get(uid)
            where = fl_xy(pos[0], pos[1]) if pos else None
            if tower:
                self.events.append(_make(EventV1, "runtime_provenance", EVENT_RUNTIME_SEMANTIC_FIELDS, tick=tick, event_type="tower_destroyed", owner=team, entity_id=uid,
                                           position=where))
            else:
                self.events.append(
                    self._combat(CombatEventKind.DEATH, tick, team, target=uid, source_card=self.fl_card(card),
                                 pos=where)
                )
        # velocity memory is read before this overwrite, in observation(); keep the previous frame
        self._prev_pos = dict(self.last_pos)
        for e in state.entities:
            self.last_pos[e.uid] = (e.x, e.y, tick)

    def _update_cycles(self, state: BattleState) -> None:
        for team in (0, 1):
            player = state.players[team]
            hand = tuple(card for card in player.hand)
            memory = self.seat[team]
            if not memory.queue and not memory.last_hand:
                rest = [card for card in self.decks[team] if card not in hand]
                if player.next_card in rest:
                    rest.remove(player.next_card)
                    rest.insert(0, player.next_card)
                memory.queue = rest
            else:
                for card in memory.last_hand:
                    if card != EMPTY_CARD and card not in hand and card not in memory.queue:
                        memory.queue.append(card)
                memory.queue = [card for card in memory.queue if card not in hand]
                if player.next_card in memory.queue and memory.queue[0] != player.next_card:
                    memory.queue.remove(player.next_card)
                    memory.queue.insert(0, player.next_card)
                for card in self.decks[team]:
                    if card not in hand and card not in memory.queue:
                        memory.queue.append(card)
            memory.last_hand = hand

    # ------------------------------------------------------------------ build
    def _towers(self, state: BattleState) -> tuple[TowerStateV1, ...]:
        result = []
        present = {}
        for e in state.entities:
            if e.kind == EntityKind.KING_TOWER:
                present[(e.team, 0)] = e
            elif e.kind == EntityKind.PRINCESS_TOWER:
                present[(e.team, int(e.tower_slot))] = e
        for team in (0, 1):
            player = state.players[team]
            for slot in (0, 1, 2):
                e = present.get((team, slot))
                if slot == 0:
                    x, y = self.arena.king_centers[team]
                else:
                    x, y = self.arena.princess_centers[team][slot - 1]
                if e is not None:
                    x, y = e.x, e.y
                    self.tower_uid[(team, slot)] = e.uid
                uid = self.tower_uid.setdefault((team, slot), 900_000 + team * 10 + slot)
                kind = "king" if slot == 0 else ("princess_left" if x < self.arena.subtile * 9 else "princess_right")
                hp = float(e.hp) if e is not None else 0.0
                max_hp = float(player.tower_max_hp[slot]) if player.tower_max_hp else (float(e.max_hp) if e else 1.0)
                attack = None
                if e is not None and e.attack_phase >= 0:
                    attack = _make(AttackStateV1, "provenance", ATTACK_STATE_FIELDS,
                        phase=(AttackPhase.IDLE, AttackPhase.WINDUP, AttackPhase.COOLDOWN)[min(e.attack_phase, 2)],
                        target_entity=e.target_uid if e.target_uid >= 0 else None,
                    )
                result.append(
                    _make(TowerStateV1, "runtime_provenance", TOWER_RUNTIME_SEMANTIC_FIELDS,
                        entity_id=uid,
                        owner=team,
                        tower_kind=kind,
                        position=fl_xy(x, y),
                        hitpoints=min(hp, max_hp),
                        max_hitpoints=max_hp,
                        tower_troop_id=None if slot == 0 else PRINCESS_TOWER_TROOP_ID,
                        active=hp > 0,
                        visible_target=(e.target_uid if e is not None and e.target_uid >= 0 else None),
                        status=(("activated",) if slot == 0 and player.king_active else ()),
                        attack_state=attack,
                    )
                )
        return tuple(result)

    def _synth_id(self, used: dict, team: int, card) -> int:
        """Unique within the tick; the same id always belongs to the same (team, card)."""
        key = (int(team), card)
        block = self.synth_block.setdefault(key, len(self.synth_block))
        k = used.get(key, 0)
        used[key] = k + 1
        return 5_000_000 + block * 1000 + k

    def _entities(self, state: BattleState) -> tuple[EntityStateV1, ...]:
        tick = int(state.tick)
        items: list[EntityStateV1] = []
        prev = getattr(self, "_prev_pos", {})
        for e in state.entities:
            if e.kind in (EntityKind.KING_TOWER, EntityKind.PRINCESS_TOWER):
                continue
            status = status_of(e) or 0
            fl = self.fl_card(e.card_id)
            gid = self.catalog.fl_char_by_unit_type.get(int(e.unit_type))
            if gid is None:
                unit_name = (
                    self.catalog.unit_types[e.unit_type] if 0 <= e.unit_type < len(self.catalog.unit_types) else "?"
                )
                self._note_unknown(f"unit:{unit_name}")
            if e.kind == EntityKind.BUILDING:
                kind = "building"
            elif status & STATUS_HERO:
                kind = "hero"
            else:
                kind = "troop"
            velocity = None
            before = prev.get(e.uid)
            if before is not None and tick > before[2]:
                dt = tick - before[2]
                velocity = ((e.x - before[0]) / SCALE / dt, (e.y - before[1]) / SCALE / dt)
            attack = None
            if e.attack_phase >= 0:
                attack = _make(AttackStateV1, "provenance", ATTACK_STATE_FIELDS,
                    phase=(AttackPhase.IDLE, AttackPhase.WINDUP, AttackPhase.COOLDOWN)[min(e.attack_phase, 2)],
                    target_entity=e.target_uid if e.target_uid >= 0 else None,
                )
            evolution = None
            if status & STATUS_EVOLVED and fl is not None:
                unit_name = self.catalog.unit_types[e.unit_type] if 0 <= e.unit_type < len(self.catalog.unit_types) else None
                evolution = _make(EvolutionRuntimeStateV1, "provenance", EVOLUTION_RUNTIME_STATE_FIELDS,
                    card_id=fl, phase=EvolutionPhase.EVOLVED, active=True, current_form_id=unit_name
                )
            visibility = None
            if status & STATUS_UNDERGROUND:
                visibility = _make(VisibilityStateV1, "provenance", VISIBILITY_STATE_FIELDS, phase=VisibilityPhase.BURROWED)
            elif status & STATUS_INVISIBLE:
                visibility = _make(VisibilityStateV1, "provenance", VISIBILITY_STATE_FIELDS, phase=VisibilityPhase.HIDDEN)
            effects = []
            for name, ms_left in e.buffs:
                base = name.split("|", 1)[0]
                gid_buff = self.buff_gid.get(base)
                if gid_buff is None:
                    self._note_unknown(f"buff:{base}")
                    continue
                effects.append(
                    _make(EffectStateV1, "provenance", EFFECT_STATE_FIELDS,
                        effect_id=f"buff:{gid_buff}",
                        remaining_ms=int(ms_left),
                        active=True,
                        attributes={"native_buff_global_id": gid_buff},
                    )
                )
            items.append(
                _make(EntityStateV1, "runtime_provenance", ENTITY_RUNTIME_SEMANTIC_FIELDS,
                    entity_id=int(e.uid),
                    owner=int(e.team),
                    card_id=fl,
                    entity_kind=kind,
                    position=fl_xy(e.x, e.y),
                    velocity=velocity,
                    hitpoints=float(e.hp),
                    max_hitpoints=float(e.max_hp),
                    shield=float(e.shield),
                    age_ms=(tick - self.birth.get(e.uid, tick)) * TICK_MS,
                    visible_target=e.target_uid if e.target_uid >= 0 else None,
                    native_data_global_id=gid,
                    causal_group=self.group_of.get(e.uid),
                    attack_state=attack,
                    deployment_runtime={
                        "version": "deployment-runtime.v1",
                        "phase": "deploying" if e.deploy_ticks > 0 else "active",
                        "remaining_native_ms": max(0, int(e.deploy_ticks)) * TICK_MS,
                        "remaining_wall_ms": None,
                        "configured_deploy_time_ms": None,
                        "observed_step_native_ms": TICK_MS,
                        "observed_tick": tick,
                        "complete": True,
                    },
                    evolution_state=evolution,
                    visibility_state=visibility,
                    effect_states=tuple(effects),
                )
            )
        used: dict = {}
        for p in state.projectiles:
            if p.firer_card_id >= 0:
                name = self.catalog.unit_projectile_by_card.get(int(p.firer_card_id))
                fl = self.fl_card(p.firer_card_id)
            else:
                king_y = self.arena.king_centers[p.team][1]
                near_king = abs(p.y - king_y) < self.arena.subtile * 2.5
                name = "KingProjectile" if near_king else "TowerPrincessProjectile"
                fl = None
            next_id = self._synth_id(used, p.team, fl if fl is not None else name)
            gid = self.catalog.projectile_gid.get(name or "")
            if gid is None:
                self._note_unknown(f"projectile:{name}")
            items.append(
                _make(EntityStateV1, "runtime_provenance", ENTITY_RUNTIME_SEMANTIC_FIELDS,
                    entity_id=next_id,
                    owner=int(p.team),
                    card_id=fl,
                    entity_kind="projectile",
                    position=fl_xy(p.x, p.y),
                    native_data_global_id=gid,
                    projectile_state=_make(ProjectileStateV1, "provenance", PROJECTILE_STATE_FIELDS,
                        projectile_id=f"p{next_id}",
                        phase=ProjectilePhase.IN_FLIGHT,
                        source_card_id=fl,
                        target_entity=p.target_uid if p.target_uid >= 0 else None,
                        target_position=fl_xy(p.aim_x, p.aim_y),
                        radius_tiles=(p.splash / self.arena.subtile) if p.splash > 0 else None,
                        attributes=({"native_projectile_data_global_id": gid} if gid is not None else {}),
                    ),
                )
            )
        for s in state.spells:
            fl = self.fl_card(s.card_id)
            next_id = self._synth_id(used, s.team, fl if fl is not None else f"spell:{int(s.card_id)}")
            motion = int(s.motion)
            if motion in SPELL_FLIGHT_MOTIONS or motion == ROLLING:
                table = self.catalog.spell_roll_by_card if motion == ROLLING else self.catalog.spell_flight_by_card
                name = table.get(int(s.card_id)) or self.catalog.spell_flight_by_card.get(int(s.card_id))
                gid = self.catalog.projectile_gid.get(name or "")
                if gid is None:
                    self._note_unknown(f"spell-projectile:{name}:{s.card_id}")
                items.append(
                    _make(EntityStateV1, "runtime_provenance", ENTITY_RUNTIME_SEMANTIC_FIELDS,
                        entity_id=next_id,
                        owner=int(s.team),
                        card_id=fl,
                        entity_kind="projectile",
                        position=fl_xy(s.x, s.y),
                        native_data_global_id=gid,
                        projectile_state=_make(ProjectileStateV1, "provenance", PROJECTILE_STATE_FIELDS,
                            projectile_id=f"s{next_id}",
                            phase=ProjectilePhase.IN_FLIGHT,
                            source_card_id=fl,
                            target_position=fl_xy(s.aim_x, s.aim_y),
                            attributes=({"native_projectile_data_global_id": gid} if gid is not None else {}),
                        ),
                    )
                )
            else:
                name = self.catalog.spell_area_by_card.get(int(s.card_id))
                gid = self.catalog.area_gid.get(name or "")
                if gid is None:
                    self._note_unknown(f"area:{name}:{s.card_id}")
                items.append(
                    _make(EntityStateV1, "runtime_provenance", ENTITY_RUNTIME_SEMANTIC_FIELDS,
                        entity_id=next_id,
                        owner=int(s.team),
                        card_id=fl,
                        entity_kind="area",
                        position=fl_xy(s.x, s.y),
                        native_data_global_id=gid,
                    )
                )
        return tuple(items)

    def _evolution_states(self, team: int) -> tuple[EvolutionRuntimeStateV1, ...]:
        player = self.state.players[team]
        result = []
        for row in player.evo:
            card, plays, ready, need = (list(row) + [0, 0, 0, 0])[:4]
            fl = self.fl_card(card)
            if fl is None:
                continue
            deck = self.decks[team]
            result.append(
                _make(EvolutionRuntimeStateV1, "provenance", EVOLUTION_RUNTIME_STATE_FIELDS,
                    card_id=fl,
                    deck_slot=deck.index(card) if card in deck else None,
                    phase=EvolutionPhase.READY if ready else EvolutionPhase.CYCLING,
                    cycle_required=int(need) if need else None,
                    cycle_remaining=0 if ready else max(0, int(need) - int(plays)),
                    ready=bool(ready),
                    active=False,
                )
            )
        return tuple(result)

    def _hero_sources(self, team: int) -> dict[int, tuple[int, object, int]]:
        """button -> (hero uid, ability row, rs card) for this side's live hero buttons."""
        state = self.state
        result = {}
        for button, raw in enumerate(state.players[team].abilities):
            row = ability_row(raw)
            hero = next(
                (
                    e
                    for e in state.entities
                    if e.team == team and e.card_id == row.card_id and (status_of(e) or 0) & STATUS_HERO
                ),
                None,
            )
            if hero is not None:
                result[button] = (hero.uid, row, row.card_id)
        return result

    def _ability_states(self, team: int) -> tuple[AbilityRuntimeStateV1, ...]:
        result = []
        for button, (uid, row, rs_card) in self._hero_sources(team).items():
            fl = self.fl_card(rs_card)
            spec = self.specs.get(fl) if fl is not None else None
            if spec is None or len(spec.ability_ids) != 1:
                continue
            if row.spent:
                phase = AbilityPhase.EXHAUSTED
            elif row.available:
                phase = AbilityPhase.READY
            else:
                phase = AbilityPhase.UNAVAILABLE
            result.append(
                _make(AbilityRuntimeStateV1, "provenance", ABILITY_RUNTIME_STATE_FIELDS,
                    ability_id=spec.ability_ids[0],
                    source_entity=int(uid),
                    phase=phase,
                    elixir_cost=float(row.cost),
                    charges=0 if row.spent else 1,
                    available=bool(row.available),
                    remaining_cooldown_ms=(max(0, row.cooldown_ticks) * TICK_MS if row.cooldown_ticks >= 0 else None),
                    attributes={"controller_slot": int(button), "source_card_id": int(fl)},
                )
            )
        return tuple(result)

    def hand_runtime(self, team: int) -> dict[int, dict]:
        """slot -> FirstLight hand-runtime contract for this side's hand."""
        player = self.state.players[team]
        evo_ready = {int(row[0]) for row in player.evo if len(row) > 2 and row[2]}
        result = {}
        for slot, card in enumerate(player.hand):
            fl = self.fl_card(card)
            if fl is None:
                continue
            form = self.form_of(team, card)
            form_code = 2 if form == 2 else (1 if card in evo_ready else 0)
            cost = float(self.specs[fl].elixir_cost)
            result[slot] = {
                "hand_slot": slot,
                "visible_card_id": fl,
                "effective_card_id": fl,
                "native_effective_card_id": fl,
                "effective_cost": cost,
                "form_code": form_code,
                "native_form_code": form_code,
            }
        return result

    def _action_mask(self, team: int, towers, entities, reserved: float) -> ActionMaskV1:
        player = self.state.players[team]
        elixir = player.elixir_milli / 1000.0
        effective = max(0.0, elixir - reserved)
        runtime = self.hand_runtime(team)
        lanes = tuple(
            "left" if t.tower_kind == "princess_left" else "right"
            for t in towers
            if t.owner != team and not t.active and t.tower_kind != "king"
        )
        footprints = tuple(
            OccupiedFootprintV1(t.position[0], t.position[1], 4 if t.tower_kind == "king" else 3,
                                4 if t.tower_kind == "king" else 3, f"tower:{t.entity_id}")
            for t in towers
            if t.active
        )
        occupied = []
        for ent in entities:
            if ent.entity_kind != "building" or ent.card_id is None:
                continue
            spec = self.specs.get(ent.card_id)
            footprint = native_building_footprint(spec) if spec is not None else None
            profile = building_placement_profile(ent.card_id)
            if footprint is None or (profile is not None and not profile.stationary_collision_rectangle):
                continue
            occupied.append(OccupiedFootprintV1(ent.position[0], ent.position[1], footprint.width_tiles,
                                                footprint.height_tiles, f"building:{ent.entity_id}"))
        slots, masks, reasons = [], {}, {}
        for slot in range(HAND_SIZE):
            entry = runtime.get(slot)
            if entry is None:
                slots.append(False)
                reasons[str(slot)] = "not_in_hand"
                continue
            fl = entry["visible_card_id"]
            form = "evolution" if entry["form_code"] == 1 else "base"
            is_building = self.specs[fl].kind.value == "building"
            key = (fl, team, form, lanes, footprints if is_building else len(footprints),
                   tuple(occupied) if is_building else ())
            base = self._mask_cache.get(key)
            if base is None:
                artifact = card_placement_mask(
                    self.specs[fl], team, ruleset_id=RULESET_ID, form=form,
                    destroyed_enemy_princess_lanes=lanes, active_tower_footprints=footprints,
                    occupied_building_footprints=tuple(occupied),
                )
                base = {
                    "card_id": fl,
                    "shape": (32, 18),
                    "row_major": artifact.rows,
                    "placement_rule": artifact.rule.value,
                    "accuracy": artifact.accuracy.value,
                    "mask_id": artifact.mask_id,
                    "collision_radius_units": artifact.collision_radius_units,
                    "footprint_width_tiles": artifact.footprint_width_tiles,
                    "footprint_height_tiles": artifact.footprint_height_tiles,
                    "model_subcell_offset": artifact.model_subcell_offset,
                    "form": artifact.form,
                    "any_legal": any(any(row) for row in artifact.rows),
                }
                if len(self._mask_cache) > 4096:
                    self._mask_cache.clear()
                self._mask_cache[key] = base
            masks[str(slot)] = {**{k: v for k, v in base.items() if k != "any_legal"}, **entry}
            legal = entry["effective_cost"] <= effective + 1e-9 and base["any_legal"]
            reasons[str(slot)] = "legal" if legal else "insufficient_elixir"
            slots.append(legal)
        sources = []
        for state in self._ability_states(team):
            if state.available and state.elixir_cost is not None and state.elixir_cost <= effective + 1e-9:
                sources.append(int(state.source_entity))
        return ActionMaskV1(
            kinds={
                ActionKind.WAIT.value: True,
                ActionKind.WAIT_UNTIL_AFFORDABLE.value: False,
                ActionKind.PLAY_CARD.value: any(slots),
                ActionKind.ACTIVATE_ABILITY.value: bool(sources),
            },
            hand_slots=tuple(slots),
            placement_masks=masks,
            ability_sources=tuple(sorted(sources)),
            reasons={
                "decision_ready": True,
                "slots": reasons,
                "raw_elixir": elixir,
                "reserved_elixir": reserved,
                "effective_elixir": effective,
            },
        )

    def observation(self, team: int, *, reserved_elixir: float = 0.0) -> ObservationV1:
        state = self.state
        towers = self._towers(state)
        entities = self._entities(state)
        players = []
        crowns = [state.players[t].crowns for t in (0, 1)]
        for t in (0, 1):
            p = state.players[t]
            if t == team:
                runtime = self.hand_runtime(t)
                hand = tuple(runtime[s]["visible_card_id"] for s in sorted(runtime))
                players.append(
                    _make(PlayerStateV1, "runtime_provenance", PLAYER_RUNTIME_SEMANTIC_FIELDS,
                        owner=t,
                        crowns=crowns[t],
                        tower_ids=tuple(x.entity_id for x in towers if x.owner == t),
                        elixir_exact=p.elixir_milli / 1000.0,
                        elixir_visible=p.elixir_milli / 1000.0,
                        hand=hand,
                        next_card=self.fl_card(p.next_card),
                        deck=tuple(self.fl_card(c) for c in self.decks[t]),
                        cycle=tuple(self.fl_card(c) for c in self.seat[t].queue),
                        revealed_cards=tuple(sorted(self.seat[t].revealed)),
                        private_state_visible=True,
                        metadata={
                            "hand_slot_count": len(hand),
                            "hand_slot_by_card": {str(runtime[s]["visible_card_id"]): s for s in runtime},
                            "hand_runtime_by_slot": {str(s): runtime[s] for s in runtime},
                            "native_hand_capacity": 4,
                        },
                        ability_runtime_states=self._ability_states(t),
                        evolution_runtime_states=self._evolution_states(t),
                    )
                )
            else:
                players.append(
                    _make(PlayerStateV1, "runtime_provenance", PLAYER_RUNTIME_SEMANTIC_FIELDS,
                        owner=t,
                        crowns=crowns[t],
                        tower_ids=tuple(x.entity_id for x in towers if x.owner == t),
                        revealed_cards=tuple(sorted(self.seat[t].revealed)),
                    )
                )
        tick = int(state.tick)
        return ObservationV1(
            tier=ObservationTier.FAIR,
            tick=tick,
            owner=team,
            time=TimeStateV1(
                elapsed_ms=tick * TICK_MS,
                remaining_ms=max(0, GAMEPLAY_END_TICK - tick) * TICK_MS,
                elixir_multiplier=float(state.elixir_rate),
                tick_ms=TICK_MS,
            ),
            phase="overtime" if state.overtime else "regular",
            players=tuple(players),
            towers=towers,
            entities=entities,
            events=tuple(e for e in self.events if e.tick <= tick),
            action_mask=self._action_mask(team, towers, entities, reserved_elixir),
            episode_id=self.episode_id,
            ruleset_id=RULESET_ID,
            metadata={"entity_overflow_count": 0},
        )
