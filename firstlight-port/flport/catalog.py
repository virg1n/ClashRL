"""Name-level identity map between RoyaleSim and FirstLight.

RoyaleSim and FirstLight both decode the same Supercell csv_logic tables, so a
card, a character, a projectile and an area effect carry the same NAME on both
sides. FirstLight keys everything by native GlobalID (26/27/28M for cards, 34M
or form ids for characters, 10M for projectiles, 22M for area effects);
RoyaleSim keys by catalogue index. This module joins the two by name, once.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path

import royalesim

from native_runner.training.v4.factory import production_semantic_bundle

def _fl_data_dir() -> Path:
    import native_runner

    return Path(native_runner.__file__).resolve().parent / "data" / "competitive"


@dataclass
class Catalog:
    rs_cards: list  # RoyaleSim CardInfo rows, by catalogue index
    unit_types: list[str]
    fl_card_by_rs: dict[int, int] = field(default_factory=dict)
    rs_card_by_fl: dict[int, int] = field(default_factory=dict)
    fl_char_by_unit_type: dict[int, int] = field(default_factory=dict)
    projectile_gid: dict[str, int] = field(default_factory=dict)
    area_gid: dict[str, int] = field(default_factory=dict)
    # RoyaleSim card name -> the projectile its unit fires / its spell flies as / rolls as / area
    unit_projectile_by_card: dict[int, str] = field(default_factory=dict)
    spell_flight_by_card: dict[int, str] = field(default_factory=dict)
    spell_roll_by_card: dict[int, str] = field(default_factory=dict)
    spell_area_by_card: dict[int, str] = field(default_factory=dict)

    @classmethod
    def build(cls, engine) -> "Catalog":
        bundle = production_semantic_bundle()
        cat = cls(rs_cards=list(engine.cards()), unit_types=list(engine.unit_types() or []))
        fl_by_name = {spec.name: int(card_id) for card_id, spec in bundle.card_specs.items()}
        for card in cat.rs_cards:
            fl_id = fl_by_name.get(card.name)
            if fl_id is not None:
                cat.fl_card_by_rs[card.card_id] = fl_id
                cat.rs_card_by_fl[fl_id] = card.card_id

        models = json.load(gzip.open(_fl_data_dir() / "model_catalogs.json.gz"))
        arche = models["catalogs"]["entity_archetype_catalog"]
        chars = {name: int(gid) for gid, name in arche["character_global_ids"]}
        for index, name in enumerate(cat.unit_types):
            if name in chars:
                cat.fl_char_by_unit_type[index] = chars[name]
        for key, node in zip(arche["archetype_keys"], arche["archetype_node_ids"], strict=True):
            kind, _, raw = key.partition(":")
            namespace, _, name = node.partition(".")
            if kind == "projectile" and namespace == "PROJECTILE":
                cat.projectile_gid[name] = int(raw)
            elif kind == "area" and namespace == "AEO":
                cat.area_gid[name] = int(raw)

        table = json.loads((Path(royalesim.data_dir()) / "derived" / "cards-15.535.json").read_text())
        rows = {row["name"]: row for row in table["cards"]}
        units = table.get("units", {})
        for card in cat.rs_cards:
            row = rows.get(card.name)
            if row is None:
                continue
            unit = units.get(row.get("summon_character") or "", {})
            shot = unit.get("projectile") if isinstance(unit, dict) else None
            if isinstance(shot, dict) and shot.get("name"):
                cat.unit_projectile_by_card[card.card_id] = shot["name"]
            elif isinstance(shot, str):
                cat.unit_projectile_by_card[card.card_id] = shot
            flight = row.get("projectile")
            if isinstance(flight, dict) and flight.get("name"):
                cat.spell_flight_by_card[card.card_id] = flight["name"]
                roll = flight.get("spawn_projectile")
                if isinstance(roll, dict) and roll.get("name"):
                    cat.spell_roll_by_card[card.card_id] = roll["name"]
            spell = row.get("spell") or {}
            area = spell.get("area_effect_object") if isinstance(spell, dict) else None
            if isinstance(area, dict) and area.get("name"):
                cat.spell_area_by_card[card.card_id] = area["name"]
        return cat

    def rs_card_id(self, name: str) -> int:
        for card in self.rs_cards:
            if card.name == name:
                return card.card_id
        raise KeyError(name)
