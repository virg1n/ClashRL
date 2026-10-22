"""Play FirstLight checkpoints inside RoyaleSim.

    python -m flport.match --a hog2 --b general --deck hog26 --games 20

Seats alternate each game, like FirstLight's own evaluator. A side is a FirstLight
checkpoint name (hog2, hog1, general, il, or a path) or ``random`` (RoyaleGym's
TileActionParser, a uniformly random legal play whenever elixir is at least 7).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

from royalegym.action import TileActionParser
from royalegym.protocol import HAND_SIZE, DeployStatus, MatchSetup, ShuffleMode
from royalegym.rust_engine import RustEngine, client_overrides

from native_runner.training.v4.factory import production_semantic_bundle

from .agent import CHECKPOINTS, FirstLightSeat, load
from .catalog import Catalog
from .observation import Scene

DECKS = {
    # FirstLight's Hog 2.6 specialist deck (checkpoints/README.md), with its forms
    "hog26": (
        ["HogRider", "Musketeer", "Cannon", "Fireball", "Log", "Skeletons", "IceGolemite", "IceSpirits"],
        [0, 2, 1, 0, 0, 1, 0, 0],
    ),
    "hog26-plain": (
        ["HogRider", "Musketeer", "Cannon", "Fireball", "Log", "Skeletons", "IceGolemite", "IceSpirits"],
        [0] * 8,
    ),
    "giant-double": (
        ["Giant", "Musketeer", "MiniPekka", "Arrows", "Fireball", "Knight", "Archer", "Minions"],
        [0] * 8,
    ),
}


def wilson(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = wins / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


class RandomSeat:
    def __init__(self, team, engine, rng, threshold=7.0):
        self.team, self.rng, self.threshold = team, rng, threshold
        self.parser = TileActionParser()
        self.parser.bind(engine)
        self.engine = engine
        self.stats = {"decisions": 0, "plays": 0, "abilities": 0, "errors": 0}
        self.scene = None

    def act(self, tick):
        if tick % 5 or tick < 90:
            return []
        state = self.scene.state
        if state.players[self.team].elixir_milli < self.threshold * 1000:
            return []
        mask = self.parser.action_mask(state, self.team)
        legal = [i for i in range(1, len(mask)) if mask[i]]
        if not legal:
            return []
        command = self.parser.parse(self.rng.choice(legal), state, self.team)
        self.stats["plays"] += command is not None
        return [command] if command is not None else []

    def end(self):
        pass


@dataclass
class Result:
    seed: int
    a_team: int
    winner: int  # -1 none, 0, 1, 2 draw
    ticks: int
    crowns: tuple[int, int]
    plays: tuple[int, int]
    rejected: tuple[int, int]
    reasons: dict
    unknown: dict
    seconds: float
    first_play: tuple = ()


def play(engine, catalog, bundle, sides, deck_names, forms, seed, a_team, max_ticks=7200, device="cpu"):
    max_ticks = int(__import__('os').environ.get('FLPORT_MAX_TICKS', max_ticks))
    rng = random.Random(seed)
    # Deal each deck unshuffled in a seeded random order, so the deal is still random but the
    # own cycle order FirstLight's tracker reconciles against is known exactly from the start.
    decks, dealt_forms = [], []
    for t in (0, 1):
        order = list(range(8))
        rng.shuffle(order)
        decks.append([catalog.rs_card_id(deck_names[t][i]) for i in order])
        dealt_forms.append([forms[t][i] for i in order])
    forms = dealt_forms
    setup = MatchSetup(decks=decks, forms=forms, shuffle=ShuffleMode.NONE)
    engine.reset(seed, setup)
    state = engine.state()
    scene = Scene(catalog, bundle, decks=decks, forms=forms, arena=engine.arena(), episode_id=f"rs-{seed}")
    scene.ingest(state)
    seats = []
    for team in (0, 1):
        side = sides[0] if team == a_team else sides[1]
        if side == "random":
            seat = RandomSeat(team, engine, rng)
            seat.scene = scene
        else:
            seat = FirstLightSeat(team, load(CHECKPOINTS.get(side, side), device), scene, device=device)
        seats.append(seat)
    rejected = [0, 0]
    reasons: dict = {}
    started = time.perf_counter()
    while not state.game_over and state.tick < max_ticks:
        commands = []
        for seat in seats:
            commands.extend(seat.act(state.tick))
        evo_ready = [{int(r[0]) for r in state.players[t].evo if len(r) > 2 and r[2]} for t in (0, 1)]
        hand = [list(state.players[t].hand) for t in (0, 1)]
        results = engine.step(commands, 1)
        played = []
        for command, result in zip(commands, results):
            if result.status != DeployStatus.OK:
                rejected[command.team] += 1
                reasons[str(result.status)] = reasons.get(str(result.status), 0) + 1
                continue
            if command.hand_slot >= HAND_SIZE:
                played.append((result.tick, command.team, -1 - (command.hand_slot - HAND_SIZE), 0, 0, 0))
                continue
            card = hand[command.team][command.hand_slot]
            form = scene.form_of(command.team, card)
            code = 2 if form == 2 else (1 if card in evo_ready[command.team] else 0)
            played.append((result.tick, command.team, card, code, result.x, result.y))
        state = engine.state()
        scene.ingest(state, played)
    for seat in seats:
        seat.end()
    plays = tuple(seats[t].stats["plays"] + seats[t].stats["abilities"] for t in (0, 1))
    return Result(
        seed=seed,
        a_team=a_team,
        winner=int(state.winner),
        ticks=int(state.tick),
        crowns=(state.players[0].crowns, state.players[1].crowns),
        plays=plays,
        rejected=tuple(rejected),
        reasons=reasons,
        unknown=dict(scene.unknown_archetypes),
        seconds=time.perf_counter() - started,
        first_play=tuple(seats[t].stats.get("first_play_tick") for t in (0, 1)),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", default="hog2")
    parser.add_argument("--b", default="general")
    parser.add_argument("--deck", default="hog26", help="deck for both sides, or a:b")
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    args = parser.parse_args(argv)
    engine = RustEngine(calibration_overrides=client_overrides("15.535.29"))
    catalog = Catalog.build(engine)
    bundle = production_semantic_bundle()
    deck_a, deck_b = (args.deck.split(":") + [args.deck])[:2]
    a_wins = b_wins = draws = 0
    rows = []
    for game in range(args.games):
        a_team = game % 2
        names = [None, None]
        forms = [None, None]
        names[a_team], forms[a_team] = DECKS[deck_a]
        names[1 - a_team], forms[1 - a_team] = DECKS[deck_b]
        result = play(engine, catalog, bundle, (args.a, args.b), names, forms, args.seed + game, a_team, device=args.device)
        if result.winner == result.a_team:
            a_wins += 1
        elif result.winner in (0, 1):
            b_wins += 1
        else:
            draws += 1
        rows.append(result.__dict__)
        print(json.dumps(result.__dict__), flush=True)
    low, high = wilson(a_wins, a_wins + b_wins + draws)
    summary = {
        "a": args.a, "b": args.b, "deck": args.deck, "games": args.games,
        "a_wins": a_wins, "b_wins": b_wins, "draws": draws,
        "a_win_rate": a_wins / max(1, args.games), "wilson95": [low, high],
    }
    print(json.dumps(summary), flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"summary": summary, "games": rows}, indent=1))


if __name__ == "__main__":
    main()
