# flport: FirstLight checkpoints playing inside RoyaleGym

`flport` runs FirstLight_CR's published V4 checkpoints (Hog 2.6 specialists 1 and 2, General, IL)
as seats in a RoyaleGym / RoyaleSim battle. Each engine tick it rebuilds the FAIR `ObservationV1`
FirstLight's own tensorizer expects from the RoyaleSim `BattleState`. FirstLight's own
`PolicySessionV4` then decides, and its actions go back to the engine as `DeployCommand`s.
The model and its tensorizer are not modified.

See [REPORT.md](REPORT.md) for what maps exactly, what is approximated, and the measured results.

## Layout

| File | What it does |
|---|---|
| `flport/catalog.py` | Name-level join of RoyaleSim ids and FirstLight native GlobalIDs (cards, characters, projectiles, area effects) |
| `flport/observation.py` | `Scene`: one per battle. `ingest(state, played)` every tick, `observation(team)` builds `ObservationV1` |
| `flport/agent.py` | `FirstLightSeat`: one recurrent FirstLight episode on one seat, decides every 5 ticks from tick 90 |
| `flport/match.py` | Match runner and CLI, with seats alternating each game, Wilson intervals and JSON output |

## Setup

Tested with FirstLight_CR at `3ade158` and RoyaleGym at `ab0df8f` (royalegym 0.1.x with RustEngine,
calibration `client_overrides("15.535.29")`), Python 3.12, torch 2.14 on CPU.

```bash
git clone https://github.com/jaasssoooonnnnn/FirstLight_CR
python -m venv venv && source venv/bin/activate
pip install royalegym torch          # plus FirstLight_CR's own requirements
cd FirstLight_CR                     # checkpoint paths are relative to the FirstLight checkout
```

## Run

```bash
PYTHONPATH=.:/path/to/firstlight-port \
OMP_NUM_THREADS=1 python -m flport.match --a hog2 --b general --deck hog26 --games 12 --seed 1000 --out runs/x.json
```

- `--a/--b`: `hog2`, `hog1`, `general`, `il`, a checkpoint path, or `random` (RoyaleGym's TileActionParser,
  a random legal play whenever elixir is at least 7).
- `--deck`: `hog26` (FirstLight's specialist deck with its hero Musketeer and evo Cannon / Skeletons),
  `hog26-plain`, `giant-double`, or `a:b` for different decks.
- `FLPORT_MAX_TICKS` caps game length for quick tests.

Speed: about 800 s per full game per process on one CPU thread. Run one process per core with
`OMP_NUM_THREADS=1`. The time is almost all model forward passes (two 12M-param models, one decision
per seat every 5 ticks). A GPU or batching seats across games would be the next speedup.

## Using a seat from your own loop

```python
from flport.agent import FirstLightSeat, load, CHECKPOINTS
from flport.observation import Scene
scene = Scene(catalog, bundle, decks=decks, forms=forms, arena=engine.arena(), episode_id="g1")
seat = FirstLightSeat(team=1, policy=load(CHECKPOINTS["hog2"]), scene=scene)
# each tick: commands = seat.act(state.tick); results = engine.step(commands + yours, 1)
#            scene.ingest(engine.state(), played)   # played = accepted plays this tick, see match.play
```
