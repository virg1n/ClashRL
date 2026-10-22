# A (Atlas) and B (Hog) training

Code for the A/B plan in `docs/ab-training.md` (dpo thread), for the 4× RTX 6000 Ada server.
One command runs everything unattended for 2–3 days and resumes after any interruption.

## Run it

```bash
git clone <this repo> && cd ClashRL
bash run_ab.sh            # first run builds the env, then everything runs in the background
bash run_ab.sh status     # progress report
bash run_ab.sh stop       # stop cleanly; `bash run_ab.sh` again resumes
```

Everything is written under `AB_WORK` (default `~/clashrl_ab`, the network home) and
`/dev/shm/clashrl_ab_$USER` (shards and PPO rollouts in RAM). Nothing goes to the root disk:
pip/uv/HF/torch caches and `TMPDIR` point into `AB_WORK` (`abtrain/env.sh`).

`setup_ab.sh` (called by `run_ab.sh`) needs no sudo: it fetches `uv`, builds a Python 3.12
env in `AB_WORK/venv` with `abtrain/requirements.txt` (torch from PyPI), and clones
FirstLight_CR at `3ade158` into `AB_WORK/ext/FirstLight_CR`.

Overrides: put any key of `DEFAULTS` in `abtrain/common.py` into `AB_WORK/config.json`
(flat JSON). Example: `{"deadline_hours": 70}` to run longer. The deadline counts from the
first start, so restarts don't extend it.

## What runs

```
main  download (HF, pinned, checksummed) -> extract (46 workers) -> stage shards to /dev/shm
      -> stage 1: plain BC + win/draw/loss head, 1 pass, ALL 4 GPUs (DDP)
      -> calibration check -> V_ref over every shard (4 GPUs)
      -> stage 2 for A: 0.3 plain + 0.7 advantage-weighted BC, 1.5 passes, ALL 4 GPUs
      -> checkpoints/A/bc_stage2_human_only.pt (kept, read-only) -> A/B test vs stage 1
         (200 sim games; stage 2 must score >= 0.53, or a missing result counts as a fail and
         A's PPO starts from stage 1)
      -> split:  GPUs 0-1  PPO A (actor GPU 0, learner GPU 1, ~19 env cores)
                 GPUs 2-3  stage 2 for B (Hog 2.6 + 30% general) -> checkpoints/B/... -> A/B test
                           -> PPO B
fl    after extraction: record FirstLight games through flport (FL General, FL Hog 2)
      -> after stage 1: clone them into our network (the stand-ins) -> check each against the
      real model -> league/standins/{general,hog2}.pt (A's and B's league keep them; the
      lane keeps retrying every 30 min until both are in)
eval  after extraction: FL Hog 2 vs FL General baseline in our simulator
      -> screen stage 1/2 and every PPO snapshot vs the real FL General, promote the best
```

Each stage is a subprocess with retries, a hang detector (no log/heartbeat activity for 40
min) and a log in `AB_WORK/logs/<stage>.out`. Stage state is in `AB_WORK/state.json`; a
finished stage is never rerun. Inside a stage, BC checkpoints every 12 min (`latest.pt`) and
PPO every 12 min, so a restart loses at most that much. Stopping (`bash run_ab.sh stop`, the
deadline, a hang) signals each stage's own process first, so PPO writes a checkpoint before
its process group is killed; a supervisor that crashed is restarted by `run_ab.sh`'s loop and
first kills any stage processes it left behind.

## Where results land

| Path (under `AB_WORK`) | What |
|---|---|
| `STATUS.md` | progress report, refreshed every 2 min |
| `checkpoints/A/bc_stage1_human_only.pt` | stage 1 (human BC + win-prob head) |
| `checkpoints/A/bc_stage2_human_only.pt` | **stage 2, human matches only, before any PPO or FirstLight exposure. Never overwritten (read-only). Model D reuses its win-prob head.** |
| `checkpoints/B/bc_stage2_human_only.pt` | B's stage 2 (Hog 2.6 human matches) |
| `checkpoints/{A,B}/best_vs_general.pt` (+ `.json`) | best promoted PPO snapshot vs FL General, with its numbers and chosen decoding |
| `checkpoints/{A,B}/ppo_final.pt` | last PPO weights |
| `runs/eval/{A,B}.jsonl`, `runs/eval/fl_baseline.json` | every screen vs FL General; the FL Hog 2 baseline |
| `runs/ppo_{A,B}/learner_metrics.jsonl`, `actor_metrics.jsonl` | PPO curves, games by opponent, anti-passivity |
| `league/` | snapshots, BC models, stand-ins, promoted checkpoints |

## Evaluation against FirstLight General

FirstLight's own protocol (`docs/training_history.md`): argmax FirstLight, no forced first
action, seat-swapped pairs with the same seed and decks, no command delay. Two suites:

- `hog26`: both seats on Hog 2.6 with its forms. FL Hog 2 scored 88% vs General here in
  FirstLight's engine. B's primary suite.
- `mixed`: both seats on the same FirstLight focus deck. FL Hog 2 scored 59% on mixed decks.
  A's primary suite.

The goal is above FL Hog's ~85%. `runs/eval/fl_baseline.json` measures FL Hog 2 vs General
in our simulator under the same protocol, so the bar is known in this simulator too.

Screens use 40 games on the primary suite and 20 on the other (decoding: card/tile at T=0.5,
gate sampled). A screen ≥ `promote_gate` (0.6) and within 3 points of the best gets 160 more
games; if the pooled win rate beats the best, the same number of games is played with argmax
cards/tiles, and the better decoder is kept with its own statistics: the snapshot is promoted
(league hard pool + `best_vs_general.pt`). `goal_reached` is set when the 95% lower bound
clears 0.85.
Anti-passivity is logged on every screen (plays per game, first play, share of wins where
the opponent made ≤3 plays).

## PPO details

- PPO: clip 0.1, 2 epochs, approx-KL early stop at 1.5× 0.01, AdamW lr 1e-5 (eps 1e-5,
  wd 0), γ 0.9997, λ 0.98, grad norm 0.5, value coef 0.5, entropy 0.003 → 0.001.
  BPTT chunks of 64 decisions with 16 burn-in from stored LSTM states; 40 s segments with
  the LSTM state carried across.
- Critic warm-up: policy frozen until the explained variance settles.
- **KL to the stage-2 model is kept low (by request):** coefficient starts at 0.01, is capped
  at 0.05, adapts to a KL budget of 0.03 that relaxes toward 0.1 only while the FL General
  screens improve, and fades out after the nominal run (64M for A, 48M for B).
- Command delay ramps 0 → 21 ticks (1.05 s) over the first 20%, then mostly near 21; B's
  second half is all live latency.
- Reward: ±1 terminal, 0.05·(γΦ′−Φ) tower/crown shaping, FirstLight's two-clock elixir
  overflow penalty. B only: elixir-balance shaping 0.01·(γE′−E). No card-specific rewards.
- B: a human Hog BC auxiliary loss (0.2 → 0 over the first 25%).
- PPO minibatches are split under the same token budget as BC (one late-game state with many
  units pads the whole batch); B's BC auxiliary batch has its own backward pass.
- Policy inputs leave out per-unit timers the live bridge does not validate (attack phase,
  charge, windup); entities are capped at 128 the same way for acting, training and eval.
- League (A): 40% self-play, 40% PFSP frozen pool, 20% hard (FL stand-ins, hardest snapshot);
  own deck by popularity, opponent decks 65% popular / 30% FL General's focus decks / 5% hardest.
  League (B): Hog 2.6 always; 10% mirror self-play, 70% A's and B's snapshots on popular decks,
  20% hard (FL-Hog stand-in mirror, FL-General stand-in, hardest).

Not run, for lack of spare GPUs: the elixir-balance control copy of B.

## Local smoke test

Same one command on a CPU box, with CPU torch and a small `AB_WORK/config.json` (for example
`extract_limit_tasks`, `bc_max_steps`, `max_decisions_A/B`, `fl_*_games` set tiny) and one
parquet file copied into `AB_WORK/data/IL_Replay/replays` with an empty `.complete` next to it:

```bash
AB_WORK=~/abtest/work AB_TORCH_INDEX=https://download.pytorch.org/whl/cpu AB_FL_MAX_TICKS=1200 bash run_ab.sh fg
```
