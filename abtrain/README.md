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

## A, phase 2 (from the best A; B paused)

Phase 1's A stopped improving: it trained toward the 1 s live delay while every evaluation ran
at delay 0, and its self-play pool drifted (A vs FL Hog 2 fell from ~57% to 33%). Phase 2
(`a_phase: 2`, the default) restarts A's PPO from `checkpoints/A/best_vs_general.pt` (the 8M
snapshot, 0.73 vs FL General over 200 argmax games):

- The first supervisor start archives phase 1's A run (`runs/ppo_A`, `league/A`, `runs/A`,
  `runs/eval/A*`) to `archive/a_phase1_<time>/`, copies the best to
  `checkpoints/A/phase2_init.pt` (PPO start and KL reference; later promotions never touch
  it) and starts a new clock: the pipeline stops `a_phase_hours` (72) after that start.
- Delay 0 in training (`ppo_fixed_delay_A`), as in the evaluation.
- Opponents (`a_mix`): 20% the real FL General on its focus decks, 10% the real FL Hog 2 on
  Hog 2.6, 50% self-play and past snapshots, 20% A's stage-2 BC model and the hardest past
  snapshot. The stand-ins are no longer used. A's decks are as before (popular human decks;
  against FirstLight, popular decks whose cards FirstLight can name).
- The real FirstLight games run in their own env workers (`ppo_fl_workers` cores, each with
  `ppo_fl_envs_per_worker` battles, FirstLight on GPUs 2-3). The actor serves them whenever they are ready, so they
  never hold the simulator workers back. Their share of training decisions is therefore set
  by how many FirstLight battles there are (each moves one decision per actor round):
  `STATUS.md` and `actor_metrics.jsonl` (`decision_share`) show the real split. Raise or lower
  `ppo_fl_envs_per_worker` to move the FL share toward 30%.
- All 4 GPUs for A: actor GPU 0, learner GPU 1, FirstLight workers and the evaluator GPUs 2-3.
  A gets the cores B had: 40 of 48 (22 FirstLight workers with 6 battles each, 18 simulator
  workers with 12 battles each; 6 cores for the evaluator, 2 spare). Expect roughly 1.4M decisions/h.
- B is paused (`run_B: false`): no B PPO, and the evaluator screens A only.
- Evaluation: every game argmax (`fl_eval_decode`), 80 games on `mixed` and 40 on each other
  suite; phase 1's best (0.73 over 200 games) is the bar to beat. The start checkpoint is
  screened once (label `phase2_start`) for comparison.

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

Screens use 80 games on the primary suite and 40 on each other suite, all argmax
(`fl_eval_decode`; phase 1 screened at T=0.5 but recorded the best's argmax score, so its gate
compared two different measurements). A screen ≥ `promote_gate` (0.6) and within 3 points of
the best gets 160 more games; if the pooled win rate beats the best, the snapshot is promoted
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
