# Four models to train in parallel

Written 2026-10-15. Sources: your earlier work in `C:\Users\bogda\Documents\temp\CR` (the 14 Oct report, `v2/`, `remote/cr/`), the project plan (`plan/clashrl-plan.md`), FirstLight_CR's docs and code (`docs/training_history.md`, `native_runner/training/v4/config.py`, `train_ppo_self_play_cluster.py`), other public Clash Royale RL work, and a second opinion from gpt-6-astra through codex-cli. Anything marked *estimate* has not been measured.

## 1. The short version

| | Model | Bet | Params | Starts from | Machine |
|---|---|---|---|---|---|
| **A** | **Atlas**: general entity + LSTM policy | More data and a better representation beat FL General on any deck | ~7M | Scratch, BC on all ~220k replayable matches | Server B: 2× Ada, 14 cores |
| **B** | **Hog**: Hog 2.6 specialist, same network as A | A well-piloted Hog mirror is what beats FL Hog 2.6 specialist 2 | ~7M | A's BC checkpoint (ours), then Hog-only BC | Server B: 2× Ada, 14 cores |
| **C** | **Anchor**: r12 lineage, counter-Hog training | Keep the one model that already beats FL General, and fix its Hog weakness | ~1.5M | Your r12 (via a transfer-safe self-distill) | Server A: 1× V100, 5 cores |
| **D** | **Oracle**: search-distilled policy | Lookahead in a forkable simulator finds better moves than PPO alone | ~3M | Scratch BC, then expert iteration | Server A: 2× V100, 9 cores |

A and B are the main line and follow FirstLight's own recipe (IL, then PPO with a league, then a specialist with outcome-only reward), moved into RoyaleSim. C is the cheap, safe hedge built on your best result so far. D is the high-ceiling research bet. None of them starts from a FirstLight checkpoint; FirstLight models appear only as frozen opponents and benchmarks.

## 2. What the earlier work tells us

- **r12 is a real result.** A 1.5M ResNet clone plus 16M steps of PPO with a KL-to-clone penalty beat FL General 45–35 (56%) on its own Goblin Giant / Heal / Rage deck. r11 at 12M params did no better, and BC log-loss was identical at 1.5M, 8.5M and 12M.
- **Its gap is Hog.** FL Hog 2.6 specialist 2 beats r12 51–9 and our Hog clone 55–5. Our Hog RL run started from a weak early clone and failed.
- **Fine-tuning FirstLight in RoyaleGym hurt it** (`v2/fl_rl`): after 136 iterations it beat General 70% against the original's 82%. You've ruled this path out anyway.
- **The v2 entity BC model lost 0/40 to both FL models.** That alone is not evidence against entities. A BC-only clone is expected to be weak: our bcD1 clone won only 18% against General, and FirstLight's own IL loses 73% to General. But 0 out of 80 is below even that, so a bug is likely on top (coordinate frame, mask, LSTM state reset, gate class weight 8 causing over-play, or too few steps). **Audit before scaling:** play v2 BC against bcD1 in sim. If it loses heavily to a clone of similar quality, it's a bug, not the architecture.
- **Sampling matters.** Card and tile at temperature 0.5 beat plain sampling 65%. Training on both seats of a match helped.
- **FirstLight's lessons:** win rate against a passive opponent is exploitable (296 of 306 wins came from games where IL made ≤3 plays). Their best specialist came from removing Hog-specific rewards. PPO was very conservative: lr 1e-5, clip 0.1, 2 epochs, target KL 0.01.

Other public work is thinner. KataCR (wty-yy) does vision plus offline RL on the real game and reached about 8,000 trophies. The oracle-planner paper (arXiv 2012.12186) trained a Clash-Royale-like policy by imitating a full-information planner. A community deterministic simulator post reports a 1-ply lookahead lifting win rate against a heuristic bot from 62.5% to 94.4%. Those last two are the basis for model D.

## 3. Shared rules for all four models

**Observation (transfer-safe only).** Only fields the live Nulls bridge can supply: visible units and buildings (type, card, owner, x/y, HP/max HP, deploy, stun, shield), towers, own hand, next card, elixir, clock, overtime, crowns, publicly revealed enemy cards, enemy elixir counted from observed plays, previous own actions and pending commands. **No spell or projectile channels** (zero live, non-zero in sim) and no `Reveal` options. A CI test builds the observation from the same recorded battle via RoyaleSim and the FirstLight adapter and requires matching tensors.

**Critic exception (where I disagree with astra).** The value head may additionally see privileged sim state such as enemy hand and elixir, because it is never deployed. That is standard asymmetric actor-critic and helps the value fit in a partially observed game. Keep it in a separate value-only branch so nothing privileged leaks into the policy.

**Action.** `Discrete(2305)` (wait + 4 slots × 18×32 tiles) with the exact legality mask, plus one ability button for hero decks. The ability is on in sim and in the FirstLight engine, and masked off live until the bridge supports it. Use one play per decision and no duplicate commands while one is pending.

**Timing and robustness.**
- 250 ms decisions for A, B and D. C stays at 500 ms, as r12 was trained.
- Command delay ramps from 0 to the measured live 1.05 s over the first 20% of RL, then is randomized over 1–25 ticks with jitter, mostly near 21 ticks.
- 0–150 ms stale observations and occasional dropped decisions.
- ±3–5% noise on speed, hit speed, HP and damage, plus placement jitter.
- `make_env(client="15.535.29")` everywhere.

**Reward.** Terminal +1/−1/0, plus potential shaping 0.05 × (γΦ′ − Φ) where Φ is the crown difference plus tower-HP-loss fractions (king ×3), with Φ = 0 at terminal. Add a small elixir-overflow penalty (0.001 per wasted elixir, capped at 0.02 per game), and drop it for the last 20% of training. No card-specific rewards.

**PPO defaults** (FirstLight's values, which astra also agreed with): AdamW, γ = 0.9997 and λ = 0.98 at 250 ms (γ = 0.9994 and λ = 0.96 at 500 ms), clip 0.1, 2 epochs, target KL 0.01, grad norm 0.5, value coef 0.5. Rollouts are 40-second segments with LSTM state carried across them. BPTT chunks are 64 decisions plus 16 burn-in, with about 2,048 decisions per minibatch. The KL-to-BC penalty starts at a budget of 0.03 and relaxes towards 0.1 only while external evaluation keeps improving.

**League** (A, B and D share one opponent pool on disk):
- 40% current self-play.
- 40% frozen pool: BC clones on top meta decks, past selves, and the other runs' promoted checkpoints. Sampling is PFSP, weighted towards opponents we lose to.
- 20% hard matchups: decks we lose to most, plus FL proxies (below).
- Opponent decks: 30% focus / 40% weighted by usage / 30% hard, the mix FirstLight used.
- **Anti-passivity check on every eval:** log first-play time, plays per game, and the share of wins where the opponent made ≤3 plays.

**FL proxies.** Record FL Hog2 and FL General games through `local/firstlight_bridge.py`, then clone them into our network as frozen opponents. They are fast stand-ins for the slow real thing (70 s per game on the laptop GPU and about 10 minutes on CPU, from `v2/eval/results.jsonl`). This trains *against* FirstLight; it does not fine-tune it.

**Precision.** bf16 on the Ada GPUs. fp16 with GradScaler on the V100s (no bf16). Losses, softmax and optimizer state stay in fp32.

## 4. The four models

### A. Atlas: general entity + LSTM policy (~7M)

**Bet:** a FirstLight-shaped network trained on 6–7× more human data than r12 generalizes across decks and beats FL General on mixed decks, not just on our one deck.

**Architecture** (~7M; FirstLight's V4 is ~12M with extra event and projectile encoders we can't feed live):

| Block | Spec | ~Params |
|---|---|---|
| Entity tokens | MLP(36 features) + unit-type, card and kind embeddings (vocab from `card_ids`, never hard-coded); cap 128 entities with an overflow flag | 0.15M |
| Card tokens | 8 own and 8 seen-enemy rows: embedding + MLP(card features) + slot embedding | 0.05M |
| Transformer | 4 layers, d=256, 8 heads, FFN 1024, pre-norm | 3.2M |
| Spatial | Scatter 32-channel entity projections + 14 explicit channels onto 18×32, then 3 residual blocks at 64 ch, pooled to 128 | 0.25M |
| Fusion + core | [global token, mean entity, spatial] → 512 → LSTM 512 | 2.6M |
| Policy head | Gate (wait/play) → slot pointer over hand tokens → tile logits from a FiLM query on per-tile spatial features | 0.5M |
| Value head | 512 → 256 → 1, plus a privileged branch (see §3) | 0.2M |

**Pipeline**
1. **Data.** Extract every replayable IL_Replay match (~220k; 33k were done before) in the v2 entity-shard format at 250 ms, keeping both seats together. Run this on all 48 cores of both servers before RL starts. *Estimate:* about 300M decision rows.
2. **BC**, on 4× Ada with DDP first, sharing the box with B's BC stage:
   - AdamW, lr 2e-4 → 2e-5 cosine, wd 0.01, dropout 0.1.
   - 64 sequences × 80 steps per GPU, with 16 burn-in steps.
   - Gate BCE with class weight 2–4 (not 8), plus slot CE and tile CE.
   - Up to 3 passes, stopping on validation NLL held out by match.
   - Exit when held-out NLL beats bcD1's and the model beats bcD1 in a 200-game sim match.
3. **PPO** with the shared league, lr 1e-5, entropy 0.003 → 0.001.
4. **Decoding.** Pick argmax, or card/tile at T=0.5, on held-out games, then freeze it.

**Budget:** about 64M decisions of RL after BC. **Compute:** Ada 0–1 (learner on one GPU, batched actor inference on the other) and 12 simulator cores plus 2 control cores.

**Hedges against:** r12's strength depending on one deck and a CNN that may not generalize.

### B. Hog: Hog 2.6 specialist (~7M, same network as A)

**Bet:** benchmark H1 is a Hog mirror. Only a strong Hog pilot can win it, and FirstLight showed the path: a general model, then Hog specialization, then outcome-only reward.

**Deck:** Hog Rider, Musketeer (Hero), Cannon (Evo), Skeletons (Evo), Ice Golem, Ice Spirit, Fireball, The Log. In RoyaleGym that is `evolved=["Cannon","Skeletons"]` and `heroes=["Musketeer"]`. RoyaleSim's accuracy for evo and hero forms is unmeasured, so confirm in the FirstLight engine early.

**Pipeline**
1. Initialize from A's BC checkpoint (our own weights).
2. Hog BC: all Hog 2.6 matches, both seats, plus 30% general decisions to avoid forgetting. lr 1e-4 → 1e-5, up to 5 passes with early stopping.
3. PPO with a short phase of BC auxiliary loss (coef 0.2, as FirstLight's `expert_bc_coefficient`), then drop it.
4. Opponents:
   - 30% self-play mirror.
   - 30% A's checkpoints on pool decks.
   - 40% hard: the FL-Hog proxy in the mirror, the FL-General proxy on pool decks, and the decks we lose to.
5. Run the last half entirely under live latency.
6. No Hog-specific shaping at any point.

**Budget:** about 48M decisions. **Compute:** Ada 2–3, 12 simulator cores plus 2 control cores. B's RL starts once A's BC is done.

**Hedges against:** A being strong on average but never mastering the one matchup the benchmark cares most about.

### C. Anchor: r12 lineage, counter-Hog (~1.5M)

**Bet:** the cheapest path to a model that beats both benchmarks starts from the one that already beats General.

**Architecture:** RoyaleLearn's ResNet trunk and pointer head as in r12, 1.5M params, 500 ms decisions. *I haven't checked its exact layer list; it's the same network as `bots/r12`.*

**Pipeline**
1. **Transfer-safe self-distill.** r12 was trained with RoyaleGym's default spatial obs, which includes spell and projectile channels that are zero live. Train a same-size student without those channels on r12's own games, using KL to r12's policy plus human labels at 50/50. Exit when the student beats r12 ≥45% in sim.
2. **Counter-Hog PPO** from the student, continuing your 14 Oct plan:
   - 60–70% of opponents are Hog: FL-Hog proxy, Hog clones, and B's checkpoints once they exist.
   - The rest: past selves and meta-deck clones.
   - 2–5% of games against the real FL Hog2 through the bridge, while throughput allows.
3. Lower lr (2e-5 → 5e-6) and a KL-to-student budget of 0.03 to protect what r12 already does well.
4. Keep r12's own deck. Optionally run a second seed with the student on Hog 2.6.

**Budget:** about 30M decisions. **Compute:** V100 #0 and 5 cores. This can start on day one, because it needs no new data.

**Hedges against:** A and B both being late or weaker than expected. It also isolates the effect of the transfer-safe obs and latency training on a known model.

### D. Oracle: search-distilled policy (~3M)

**Bet:** RoyaleSim is deterministic and forkable (`RustEngine.save_state()` / `load_state()` in `royalegym/rust_engine.py`), so at training time we can look ahead and teach the network moves PPO's noisy gradient rarely finds. The deployed network never searches, so live play needs nothing extra.

**Architecture:** spatial CNN with 4 residual blocks at 96 ch on 18×32, plus scalar and card fusion to 384 → LSTM 384, with the same gate/slot/tile head as A and a value head. About 3M params, small enough to evaluate thousands of rollouts.

**Pipeline**
1. **BC** on the shared shards (a subset is fine), 1–2 passes.
2. **Expert iteration:**
   - At a sampled 10–20% of decisions where a play is legal, take the policy's top 8 (slot, tile) candidates plus wait.
   - Determinize the hidden enemy hand from the counted cycle (2–4 samples).
   - Roll each candidate forward about 4 s with the current policy for both sides.
   - Score each line with the value net plus the tower-HP delta.
   - The search's softmax becomes a distillation target, giving a cross-entropy term in addition to PPO.
3. **Critic:** asymmetric and privileged (§3).

Opponents come from the shared league, with decks rotated over the top 24 meta decks.

**Budget:** about 16M decisions, since search makes each one more expensive. *Estimate:* RoyaleSim runs about 25k ticks/s per core, so one 4-second rollout (80 ticks) costs a few ms. Obs building in Python, not the engine, is the bottleneck: RoyaleGym's docs measure about 520–830 env steps/s per process. **Compute:** V100 #1–2 and 9 cores.

**Hedges against:** PPO plateauing. This is also where a jump past FirstLight could come from, since FirstLight could not fork its real-engine emulator.

**Highest risk.** If the search gives no clear lift by 4M decisions (searched move vs raw policy, 200 games), stop D and give its cores to A and B.

## 5. Compute layout

| Machine | GPU | Cores | Job |
|---|---|---|---|
| Server B (4× RTX 6000 Ada, 32 cores) | Ada 0–1 | 14 | **A** Atlas |
| | Ada 2–3 | 14 | **B** Hog (BC on all 4 GPUs first, shared with A) |
| | — | 4 | OS, league store, checkpoint evals |
| Server A (3× V100, 16 cores) | V100 0 | 5 | **C** Anchor |
| | V100 1–2 | 9 | **D** Oracle |
| | — | 2 | OS, extraction tail, proxy recording |
| Laptop (RTX 4070, 16 cores) | 4070 | 16 | Queued FL-bridge evals (FL Hog2 / General on GPU), FirstLight emulator benchmark, live Nulls tests |

**Order of work**
1. **Day 0–1:** extraction on all spare cores. C's self-distill runs meanwhile.
2. **Day 1–2:** A and B BC on 4× Ada. D's BC runs on the V100s.
3. **Day 2 onward:** all four in RL.

*Estimate:* r12 did 16M steps in roughly 20 h on 4 cores. With batched GPU inference and 12 cores per run, 300–600 decisions/s per run is plausible, which means about 1–2M decisions per hour. Measure this in the first hour of each run before trusting the budgets above.

## 6. Evaluation gates

1. **Every ~2M decisions:** 100 seat-balanced games in sim against FL Hog2 and FL General via the bridge, on the laptop queue. Log the anti-passivity metrics.
2. **Promotion:** 400 fresh games per opponent. Targets are General ≥ 60% for A and C, and Hog2 ≥ 40% for B and C, then ≥ 50% with the lower 95% bound above 50%. Promoted checkpoints join every run's hard pool.
3. **Perturbed sim:** the same games at a fixed 1.05 s delay. A large drop means the policy is brittle.
4. **Official:** FirstLight's real-engine emulator (H1 and G suites from the plan: 760 and 382 games, argmax). The bridge numbers are only a screen, because FL sees less in RoyaleGym than in its own engine.
5. **Live Nulls:** 100+ games at a fixed trophy band.

## 7. Where I departed from astra's proposal

Astra suggested: (1) an r12 student, (2) a 6.1M entity-LSTM generalist, (3) a 2.45M CNN-LSTM latency-trained Hog specialist, and (4) a 1.8M recurrent Double-DQN. Its codex sandbox couldn't read the local files, so it worked from my summary.

- **Kept:** the r12 anchor (C), the entity-LSTM generalist (A), the shared PPO, reward and latency rules, the gates, and its read of the v2 0/40 (likely bugs or undertraining, not proof against entities).
- **Hog specialist:** same network as A, warm-started from A's BC, rather than a separate CNN. FirstLight's specialist only worked once it started from the General model; our Hog run from a weak clone failed for the same reason. Latency training now applies to all four models.
- **DQN replaced by search distillation:** a 2305-way Q-function over 3-minute games with a nonstationary league is the riskiest possible bet. Lookahead uses the simulator's one real advantage over FirstLight, forkability. If you'd rather test an off-policy learner, astra's DQN spec is the one to use for slot D.
- **Critic:** astra wanted the critic restricted to transfer-safe fields too. I allow privileged fields in a value-only branch, since the critic is never deployed.

## 8. First concrete steps

1. Audit v2 BC against bcD1 in sim, and fix it before A and B rely on that code.
2. Write the transfer-safe obs builder and the three-adapter consistency test.
3. Start full IL_Replay extraction on both servers.
4. Start C's self-distill on V100 #0.
5. Record FL Hog2 and FL General games on the laptop for the proxies.

Sources: [FirstLight_CR](https://github.com/Jaasssoooonnnnn/FirstLight_CR) ([training history](https://github.com/Jaasssoooonnnnn/FirstLight_CR/blob/main/docs/training_history.md)), [KataCR](https://github.com/wty-yy/katacr) and [its paper](https://arxiv.org/abs/2504.04783), [Learning to Play Imperfect-Information Games by Imitating an Oracle Planner](https://arxiv.org/abs/2012.12186), [Real-Time-Strategy-RL-Clash-Royale](https://github.com/Jaso1024/Real-Time-Strategy-RL-Clash-Royale), [RoyaleGym](https://github.com/RoyaleGym/RoyaleGym).
