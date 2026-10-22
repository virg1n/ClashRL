I recommend one conservative continuation, one generalist, one latency-trained Hog specialist, and one off-policy experiment. All weights originate from your work or fresh initialization.

I could not inspect local files: the command runner failed before reading them. The recommendations below use your supplied results and public FirstLight documentation. Parameter counts are estimates for the proposed architectures, not audited checkpoint counts. No files were changed.

Shared contract and training rules

Use client="15.535.29" throughout extraction, training and evaluation. Audit card/form support before including a replay or deck.

Observations must pass through the same adapter used live: visible troop/building identity, owner, position and HP; tower state; own hand/elixir; clock; publicly revealed cards; previous observations/actions; pending commands and elapsed times. Include field-availability masks. Opponent elixir/cycle estimates must derive solely from public history. Exclude simulator-only cooldowns, targets, hidden cards, spell/projectile lists and unobserved evolution state. Apply this restriction to critics too.

Actions remain exactly 2305: wait or slot/tile. Mask using bridge-observable information. No abilities, extra delay actions or multi-action decoder. Model the bridge’s actual queue, card identity at submission, acknowledgements and stale-command rejection. A 250 ms policy must not send duplicate commands every tick.

For all runs, ramp command latency from zero to measured live latency over the first 20% of RL; finish predominantly at 1.05 seconds, with modest measured jitter. Validate at fixed 1.05 seconds separately. Keep human BC labels at their original timing; blindly shifting them backward would introduce impossible anticipation.

Split demonstrations by match, keeping both seats together; preferably hold out players/time periods too. Expand toward all approximately 220k replayable matches, prioritizing deck coverage and replay fidelity. Train contiguous sequences, preserve wait frequency or importance-correct subsampling, and report gate/card/tile losses separately.

Shared PPO defaults: AdamW, gamma=0.9997 and lambda=0.98 at 250 ms; at 500 ms use 0.99940009 and 0.9604. Clip=0.1, two epochs, target KL=0.01, gradient norm=0.5, value coefficient=0.5. Use 40-second rollouts, recurrent chunks of 64 decisions plus 16 burn-in, approximately 2048 valid decisions/minibatch. Keep recurrent state across rollout boundaries.

Reward: terminal +1/−1/0; potential shaping 0.05×[gamma×Phi(next)−Phi(current)], where Phi is normalized own-minus-enemy tower HP and terminal potential is zero. Optional overflow penalty: 0.001 per wasted elixir, capped at 0.02/match; remove it for the final 20%. No Hog-deployment or opening-time rewards. FirstLight’s final specialist improved after removing Hog-specific shaping. [Training history](https://github.com/Jaasssoooonnnnn/FirstLight_CR/blob/main/docs/training_history.md)

1. Anchor-1.3: retain the demonstrated strength cheaply.

Architecture: approximately 1.32M parameters. Raster input standardized to 32 channels; 3×3 convolution to 64 channels; four residual blocks, two 3×3 convolutions each. Pool to 4×3, project 768→512. Scalars 64→128; shared 256-entry card embedding of width 32. Fuse spatial/scalar/four-hand embeddings, 768→512. Five-way wait/slot head; selected-card-conditioned 128-dimensional query over spatial keys; value MLP 512→256→1.

This is a proposed student of your r12, avoiding an invented claim about its exact layers. Distill transfer-safe r12 outputs and human actions, initially 50:50, on its successful Goblin Giant deck. Use 500 ms decisions.

BC: lr=1e-4, batch=4096, up to three passes; retain the best held-out checkpoint. RL: lr=2e-5→5e-6, entropy coefficient=0.005→0.001. Adaptive clone-KL budget 0.03 initially, relaxing to 0.1 only when external evaluation improves.

Opponents: 40% current self-play, 40% active frozen clones/past selves, 20% hard frozen league opponents. Keep the learner deck fixed.

Budget: 32M new decisions. Server A: V100 #0, six cores including one collector/driver core; start with 64 concurrent matches.

Hedge: larger architectures and new algorithms may fail; this preserves your strongest demonstrated starting point.

2. Atlas-6.1: learn interactions and deck generalization from substantially more data.

Architecture: approximately 6.1M parameters. Four entity-transformer layers, d=256, eight heads, FFN=1024; spatial CNN with three 64-channel residual blocks; 128-dimensional spatial/scalar summaries; fusion to 512; LSTM 512→512. Same conditional slot/tile pointer head and 512→256→1 critic. Cap entities at 128 with explicit overflow indicators and deterministic, tactically sensible retention.

Use safe entity observations plus the shared raster/history contract, at 250 ms.

BC from scratch: AdamW lr=2e-4→2e-5, weight decay=0.01, dropout=0.1, 4096 valid tokens/update, up to three full passes with validation-based stopping. Sample decks with capped inverse-frequency weighting. Do not start RL merely because a scheduled epoch finished.

RL: shared PPO, lr=1e-5, entropy=0.003→0.001, clone-KL budget 0.03→0.1. Opponents: 40% self-play, 40% frozen league, 20% hard matchups. Cover supported deck families, with held-out deck combinations.

Budget: 64M decisions. Server B: Ada #0 learner, #1 batched inference; 14 cores, twelve for simulation; start with 256 matches.

Hedge: r12’s success may depend on one deck and a representation that generalizes poorly.

3. Sentinel-Hog-2.45: specialize for delayed defensive execution.

Architecture: approximately 2.45M parameters. Four 64-channel spatial residual blocks; 4×3 pooling projected to 256; scalar/card fusion to 256; LSTM input 256, hidden 512; conditional spatial pointer and value head as above. No entity transformer. Use explicit observation-history differences and pending-command features.

Train a fixed, supported Hog 2.6 deck with abilities disabled. Do not silently equate classic cards with FirstLight’s evolution/hero deck.

BC: initialize compatible visual features from your own work if useful; otherwise fresh. Use both seats of Hog demonstrations, supplemented by 30% general defensive situations. lr=1e-4→1e-5, 4096 tokens/update, up to five passes with early stopping.

RL at 250 ms: lr=1e-5; entropy=0.002; clone-KL budget 0.05, annealed away after 10M decisions if stable. Opponents: 30% self-play, 30% active generalists, 40% hard frozen matchups. Sample opponent decks 30% focus/40% broad/30% hard. Finish the last half entirely under measured live latency/jitter.

Budget: 48M decisions. Server B: Ada #2 learner, #3 inference; 14 cores, twelve simulators; start with 128 matches.

Hedge: the binding constraint may be timing and partial observability, not representational capacity.

4. Replay-Q-1.8: reuse scarce simulator experience.

Architecture: approximately 1.8M online parameters, plus a separate target copy. Three 64-channel spatial residual blocks; pooled projection to 256; scalar/card fusion to 256; LSTM 256→256; shared 256-unit layer; dueling scalar-value and 2305-action advantage outputs. Legal-action masking applies to selection and bootstrap targets.

BC: three passes, lr=1e-4, using a temporary policy head. Retain the encoder/LSTM, initialize Q heads separately, and seed replay with 250k decisions from your own competent policy.

Then recurrent Double DQN at 500 ms: lr=1e-4, gamma=0.99940009, 10-step returns, Huber loss, gradient clip=10, target synchronization every 2000 updates. Replay capacity=2M decisions; prioritized replay alpha=0.6, importance beta=0.4→1.0. Batch 32 sequences with 32 burn-in/64 learning steps; one update per 256 new decisions. Actor epsilon spans 0.02–0.2.

Use a fixed supported deck. Opponents are frozen league snapshots, refreshed only every 2M decisions; bound replay age across league changes.

Budget: 16M new decisions. Server A: V100 #1 and six cores; start with 64 matches.

Hedge: PPO wastes data when CPUs dominate. Highest-risk branch: long-horizon Q learning and opponent nonstationarity may outweigh replay’s benefit.

Compute, gates and interpretation

A’s third V100 handles shared frozen-opponent inference; four remaining cores serve inference, extraction and OS. B retains four shared service cores. Avoid concurrent heavy extraction stealing simulator cores. Use fp16 with scaling on V100, bf16 on Ada; FP32 losses, probabilities and optimizer state. Laptop handles queued FL evaluation.

At an illustrative measured 500 learner-decisions/second/run, RL takes approximately 18/36/27/9 hours respectively. These are conditional calculations, not throughput forecasts. Budget BC learner time at 12/36/24/12 hours, with extraction separate; benchmark actual throughput before committing. Compare simulated seconds and completed games too—250 ms doubles decision counts.

Every 2M decisions, screen with 100 seat-balanced games per FL opponent. Promotion requires 400 fresh games/opponent: target General≥60%, Hog2≥40%; claim superiority only when a 95% interval clears 50%. Report each deck/latency condition separately. Freeze decoding after validation over argmax and card/tile temperatures 0.5/1.0.

Final acceptance requires the real-engine emulator, with FL’s full observations and supported capabilities, followed by live-delay testing. Degraded FL bridge results are screening evidence, not a calibrated bound on transferred win rate.

Your v2’s 0/40 is compatible with undertrained BC, wait-dominated loss, sequence/state bugs, coordinate or mask mismatches, replay divergence and compounding imitation errors. It does not establish that entities failed. Audit those before scaling. Likewise, equal BC loss across capacities suggests the current supervision/objective is limiting; it does not prove that model capacity never matters.