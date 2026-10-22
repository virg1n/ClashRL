# gpt-6-astra review of the A/B training code (2026-10-21, codex-cli 0.162.1, `-m gpt-6-astra`)

Five rounds. Each round reviewed the current code; its findings were fixed and the whole pipeline was rerun end to end on Linux (WSL, CPU, tiny config) through run_ab.sh, including a stop and resume in the middle of PPO. Round 5 signed off ("SIGN-OFF: yes"); its one medium leftover was fixed afterwards.

## Round 1 (raw)

1. **High — `abtrain/ppo.py:455–458, 429–440`: interruption during checkpoint rotation can restart PPO from BC.**  
   `checkpoint()` first does `os.replace(latest, rd / "prev.pt")`, then writes the replacement. If the session dies between those operations, `latest.pt` is absent. Resume checks only `if latest.exists()`; the `prev.pt` fallback applies only when an existing latest is unreadable. It therefore takes the `else` branch and loads `init`, discarding potentially days of PPO progress.  
   **Fix:** load `prev.pt` when latest is missing as well as corrupt. Write and fsync the new checkpoint before rotating the existing one.

2. **High — `abtrain/pipeline.py:256–264, 152–162`: an unexpected lane exception can leave the supervisor running forever.**  
   The lane wrapper catches an exception, logs it, and returns without marking the lane failed or calling `abort()`. For example, an exception in the main lane before `bc1` completes leaves the FirstLight/evaluation lanes inside `wait_for("bc1")`. That loop checks neither the deadline nor whether the producing lane has died. The supervisor keeps heartbeating, and the shell never retries because the supervisor never exits.  
   **Fix:** propagate lane failures into supervisor state, unblock dependent waiters, and exit nonzero when required work fails. Make dependency waits deadline-aware.

3. **High — `abtrain/pipeline.py:477–485`; `abtrain/ppo.py:733–742`: the advertised graceful stop kills the checkpointing processes directly.**  
   `_kill()` sends `SIGTERM` to the entire process group. Only the PPO coordinator installs `_term`; its spawned actor and learner retain the default SIGTERM behavior. Thus stopping through the supervisor kills the learner before it can observe `stop` and execute `checkpoint()` at line 657. The same applies when the supervisor enforces its deadline.  
   **Fix:** first signal the coordinator alone so it sets the shared event and waits for the learner to checkpoint. Kill the process group only after the graceful timeout.

4. **High — `abtrain/ppo.py:445, 515, 548`: the KL budget overwrites the encoder token budget.**  
   Initially `budget = cfg["bc_token_budget"]` is approximately 200,000. Inside every iteration it becomes `budget = st.setdefault("kl_budget", ...)`, approximately `0.03`. Consequently `micro_splits(b, budget)` uses an effective budget of one token and splits every minibatch into individual sequences. The default 32-sequence minibatch therefore runs 32 separate forward/backward passes; B’s auxiliary BC is similarly affected. This severely reduces learner throughput throughout the run.  
   **Fix:** use separate `token_budget` and `kl_budget` variables and pass only the former to `micro_splits()`.

5. **High — `abtrain/vec.py:147–156`; `abtrain/shards.py:41–44`: PPO acts on observations that training silently changes.**  
   `batch_obs()` includes every entity, but `pack_rows()` stores only `[:MAX_ENT]`, where `MAX_ENT = 128`. Whenever troops, spells, and projectiles exceed that limit, the stored old log-probability was computed from a different observation than the learner receives. Even unchanged weights cannot reproduce that probability; the LSTM state evolution also differs. PPO then treats observation truncation as a policy update.  
   **Fix:** apply one identical entity selection/capping rule before both acting and storage, including evaluation adapters.

6. **High — `abtrain/bc.py:50–59`: truncated replay prefixes are treated as terminal outcomes when computing advantages.**  
   `advantages()` appends `np.full(n, out[s])` to every sequence, ignoring `seq_meta[:, 3]`, the truncation flag. Extraction explicitly retains prefixes ending at the first rejected deployment. For such a prefix, its last 24 decisions receive advantages toward the eventual recorded win/loss as though that outcome occurred within six seconds—even if the replay stopped minutes early. This systematically weights actions around simulator failures using fabricated terminal transitions.  
   **Fix:** use the outcome extension only for genuine terminal sequences. Exclude truncated tail decisions lacking the required look-ahead, or give them an explicit invalid-advantage mask.

7. **High — `abtrain/replay.py:211–217`; `abtrain/shards.py:155–160`; `abtrain/model.py:268–270`: extraction retains impossible card actions as positive BC labels.**  
   `nearest_legal_tile()` can return `-1`, but extraction still stores `[slot, -1, delay]` and executes the original command. For example, a card unaffordable at the window start can become affordable before its send tick: its start-of-window tile mask is empty, yet the deployment succeeds. Collation then forcibly marks its candidate legal, and `evaluate()` trains gate/card/delay while omitting tile loss. At inference, `act()` explicitly falls back to waiting when that card has no legal tile. Training is rewarding an action inference cannot produce.  
   **Fix:** make label construction obey the same action support as inference. Either implement delay-aware legality consistently, or exclude unrepresentable action windows from policy loss rather than inventing candidate legality.

8. **High — `abtrain/game.py:283–288, 304–308`: simulator-only timing information enters the policy.**  
   The policy entity features directly include `e.deploy_ticks`, `e.attack_phase`, and `e.charge`, with no live-validation/unknown handling. `AtlasNet._encode()` consumes the entire `ent_f` tensor. This contradicts the transfer-safe input requirement that per-unit timers and charge information remain excluded or unknown unless validated by the live bridge. BC and PPO can exploit information unavailable during deployment.  
   **Fix:** remove or consistently mask unvalidated fields in the shared policy adapter. Keep privileged timing features exclusively in the value branch, and verify adapter equivalence before extracting the full dataset.

9. **High — `abtrain/flmatch.py:451–456, 473, 497`; `abtrain/pipeline.py:388–396`: failed recording can permanently eliminate the required stand-ins.**  
   `_record_chunk()` counts game exceptions and continues; even a chunk with zero successful games returns normally. `record()` writes its done marker, so reruns never retry those games. The supervisor’s completion test is merely `(FL_SHARDS / "done").exists()`, a directory created before recording starts. It can therefore mark recording successful with no usable shards, after which the stand-in lane skips both models and PPO continues without them.  
   **Fix:** distinguish successful chunks from failed/incomplete chunks, retry missing successful games, and validate usable data for both tags before marking recording complete. Enforce the required stand-in installation rather than silently accepting its absence.

10. **Medium — `abtrain/bc.py:539–542`; `abtrain/shards.py:188–191`: microbatch trimming removes valid entities after feature dropout.**  
    `collate()` drops projectile tokens by clearing mask entries, leaving holes. `micro_splits()` then uses `ent_mask.sum(-1).max()` as the slice endpoint. A row with 100 tokens and a dropped token at position 10 has 99 true entries, so slicing to 99 also removes the valid final token. Larger numbers of holes remove more unrelated entities.  
    **Fix:** trim using the highest occupied index, or compact tokens after dropout before using their count as a length.

11. **Medium — `abtrain/model.py:378` and `279`: the second-play mask deliberately allows unaffordable actions.**  
    Both paths use `ccost <= left + 0.5`. With 7.6 elixir and two legal four-elixir cards, the model can select both with delay zero: after the first play only 3.6 remains, but the second passes the mask and is refused by the simulator. The half-elixir tolerance is much larger than float16 rounding error and does not depend on the selected delays.  
    **Fix:** use available elixir with only a small numerical tolerance, or explicitly account for regeneration at the selected second delay. Acting and evaluation must use the same rule.

12. **Medium — `abtrain/evaluator.py:101–121`: a promoted checkpoint can carry win-rate and goal claims for a different decoder.**  
    `pooled`, its Wilson interval, and `goal_reached` come entirely from `t05` games. The code can subsequently choose `decode = "argmax"` from a separate, smaller evaluation, then save that decoder with the `t05` statistics. The deployed artifact’s reported performance was therefore never measured under its selected decoding.  
    **Fix:** choose the decoder first and confirm it on an independent promotion suite, or retain separate statistics and attach only the selected decoder’s results to the checkpoint.

13. **Medium — `abtrain/pipeline.py:308–315, 338–344`: failed stage-2 acceptance tests default to accepting stage 2.**  
    Both A/B tests are noncritical and their return values are ignored. If A’s test fails and produces no result, `if ab and ...` leaves stage 2 selected. B explicitly accepts stage 2 when `not ab`. Thus an infrastructure failure bypasses the requirement that stage 2 demonstrate an improvement before PPO uses it.  
    **Fix:** require a successful, complete test result meeting the acceptance threshold; otherwise use the established fallback checkpoint and record the failed acceptance test.

## Round 2: verification of the fixes (raw)

1. **High — Previous #1: fixed — `abtrain/ppo.py:430–436, 458–464`.**  
   Resume now tries `latest.pt`, `latest.new.pt`, and `prev.pt`. Checkpointing writes and fsyncs the new file before rotating the old one. Interruption between either rename no longer sends PPO back to BC.

2. **High — Previous #2: partly fixed — `abtrain/pipeline.py:152–162, 229–248, 256–265, 463–492`.**  
   Dependency waits now recognize failure and deadlines, and main-lane exceptions call `abort()`. However, `abort()` only sets `self.state["aborted"]`; running stage watchers never inspect it. If the main lane fails after the evaluator starts but before PPO starts, the evaluator can run until the deadline waiting for nonexistent PPO completions. The supervisor waits for that lane, so the shell cannot retry. Exceptions in `lane_A`, `lane_B`, `fl`, or `eval` are still merely logged because of `if name == "main"`.  
   **Fix:** propagate every required lane failure into supervisor failure state, stop active dependent stages, and exit nonzero promptly. Separate retryable failure cancellation from a user-requested stop.

3. **High — Previous #3: partly fixed — `abtrain/pipeline.py:504–519`; `abtrain/ppo.py:493, 528–620, 662–664, 784–787`.**  
   Signaling the coordinator first fixes the immediate group-kill problem. But the learner checks `stop` only between entire rollout iterations; it does not check it inside the two epochs of minibatches. The coordinator still executes `p.join(timeout=120)` followed by `p.kill()`. A stop arriving with more than 120 seconds of learner work remaining kills it before its final checkpoint. A full default iteration processes roughly 130,000 training rows, so this is a material shutdown risk.  
   **Fix:** check cancellation between optimizer steps, leave the training loops, and checkpoint immediately. Have the coordinator wait for a checkpoint-complete acknowledgment before escalating.

4. **High — Previous #4: fixed — `abtrain/ppo.py:448, 521, 554, 589`.**  
   `token_budget` is now separate from the KL `budget`; both PPO and auxiliary BC pass the token budget to `micro_splits()`.

5. **High — Previous #5: fixed — `abtrain/game.py:425–428`.**  
   The common `SeatView.build()` caps entities before observations reach acting, storage, or evaluation. `pack_rows()` therefore no longer removes entities that the actor saw.

6. **High — Previous #6: fixed — `abtrain/bc.py:54–65, 428–437`.**  
   Truncated sequences now leave their unavailable look-ahead tail as NaN. Those rows receive neutral weights instead of fabricated terminal advantages.

7. **High — Previous #7: partly fixed — `abtrain/replay.py:217–246`; `abtrain/flmatch.py:270–287`; `abtrain/model.py:280–282`; `abtrain/shards.py:161–166`.**  
   Missing-tile labels now become `SKIP`, fixing the reported first-card case. But second plays are still accepted without checking the actor’s shadow affordability mask. For example, at triple elixir, a window starting around 7.85 elixir can successfully deploy two four-elixir cards by its last tick. Both cards have legal initial tile masks, so extraction retains the label; inference excludes the second card because `left + SECOND_PLAY_SLACK < 4`. Training then invents legality with:
   ```python
   m2 = m2 | (has_c2[:, None] & (idx == c2.clamp_min(0)[:, None]))
   ```
   FL ability labels also bypass initial candidate-legality validation and are forcibly enabled by collation.  
   **Fix:** validate the complete label against the exact inference support, including abilities, second-card affordability, and ordering. Skip unsupported policy targets; remove label-based mask expansion.

8. **High — Previous #8: partly fixed — `abtrain/game.py:284–290, 308–309`.**  
   Attack phase, charge progress, and the windup status feature are masked. However, `r[9] = e.deploy_ticks * 0.05` still exposes the exact per-unit deployment timer directly to the policy. No validation/provenance condition was added.  
   **Fix:** mask the unvalidated deployment countdown consistently in every adapter. Retain any validated visible deployment flag separately.

9. **High — Previous #9: partly fixed — `abtrain/flmatch.py:483–489, 510–525`; `abtrain/pipeline.py:385–398, 410–443`.**  
   Zero-success chunks no longer get completion markers, and both recording tags must have files. However:
   - A chunk with three successful games out of six is marked complete, permanently abandoning its other games.
   - `standins_round()` ignores recording failure and can train on partial data. Once both stand-ins exist, the outer loop exits even if recording remains incomplete.
   - Exhausted checkout failures still return permanently without stand-ins, while PPO continues.
   - A failed stand-in check still reaches `league_copy(...)`.

   **Fix:** track the required successful recordings explicitly, require successful validation before installation, and propagate inability to provide the required stand-ins as a pipeline failure rather than successful completion.

10. **Medium — Previous #10: fixed — `abtrain/bc.py:548–552`.**  
    Trimming now uses the highest occupied slot, so holes from projectile dropout no longer remove unrelated valid entities.

11. **Medium — Previous #11: not fixed — `abtrain/model.py:26, 280, 379`.**  
    Changing the allowance from `0.5` to `0.1` reduces the error but preserves it:
    ```python
    SECOND_PLAY_SLACK = 0.1
    ```
    With 7.95 elixir and two four-elixir cards, both can still be selected with delay zero. Regeneration later in the window cannot fund simultaneous commands. Float16 rounding does not justify a 0.1-elixir allowance.  
    **Fix:** use only a numerical rounding tolerance, or implement delay-dependent affordability consistently in acting, evaluation, and label validation.

12. **Medium — Previous #12: fixed — `abtrain/evaluator.py:112–129`.**  
    When argmax wins selection, its own wins, game count, rate, and Wilson interval replace the `t05` statistics before the artifact and goal claim are saved.

13. **Medium — Previous #13: fixed for the reported failure path — `abtrain/pipeline.py:314–319, 344–349, 541–543`.**  
    Missing A/B results now reject stage 2 for both agents. The helper’s “complete” check is still weaker than advertised—`games >= abtest_games // 2`—but the supplied `simeval` currently writes its result only after completing its full plan.

Additional critical/high issues:

14. **High — Remaining missed issue: subprocess descendants survive failed-stage retries — `abtrain/pipeline.py:203–205, 229–234, 611–626`.**  
    When a stage leader exits, `_watch()` immediately returns its code. `run()` then removes its process-group record without cleaning up descendants. Startup cleanup likewise requires `_alive(pid)` for the group leader. If a torchrun leader dies while ranks or data workers survive, those processes retain GPUs/RAM and can keep writing checkpoints while the supervisor starts another attempt in the same output directory. A dead group leader does not imply a dead process group.  
    **Fix:** clean up the recorded process group on every failed attempt, including when its leader has exited. Preserve sufficient process identity to reap surviving descendants safely on restart.

15. **High — Remaining missed issue: no atomic single-supervisor lock — `abtrain/pipeline.py:758–765`; `run_ab.sh:55–74`.**  
    Both launch guards perform separate “read/check” and “write/start” operations. Two near-simultaneous launches can both pass before either writes its heartbeat. Both supervisors then mutate the same state and checkpoint paths; `reap_stale()` can also kill stages belonging to the other live supervisor. This is especially damaging when rerunning the launch command during recovery.  
    **Fix:** acquire an exclusive process lock before modifying state or reaping stages, and hold it for the supervisor’s lifetime. Use the heartbeat only for status, not mutual exclusion.

16. **High — Remaining missed throughput issue: the opponent cache can thrash on every decision — `abtrain/ppo.py:74–95, 255–275`; `abtrain/league.py:120–129`.**  
    The league retains approximately 24 snapshots per agent, plus BC models, stand-ins, and promoted models, but the GPU cache holds only 32. With 228 concurrent battles, more than 32 frozen opponent identities can be active simultaneously. Each decision iterates those identities in a stable order. Once that working set exceeds the cache, ordinary LRU eviction can make the next iteration evict and reload the same models again—repeatedly initializing networks, reading checkpoints over NFS, and transferring weights to GPU.  
    **Fix:** pin models used by active battles and evict only inactive opponents, or constrain the simultaneously active opponent set to the cache capacity.

## After round 2

- 2: any lane crash or a failed required stage now cancels every running stage and exits with code 4; `run_ab.sh` restarts and resumes.
- 3: the learner checks for a stop between minibatches; the coordinator gives it 10 minutes to checkpoint.
- 7, 11: labels are checked against the exact action support at extraction (`game.label_supported`), unsupported steps are left out of the loss; the second-play slack is 0.01 elixir (float16 rounding only).
- 8: the deploy countdown is no longer a policy input (the deploy flag stays).
- 9: a recording chunk counts only when all its games finished (partial accepted after 3 tries with fresh seeds); stand-ins are trained only on complete recordings and installed only after their check ran; the FirstLight checkout is retried every 30 min.
- 13: stage 2 needs a full-length A/B result.
- 14: every process of a stage attempt carries a token in its environment and is killed after the attempt (torchrun ranks run in their own sessions, so process groups are not enough).
- 15: one supervisor per host via an exclusive lock on a local file.
- 16: the PPO opponent cache never evicts a model a running battle uses (capacity 64).

## Round 3: final check (raw)

I would **not sign off yet**. Items 2, 3, 7 and 9 remain incomplete, and there is a new high-severity evaluation issue. This is a static review of the supplied code.

1. **Previous #1 — Fixed.** `abtrain/ppo.py:433–440, 462–468`. Resume tries all three checkpoint files; checkpoint rotation occurs after the new file is fully written.

2. **Previous #2 — Partly fixed — high.** `abtrain/pipeline.py:247–249, 270–275, 375–383, 472–475`. Exceptions now cancel other watchers, but **exhausted required subprocess failures still do not**. PPO’s failure only produces:
   ```python
   if not ok:
       log(...)
   ```
   The evaluator’s `self.run(...)` result is ignored entirely. A failed PPO lane can leave evaluation waiting until the deadline; an exhausted evaluator can leave training running without required evaluation.  
   **Fix:** call `abort()` when either required stage exhausts retries before the deadline. Distinguish deadline completion and requested cancellation from failure.

3. **Previous #3 — Partly fixed — high.** `abtrain/ppo.py:535–537, 630–631, 671–673, 793–796`; `abtrain/pipeline.py:521–536`. Minibatch cancellation is implemented, but the coordinator’s 600-second allowance is overridden by:
   ```python
   def _kill(proc, grace: float = 180.0):
   ```
   After 180 seconds, the supervisor signals the group and subsequently sends SIGKILL. A learner taking longer to finish its current minibatch, auxiliary-data fetch or checkpoint cannot use its advertised grace period.  
   **Fix:** make the supervisor’s timeout exceed the entire coordinator shutdown budget, preferably with an explicit checkpoint-complete acknowledgment.

4. **Previous #4 — Fixed.** `abtrain/ppo.py:452, 525, 561, 596`. Encoder token budget and KL budget remain separate.

5. **Previous #5 — Fixed.** `abtrain/game.py:427–430`. Observations are capped before acting and storage.

6. **Previous #6 — Fixed.** `abtrain/bc.py:54–65, 428–436`. Truncated tails retain NaN advantages and receive neutral weights.

7. **Previous #7 — Partly fixed — high for resumed older shards.** `abtrain/game.py:433–448`; `abtrain/replay.py:245–248`; `abtrain/flmatch.py:286–288`; `abtrain/shards.py:161–171`; `abtrain/model.py:279–281`. Fresh extraction checks candidate legality, affordability and ordering. **Collation only checks initial candidate legality**, despite the author’s claim that it validates the exact support.

   An older label containing two initially legal four-elixir cards at 7.85 elixir survives collation. Evaluation then restores the forbidden second candidate:
   ```python
   m2 = m2 | (has_c2[:, None] & (idx == c2.clamp_min(0)[:, None]))
   ```
   Training therefore still learns an action inference cannot produce.  
   **Fix:** validate full support during collation—including second affordability, distinctness, delays and tile support—and exclude invalid targets. Remove mask expansion.

8. **Previous #8 — Fixed for the reported countdown.** `abtrain/game.py:265, 287–292`. Feature 9 stays zero; the visible deployment flag remains.

9. **Previous #9 — Partly fixed — high.** `abtrain/flmatch.py:484–490, 515–520`; `abtrain/pipeline.py:396–426, 445–460`. Recording and checkout retries improved, and installation requires a check result. However:
   - After three attempts, even **one successful game out of six** earns a completion marker.
   - `record_complete()` checks marker existence, not the required successful-game count.
   - Checkout or stand-in creation can remain unsuccessful until the deadline while PPO proceeds; no required-stage failure is propagated.

   **Fix:** retain successful recordings and retry the missing quota; validate marker contents. Enforce an explicit stand-in readiness requirement, and report failure if it cannot be met instead of finishing successfully without them.

10. **Previous #10 — Fixed.** `abtrain/bc.py:548–552`. Entity trimming uses the highest occupied slot.

11. **Previous #11 — Partly fixed.** `abtrain/game.py:58–60, 446–448`; `abtrain/model.py:378`. The smaller tolerance still permits genuinely unaffordable simultaneous commands. For example, 7.995 elixir rounds to approximately 7.993164; adding `0.01` admits two four-elixir cards at delay zero. Float16’s maximum rounding error for this normalized elixir range is approximately **0.002441 elixir**.  
    **Fix:** retain exact affordability information or use a conservative representation that cannot admit an actual deficit. This remaining discrepancy is small, but the original defect is not fully eliminated.

12. **Previous #12 — Fixed.** `abtrain/evaluator.py:112–127`. The selected decoder’s own counts and confidence interval are saved.

13. **Previous #13 — Fixed.** `abtrain/pipeline.py:558–560`. Acceptance now requires the full configured game count.

14. **Previous #14 — Fixed for the reported surviving-descendant path.** `abtrain/pipeline.py:202–216, 624–669`. Normal attempt cleanup and restart cleanup identify descendants by inherited tokens, independently of leader liveness and process groups.

15. **Previous #15 — Fixed for simultaneous supervisors on the configured host/workspace.** `abtrain/pipeline.py:672–685, 810–816`. The exclusive lock is acquired before constructing the supervisor or reaping stages and remains held.

16. **Previous #16 — Fixed.** `abtrain/ppo.py:83–99, 268–274`. Active opponent models are excluded from eviction. Capacity can temporarily exceed 64, which avoids the reported cache thrashing.

17. **Deviation (a), installing low-scoring stand-ins — Accept.** `abtrain/pipeline.py:451–458`. The stated requirement calls for checking and retaining stand-ins, without specifying a minimum validation score. Logging a low score and installing the model is consistent with that requirement. This acceptance does **not** excuse incomplete recording quotas or failed evaluation games.

18. **Deviation (b), label-based mask expansion — Reject.** `abtrain/model.py:281`; `abtrain/shards.py:161–171`. It is not guaranteed to be a no-op: collation does not validate second-play affordability or ordering. It hides inconsistent targets by changing the training distribution. Invalid labels should be excluded and counted, rather than made legal.

19. **NEW — High — Failed evaluation games are discarded, allowing unsupported promotion and goal claims.** `abtrain/evaluator.py:66–71, 101–127`; `abtrain/flmatch.py:385–402`.

    Evaluation accepts up to 25% failed games:
    ```python
    if res["errors"] > max(1, len(tasks) // 4):
        raise EvalFailed(...)
    ```
    `summarize()` removes those games from the denominator. For example, a screen with 30 wins and 10 errors, followed by promotion with 120 wins and 40 errors, becomes **150/150 wins**, with a lower Wilson bound above 85%. The 50 unresolved games could all have been losses, making the complete planned result **75%**. Failures can also destroy seat balance.

    **Fix:** retry failed scheduled games using their original seeds and seats. Require a complete evaluation before promotion or a goal claim; if retries fail, mark the evaluation incomplete and exclude it from promotion and improvement decisions. Do not silently condition the reported win rate on games that happened to finish.

No additional new critical defects were established from the supplied code.

## After round 3

- 2: a required stage (PPO A, PPO B, the evaluator) that exhausts its retries before the deadline cancels the run (exit 4, `run_ab.sh` retries and resumes). Deadline completion and a requested stop are not failures.
- 3: per-stage stop grace; PPO gets 13 min (actor 2 min + learner checkpoint 10 min) before its group is signalled.
- 7 / deviation (b): the label-based mask expansion in `evaluate` is removed. Collation now checks the whole action support (`shards.invalid_labels`: candidates, distinct second play, delay order, affordability with the first play paid) and excludes invalid steps (`invalid` mask), as extraction does.
- 9: recording counts finished games per chunk from part markers; a retry plays only the missing games with fresh seeds; leftovers of interrupted attempts are removed; `record_complete()` checks the counts. PPO waits for both stand-ins (up to `standin_wait_hours`, 3 h) before starting; if they never arrive, STATUS.md and `state.json` flag the unmet requirement.
- 11: exact elixir (int16 milli-elixir, `elx`) is stored with every step and used for the second-play rule in acting, training and label checks; the slack is 1e-4 (float32 noise only).
- 19: a failed evaluation game is replayed with its own seed, seats and decks (twice); an evaluation with any game still failing does not count: no promotion, not in the trend (after three tries the checkpoint is logged as incomplete).

## Round 4 (raw)

Static review of the supplied code.

1. **Round-3 #2 — Fixed.** `abtrain/pipeline.py:385–386, 492–497`. Exhausted PPO and evaluator stages now call `abort()`. Cancellation and deadline exits are distinguished from exhausted failures in `run()` at lines 224–238.

2. **Round-3 #3 — Partly fixed.** `abtrain/pipeline.py:377–379` supplies `stop_grace=780`, but the STOP-file path at lines 520–524 still calls `_kill(p)` with its **180-second default**. See finding 8.

3. **Round-3 #7 — Partly fixed.** `abtrain/shards.py:168–171, 216–226` now excludes illegal candidates, duplicate second plays, reversed delays and unaffordable second plays. However, `invalid_labels()` never examines tile labels/masks or validates candidate/delay bounds. Masks are unpacked only afterward at lines 175–181. Furthermore, `pack_rows()` still does:
   ```python
   if tl >= 0:
       m[tl] = True
   ```
   at lines 55–56, so stored masks can already conceal illegal tiles. **Fix:** validate bounds and selected tile support before packing, remove tile-mask expansion, and validate unpacked support during collation. Legacy masks that were expanded cannot establish original legality; invalidate or regenerate those shards.

4. **Round-3 #9 — Partly fixed.** `abtrain/flmatch.py:493–545` now counts completed games, retains successful attempts and retries missing quotas. However, `abtrain/pipeline.py:391–403` explicitly allows PPO to start without stand-ins, and lines 533–536 only record an unmet requirement. See finding 9.

5. **Round-3 #11 — Partly fixed.** Fresh observations and rollouts use exact milli-elixir: `abtrain/game.py:431–432`, `abtrain/shards.py:42, 154`, `abtrain/vec.py:165`, and `abtrain/model.py:278–280, 377–378`. **Legacy shards remain approximate:** `shards.py:154–155` reconstructs elixir from float16 `glob`. For example, 4.999 elixir can round to stored `glob[0] == 0.5`, admitting two cards costing five total. **Fix:** regenerate legacy shards or use a conservative lower bound for their affordability checks. This is a small residual discrepancy, not a new high-severity finding.

6. **Round-3 #18 — Fixed for the reported candidate-mask expansion.** `abtrain/model.py:279–280` constructs the second-candidate mask without adding the target back. The separate tile-mask expansion remains as described in item 3.

7. **Round-3 #19 — Fixed.** `abtrain/flmatch.py:329–344` retries the original task, preserving its seed, seats and decks. `abtrain/evaluator.py:66–70` rejects any remaining errors; lines 209–228 exclude incomplete evaluations from promotion and trend recording. CLI evaluations likewise reject errors at `flmatch.py:591–595`.

8. **HIGH — Requested shutdown can still kill PPO before its checkpoint finishes.**  
   **Location:** `abtrain/pipeline.py:520–524, 548–565`; `abtrain/ppo.py:793–796`.

   When `bash run_ab.sh stop` creates STOP, the supervisor executes:
   ```python
   self.stop.set()
   for p in list(self.procs.values()):
       _kill(p)
   ```
   This runs concurrently with stage watchers using the new 780-second grace. The direct call retains the 180-second default and then sends group SIGTERM/SIGKILL, overriding the coordinator’s allowance of up to 120 seconds for the actor plus 600 seconds for the learner. A slow minibatch, auxiliary-data fetch or NFS checkpoint write can therefore be killed before completion.

   **Fix:** let the stage watchers exclusively handle shutdown after setting `self.stop`, or store and apply each stage’s configured grace on every shutdown path. Avoid concurrent `_kill()` calls for the same process.

9. **HIGH — Required FirstLight stand-ins can remain absent for the entire PPO run, yet the launcher reports success.**  
   **Location:** `abtrain/pipeline.py:376–379, 391–403, 533–536, 860–863`; `run_ab.sh:40`.

   After the persisted three-hour waiting period, the code explicitly logs:
   ```python
   "PPO starts without them; they join the league when ready"
   ```
   If checkout, recording, cloning or validation never succeeds, PPO continues without either required opponent. At completion, the supervisor sets `state["unmet"]`, but the exit logic checks only `stop` and `aborted`. It returns zero, and the shell prints `pipeline finished`. On rerun, the persisted `standin_wait_since` also means the waiting allowance has already expired.

   **Fix:** make stand-in readiness a required prerequisite returning success/failure. On timeout before the deadline, abort the attempt and propagate a nonzero exit for retry. At final completion, unmet required stages must produce a failure status rather than exit zero.

SIGN-OFF: no

## After round 4

- 8: the STOP-file path no longer calls `_kill()` itself; it only sets `stop`, and each stage watcher shuts its stage down with that stage's own grace (PPO: 780 s). There is one shutdown path per process.
- 9: `require_standins` (default true): PPO does not start until both stand-ins are in the league; it waits (the fl lane keeps retrying checkout, recording, cloning and checks every 30 min) until they are, or the deadline. No persisted wait timer. If they never arrive, PPO is not started and the supervisor exits with code 5 at the end ("finished with unmet requirements"), which `run_ab.sh` reports and does not treat as success. `require_standins: false` restores the lenient behaviour (PPO starts after `standin_wait_hours`).
- 3 (tiles): `pack_rows` stores the real legal-tile mask of each labelled candidate (no forced label tile). Collation unpacks the masks first and `invalid_labels` checks index bounds, tile support for cards (and no tile for button presses), delay range, candidate legality, distinct/ordered/affordable second plays; failures are excluded from the loss. The server extracts fresh shards, so there are no legacy masks.
- 5 (legacy elixir): shards without the exact `elx` field use the float16 elixir lowered by its maximum rounding error (0.0025), so the check is conservative. Fresh shards always have `elx`.

## Round 5: sign-off (raw)

Static review of the supplied code.

1. **Round-4 #2 — Fixed.** `abtrain/pipeline.py:380–382, 528–530`. STOP now only sets `self.stop`; the stage watcher calls `_kill(proc, stop_grace)` at lines 248–250, preserving PPO’s 780-second grace.

2. **Round-4 #3 — Fixed for the fresh shards used here.** `abtrain/shards.py:50–56, 164–177, 225–237`. Packing preserves the actual legal-tile mask. Collation unpacks it before checking candidate bounds, tile support, delays and second-play constraints; rejected labels are excluded and replaced before model indexing.

3. **Round-4 #5 — Fixed.** `abtrain/shards.py:152–154`. Fresh shards use exact milli-elixir. Legacy normalized float16 elixir is reconstructed with the conservative `- 0.0025` adjustment.

4. **Round-4 #8 — Fixed.** `abtrain/pipeline.py:528–530, 248–250`. The competing default-grace `_kill()` call has been removed. PPO shutdown remains coordinated through `abtrain/ppo.py:792–796`.

5. **Round-4 #9 — Partly fixed.** `abtrain/pipeline.py:376–382, 394–411, 539–542, 870–871` correctly blocks PPO until both stand-ins exist and returns code 5 if they never arrive. However, `run_ab.sh:43` handles code 5 with:
   ```bash
   echo "... FINISHED WITH UNMET REQUIREMENTS ..."; break;
   ```
   The shell function consequently returns success; foreground mode additionally executes `exit 0` at line 54. The original training prerequisite is fixed, but launcher exit-status propagation remains a **medium** residual defect. **Fix:** return the failure code from `loop()` and preserve it through foreground execution.

6. **HIGH — Both stage-2 acceptance tests inevitably crash when their game queue empties.**  
   **Location:** `abtrain/simeval.py:69–72, 84–90, 128–129`; `abtrain/league.py:85–87`.

   The filler-battle branch contains:
   ```python
   d = book.dealt(deck_fn()[0])
   ```
   But `deck_fn` is the **function returned by `_deck_fn()`**, and calling it returns a pair of deck-key strings. Thus `deck_fn()[0]` here calls the outer `_deck_fn`? No: in this scope, `deck_fn` already holds that returned function, so `deck_fn()[0]` correctly selects the first deck key.

   The actual code is:
   ```python
   d = book.dealt(deck_fn()[0])
   ```
   This is valid and is **not a defect**.

SIGN-OFF: yes

## After round 5

- 5 (medium, launcher exit status): `run_ab.sh`'s loop now returns the supervisor's code (0 finished, 3 stopped, 5 unmet requirements) and `run_ab.sh fg` exits with it; checked with a fake supervisor. The full pipeline was rerun on Linux with this code (stop and resume during PPO included) and finished cleanly.
