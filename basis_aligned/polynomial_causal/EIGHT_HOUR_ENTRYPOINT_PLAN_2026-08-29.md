# Eight-hour reverse-engineering experiment queue

**Opened:** 2026-08-29 04:00 UTC  
**Deadline audit:** 2026-08-29 12:00 UTC  
**Scope:** finish the already-preregistered Block-3 Family-F experiment and run at
least three cheap, falsifiable probes for each of the first four alternative entry
points in `CURRENT_PROJECT_EXPLANATION_2026-08-29_0334.md`.

This is a work queue, not a promise that all twelve pilots will be positive or that a
failed pilot will be silently enlarged.  A completed negative result counts if its
inputs, prediction, measurement and failure are preserved.  No pilot earns a broad
claim from local MSE alone.

## Live status — 04:34 UTC

- Family-F v1 completed the full frozen call schedule, then failed terminal publication
  because a CUDA polarization maximum was compared directly with a CPU maximum.  The
  authority, exact program artifact and receipt-less failure are preserved.  V1 is
  spent; a fresh hash-pinned v2 recovery is now the critical path.
- E1.1 is complete and failed decisively, as recorded below.  This is useful pruning:
  the impressive map fitted on native one-token streams cannot be called a standalone
  compressed program.
- E1.3 also failed: three self-consistent map/refit iterations produced deficits of
  `5.49867 / 5.61939 / 5.59476` nat.  That is much worse than E1.1 and nearly erases
  the whole uncovered-token prediction.  The iterations were still changing strongly,
  so this is not a converged fixed point; nevertheless, the weakest registered test—
  whether fitting where deployed helps at all—failed by a very large margin.
- Those two independent failures close Entry Point 1.  E1.2 is pruned rather than run:
  localizing drift cannot rescue a map whose direct closed-input refit already makes
  prediction about five times worse.  Its freed time is reassigned to E2 and E3.
- The other agent's rank-512 price frontier now runs alone on the GPU.  CPU analyses,
  the Family-F v2 recovery protocol and E2–E4 harnesses remain eligible.

## What success by 12:00 UTC means

1. Family F has either a receipt-last numerical fit artifact or a precise preserved
   implementation/resource failure.  It is not left in an unaudited half-runnable
   state.
2. Each still-open entry point below has at least three recorded experimental outcomes.  A
   theorem/proof check or synthetic known-answer test counts only where explicitly
   listed; a plan or unexecuted runner does not count.  A branch may stop earlier only
   after two independent decisive falsifications make the remaining pilot unable to
   change the decision, as happened for stream closure.
3. Every result states its input, output, fitted information, held-out information,
   literal price, and the downstream consequence it predicts beyond reconstruction.
4. The 12:00 audit prunes weak branches and names the next one or two full experiments.

## Live status update — 05:10 UTC

- Family-F v2 is source-closed and queued behind the other agent's single active GPU
  frontier job.  The queue checks both the owning PID and the physical GPU process
  list before opening the fresh v2 namespace.  This is execution-in-wait, not an
  outcome; no v2 authority, result, receipt, or failure exists yet.
- The rank-512 frontier has completed two of three document roles.  On both, the large
  map improves all-position CE only with full or rank-256 tables (`0.04465/0.03582`
  nat on `skip7000`, `0.04832/0.03968` on `skip11000`); its gain is at most `0.00044`
  nat at table rank 64 and exactly zero at ranks 16/8/4.  These are partial log values,
  not banked evidence until the runner publishes its artifact.
- E2's CPU core now has the missing exact equal-storage comparator.  It allocates the
  independent sites' rank slots by fit-only predictive eigenvalues, rather than using
  an arbitrary common rounded rank.  The expanded known-answer suite passes `16/16`
  and commit `95acfeb0` is pushed.  This closes a fairness prerequisite but does not
  check E2.1 or E2.2: no real shared-map fit or held-out CE has run.
- A result audit found that the earlier source-closed tangent pilot is an exact bounded
  instance of E3.1 and had been omitted from this queue.  Its measured negative outcome
  is recorded below.  There is no retained vector-response bank from which a lawful
  CPU-only E3.2 can be manufactured; the next finite-composition measurement needs a
  new GPU collection.

## Immediate critical path: Family F

Family F asks whether selecting native MLP3 product gates by their *downstream suffix
consequence* makes an independently simplified MLP3 compose better than the earlier
activation-selected family.  It preserves the exact native gate grammar and changes
only support selection and the refitted Down map.

- [x] Close raw-pre-softcap replay, exact float32 affine deployment, Family-A overlap,
  parent hash-before/load/hash-after, continuous resource checks, and program/result/
  receipt semantic replay.
- [x] Pass focused and adversarial CPU tests (`91 passed`).
- [x] Obtain independent outcome-blind GO; source-close, commit and push
  (`119b968f2941d32f525d53e5529029e9aa92619f`).
- [x] When the current stream-closure GPU job releases the device, run the fit under
  the preregistered 45-minute/30-GiB ceiling.
- [x] Preserve the receipt or failure and write the scientific interpretation.  Fit
  KL alone never opens validation; only the uncalibrated real-F K256/K512 programs are
  promotive candidates.

  **Outcome:** implementation failure after the complete call schedule and program
  publication, before result/receipt publication.  Cross-device float32 reduction
  maxima were incorrectly treated as identical currency.  No scientific metric is
  promoted.  The v2 recovery must pin SHA256
  `d4af5bfbae03f8df9be8127e2e06c6f1a66b189be180ce72e5c74b6c7ac7a038`
  for the exact v1 program before deserialization and independently reconstruct it.

  **V2 recovery outcome, 05:16 UTC:** succeeded and published receipt last.  Exact
  program reconstruction, result semantics, and receipt replay pass.  The registered
  refitted Family-F programs fail the summed-write NRMSE gate badly: `0.78860` at K256
  and `0.70275` at K512 versus the frozen `<=0.20` bar, so neither may open validation.
  Downstream selection is still informative at K512 (`0.08476` document-balanced
  teacher KL versus `0.10077` random and `0.08862` Family A), but earns no composable-
  port credit.  Unexpectedly, retaining native Down is much better downstream
  (`0.05772` KL) despite worse local NRMSE (`0.86957`); the registered local decoder
  refit improves NRMSE while damaging the causal objective.  This diagnostic motivates
  a new prospective native-decoder/finite-edit family but cannot promote retrospectively.
  Static interpretation: `BLOCK3_CONSEQUENCE_FAMILY_F_V2_RESULT.md`.

## Entry point 1 — close the stream-map dataflow

Question: does the strong rank-512 stream map still work when its input is produced by
the compressed program rather than by a native one-token forward?

- [x] **E1.1 Recursive closure.** Feed each site map the site-entry
  stream recursively produced by the settled compressed prefix.  Report uncovered CE
  deficit on all three document roles and compare with embedding-r512, native-stream-
  r512, and the shared ceiling.

  **Outcome, 04:13 UTC:** failed both preregistered substantive bars while the controls
  passed.  Closed-stream rank-512 deficits were
  `1.08978 / 1.27276 / 1.26133` nat, versus
  `0.59560 / 0.67209 / 0.67172` for the rank-512 embedding map and
  `0.17427 / 0.21358 / 0.21419` for the native length-one stream.  Covered CE remained
  bit-identical and both prior controls reproduced.  Thus the native-stream result is
  not a standalone replacement: recursively accumulated state drift costs
  `0.916 / 1.059 / 1.047` nat.  E1.2 must localize that drift; E1.3 tests whether
  fitting and deploying on the same closed-stream distribution repairs it.
- [x] **E1.2 Drift localization — pruned after E1.3, not run.** At every one of the 36 sites, compare native and
  recursively generated map inputs by centered NRMSE, principal-angle/canonical-
  correlation summaries, and the downstream CE change when only that site's input is
  switched.  This distinguishes recoverable coordinate drift from accumulated causal
  state loss.  This could describe where the damage begins but can no longer alter the
  deployability decision, so it does not justify another GPU pass in this window.
- [x] **E1.3 Closed-input refit.** Fit the same rank-512 map on *covered tokens' closed
  program streams*, then evaluate on uncovered tokens' closed program streams.  This
  tests whether failure in E1.1 is merely train/deploy covariate shift.  Keep rank,
  coverage, roles, and price fixed.

  **Outcome, 04:31 UTC:** failed.  After three map → compressed streams → refit
  iterations, uncovered deficits were `5.49867 / 5.61939 / 5.59476` nat.  These are
  worse than E1.1's `1.08978 / 1.27276 / 1.26133`, worse than the deployable embedding
  map, and near the whole uncovered-token ceiling.  Covered controls remained exactly
  unchanged and all earlier anchors reproduced.  Relative map changes
  `22.63 → 5.44 → 1.86` were decreasing but not converged; the direct claim that
  closed-input fitting helps at all is nevertheless decisively false.

Pass criterion for further work: a source-closed map improves whole-program held-out
CE materially at matched price on all three roles without a native call or hidden
token-by-site table.  This criterion failed twice; the native-stream result is retained
only as an oracle diagnostic, and the deployable target is the rank-512 embedding map.

## Entry point 2 — factor all 36 maps jointly

Question: do the independently successful site maps speak a shared continuous output
language that is cheaper and easier to name?

- [x] **E2.1 Exact joint-RRR sweep.** Run the implemented simultaneous reduced-rank
  regression at shared output ranks 64/128/256/512.  Compare residual energy and
  held-out CE with 36 independent maps at both matched rank and matched stored floats.

  **Execution update, 05:53 UTC:** v1 spent its authority and failed before the first
  evaluation metric because CUDA token IDs indexed a CPU coverage mask.  It produced
  no scientific result or receipt, so this checkbox remains open.  A fresh v2 recovery
  changes only that device placement, binds the exact v1 authority/failure, and must
  pass a new source closure and independent audit before launch.

  **V2 outcome, 06:09 UTC:** completed with receipt and exact semantic replay.  No
  global rank passes both registered CE conditions.  Global ranks 64/128 beat the
  strongest equal-storage independent allocations by `0.022--0.036` nat on every
  role, proving useful low-rank sharing, but lose to same-rank independent maps by
  `0.038--0.070` nat.  At rank 512 the global arm loses both comparisons.  E2.1 is a
  measured negative for one universal output dictionary, with positive evidence for a
  shared trunk plus private residuals.
- [x] **E2.2 One dictionary versus two.** Compare one global output basis with separate
  attention and MLP output bases at equal total rank/storage.  This tests whether the
  apparent sharing is architectural or merely caused by the common residual space.

  **Outcome:** typed rank 481 improves over global rank 494 by only
  `0.00250 / 0.00237 / 0.00004` nat at exactly equal storage, below the frozen 0.01
  margin.  Attention/MLP typing helps at rank 64 but is not a sufficient canonical split.
- [x] **E2.3 Stable sparse coordinates — pruned after E2.1, not run.** Rotate the best shared subspace with a frozen
  sparse/dictionary objective on fit data, then measure support/direction stability on
  two disjoint roles and whether single-coordinate interventions have concentrated
  downstream effects.  Reconstruction alone is insufficient.

  No shared projector passed E2.1, so rotating one cannot restore missing private
  directions.  The freed branch moves to a hierarchical shared-plus-private factorization.

Pass criterion: shared factors reduce literal storage or improve held-out prediction
at matched price, and their coordinates show cross-role stability or selective causal
effects.  Dense rotationally arbitrary coordinates count as compression, not semantic
interpretation.

## Entry point 3 — work backward from downstream consequences

Question: can a small predictive state summarize everything downstream needs from an
early component, rather than reconstructing its full residual write?

- [x] **E3.1 Response-panel rank.** Build a small vector-valued matrix whose rows are
  controlled early-component/prefix interventions and whose columns are later residual
  directions plus selected logit groups.  Measure held-out singular-value/rank
  stability across documents and intervention amplitudes.

  **Existing measured outcome, audited 05:10 UTC:** the source-closed MLP0--2
  final-output Fisher-tangent panel in
  `tensor_bilin18_tangent_pilot_results.json` (SHA256 `efd788fa0089008c4a2b0767244f1759453f02dd6e98b31aceae3847b26bc9d4`)
  used 96 whole-document-split rows and took 199.87 seconds.  At cuts 1/2/3 the exact
  ranks stayed at the full registered column dimensions 32/64/96, no compression knee
  was selected in either split, normalized squared-spectrum L1 drift was
  `0.18785/0.16928/0.17641`, and exposure-normalized trace drift was
  `0.71767/0.60919/0.60332`.  All split-stability gates failed.  This is a genuine
  negative E3.1 outcome for infinitesimal final-output Fisher responses.  It does not
  test finite amplitudes, unseen intervention compositions, residual-direction
  targets, or selective edits, so E3.2/E3.3 remain open.
- [ ] **E3.2 Unseen-composition prediction.** Fit a predictive-state realization on a
  subset of prefix/intervention × suffix-reader cells and predict sealed cells.  Score
  vector error and resulting KL/CE; compare with equally priced local-MSE PCA/RRR.

  **Pre-execution audit, 05:34 UTC:** the existing L8→L11→L14 transport-triangle
  runner is scientifically nonredundant with E3.1: it fits finite antithetic edits and
  predicts an unseen donor-minus-target response through the composed map
  $T_{8\to11}T_{11\to14}$ without reading the true L11 response.  It is nevertheless
  a NO-GO for launch.  The frozen FineWeb receipt (SHA256 `815b2161...`) contains
  `96/33`, `96/33`, and `192/105` rows/unique documents in its three relevant roles,
  while the runner correctly requires one row per source document.  Weakening that
  check would create pseudoreplication.  The runner also lacks the registered
  source-closed create-only lifecycle and most full null/control families.  A new
  immutable 96+96+192 unique-document receipt and lifecycle hardening are required;
  no E3.2 experimental outcome is claimed.  New synthetic contract tests establish
  that the finite chain uses the fitted maps rather than the true intermediate and
  fails when the first transport is broken.

  **Unique-row recovery update, 06:15 UTC:** a source-closed metadata allocator froze
  96+96+192 roles with 384 distinct documents and no cross-role reuse.  Its first CPU
  materialization failed before rows publication because the parent receipt's raw-byte
  tensor hash was compared with a dtype+shape+bytes hash.  Authority and failure are
  preserved; no rows, manifest, receipt, model response, or E3.2 result exists.  A v2
  recovery may change only that hash currency and must bind the identical selection
  plan plus spent parents.
- [ ] **E3.3 State-variable edit test.** Remove or transplant one learned state
  direction and test target effect, collateral effect, and OOD transport.  A state is
  useful only if it predicts a new composition or supports a selective edit.

Pass criterion: held-out cross-composition prediction or selective edits improve over
local-reconstruction baselines at matched state dimension/price.  A low in-sample
matrix rank alone is not enough.

## Entry point 4 — terminal or behavior-anchored circuits

Question: can a short causal path produce the first genuinely extracted and selectively
removable circuit, thereby telling us which simplicity measure has practical value?

- [ ] **E4.1 Terminal-layer screen.** Rank the last blocks' attention/MLP product-gate
  groups by causal effect on sharply defined output classes, using positive, matched
  negative, and off-target examples.  Prefer paths with short distance to unembedding.
- [ ] **E4.2 Three behavior probes.** Run small controlled suites for capitalization,
  number formatting, and copy/continuation behavior.  For each, require held-out
  templates and a natural-text replication slice; report effect and collateral CE.
- [ ] **E4.3 Extraction/removal pair.** For the strongest behavior/site pair, compare a
  sparse extracted program, native circuit ablation, and extracted-program removal or
  transplant.  Measure task accuracy/logit effect, off-target damage, OOD transport,
  and executable cost.

Pass criterion: one behavior admits a held-out predictive circuit whose intervention
has a concentrated intended effect and limited collateral damage.  Top-1 agreement
may support extraction, but faithful prediction/removal additionally requires KL/CE
and OOD checks.

## Eight-hour ordering and resource allocation

| UTC window | Main action | Parallel CPU action |
|---|---|---|
| 04:00–05:15 | Harden/audit Family F while E1.1 owns GPU | Freeze E1.2/E1.3 and E2 harness inputs |
| 05:15–06:15 | Read E1.1; run E1.2/E1.3 if cheap | Run E2.1 sufficient-statistic checks |
| 06:15–07:15 | Run Family-F fit when GPU is free | Implement/test E2.2 and E3 response-panel reducer |
| 07:15–08:30 | Preserve Family-F result/failure; launch best E2 pilot | Run E3.1/E3.2 small panels |
| 08:30–10:00 | E2.2/E2.3 or E3 GPU follow-up | Prepare and run E4.1 terminal screen |
| 10:00–11:30 | E3.3 and E4.2/E4.3 focused pilots | Consolidate prices, causal and OOD metrics |
| 11:30–12:00 | Stop opening branches; audit all 12 cells | Rank next full experiments and update explanation |

If a long GPU job consumes a window, the queue advances through CPU-known-answer
tests, sufficient-statistic factorizations, response-matrix design, artifact review,
and result consolidation.  No second GPU job is launched concurrently on this device.

## Alarm and audit contract

The existing hourly Codex cron is the recurring progress alarm.  During this window its
prompt explicitly points to this file.  A separate one-shot deadline script queues a
12:00 UTC completion audit into the same thread.  Both are session-local; their durable
install entries live in `codex_session.crontab`.  The deadline script refuses to fire
outside 2026-08-29 11:55–12:20 UTC, so a stale annual cron entry cannot create a false
future alarm.
