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

  **Hierarchical successor outcome, 07:18 UTC:** receipt-complete negative.  At the
  global-rank-512, typed-rank-512, and independent-rank-512 storage budgets, shared
  rank 128 loses to the all-private exact-price endpoint on all three discovery roles.
  At the global price it lies between all-private and all-shared: private residuals
  help, but the shared trunk does not repay the 2,368 private slots it costs.  All
  integrity and endpoint controls pass.  This closes the rank-512-scale hierarchy;
  only a bounded tight-budget follow-up remains nonredundant because flat sharing was
  positive at global rank-64/rank-128 prices.  Static result:
  `HIERARCHICAL_SHARED_PRIVATE_RRR_REAL_V2_RESULT.md`.

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
- [x] **E3.2 Unseen-composition prediction.** Fit a predictive-state realization on a
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

  **V2 unique-row outcome, 06:27 UTC:** the one-change recovery completed receipt
  last.  It materialized 96 basis + 96 fit + 192 evaluation rows from 384/384 distinct
  source documents, with no cross-role reuse and every evaluation source after fit.
  Rows SHA256 is
  `102b79726b7132a6438b4080272fee1774499ac4fc83c4aa025fa86439b4074d`;
  receipt-file SHA256 is
  `3f92d8b3aa5e89e6059a010338521bffa0cf440e0815d9d67e1b65aa58a8e102`.
  This removes the pseudoreplication blocker but is not an E3.2 result and explicitly
  does not authorize the triangle runner.  Source closure, create-only lifecycle, and
  the full finite-composition controls remain required before launch.

  **Receipt-backed outcome, 08:39 UTC:** negative for the frozen pointwise rank-64
  L8 $\rightarrow$ L11 $\rightarrow$ L14 grammar. The exact full-response oracle
  passes (`E_out=1.50e-11`), but the true response projected into the rank-64 L14
  basis already fails sufficiency (`E_out=0.2709 > 0.25`). The direct map scores
  `E_out=0.4861`, coordinate `R2=0.4028`; the unseen composed chain scores
  `E_out=0.4520`, `R2=0.4024`. All harness, scale, position-shuffle, gauge, and price
  controls pass. The chain does not incur a special extra collapse relative to direct,
  but neither representation is adequate. V1's import failure and v2's receipt-only
  failure are preserved; v3 bound and semantically validated the complete v2 result
  and state without a scientific rerun. Static result:
  `GAUGE_TRANSPORT_TRIANGLE_V1_RESULT_2026-08-29.md`.
- [x] **E3.3 State-variable edit test — pruned for this interface after E3.2.** Remove or transplant one learned state
  direction and test target effect, collateral effect, and OOD transport.  A state is
  useful only if it predicts a new composition or supports a selective edit.

  No rank-64 state passed the destination-sufficiency or transport gates, so editing
  one of its coordinates cannot test a licensed state variable. Running E3.3 in this
  failed representation would turn a locator into an API retrospectively. Higher-rank,
  behavior-specific, temporal-kernel, or nonlinear states remain separate future
  grammars rather than rescues of this cell.

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

**07:40 UTC implementation update — not an evidence cell:** a prospective copy/
induction contract now freezes the named six-head family, registered four-head subset,
late pair, deterministic natural matched negatives, length/multiset-matched synthetic
nulls, and separate CE/KL/extraction/removal/OOD/price currencies. Nine synthetic
known-answer tests pass. No fresh row, checkpoint, model forward, or authority was
opened, so E4.1--E4.3 remain unchecked. Exact launch blockers are fresh four-role rows,
reviewed per-head attention and optional late-product adapters, scorer/bootstrap
authority, checkpoint binding, and create-only terminal lifecycle.

**08:55 UTC adapter update — still not an evidence cell:** the source-owned per-head
attention adapter now has a receipt-last live-checkpoint result at all five distinct
layers containing the six registered copy heads. After two fail-closed attempts exposed
contraction-layout and rotary-dtype mistakes, v3 makes the unpartitioned write and
shared value bus bit-identical to native at layers 5/7/8/13/14. The raw nine-head sum
differs by `0.002627--0.002667` relative because separate bfloat16 `c_proj` contractions
change accumulation order; this residual is explicit. Fourteen CPU contract tests pass.
Receipt SHA256 is `c5ef51670b6e23bb3cddbbef6c5cd451dff55eea8b8f7ddfdf20aca7374bb324`.
This closes the per-head formula/checkpoint binding only. Fresh four-role rows, scorer/
bootstrap authority, explicit late-MLP omission or adapter, and complete create-only
behavioral lifecycle remain, so E4.1--E4.3 stay unchecked.

**09:15 UTC scientific-contract update — still not an evidence cell:** two independent
reviewers now give GO to the corrected pure copy contract after three adversarial
review rounds. The label uses the nearest prior query occurrence; query and target
fit frequencies are distinct and preserve a zero-count sentinel; retained matched
cells remain document-balanced; causal effects are native-to-ablation changes on the
exact same row/position support; and the synthetic test is a reciprocal association
difference-in-differences with fixed length and multiset. Exact ordered row bytes are
bound into every reduction's support digest. The focused contract/freezer suite passes
`24/24`. The prospective screening amendment narrows this first run to attention-only,
copy-only localization and explicitly defers late MLP, the other two behaviors, and
E4.3 extraction/removal/transplant. No rows or model outcomes were opened. The next
blocker is the independently reviewed create-only row authority plus streaming
scorer/dispatcher; a separate GPU job is active, so current work remains CPU-side.

**09:55 UTC row/label publication — prerequisite complete, still not an evidence
cell:** v1 failed safely before model access on a stale reference to an older failed
row transaction. A narrowly scoped v2 recovery passed an independent 59-test audit and
the complete 112-registry/29-tensor census, then published 192 fit, 192 selection, 192
final, and 192 code-OOD rows. All support gates pass: selection has 303 positive and
303 matched-negative positions, final 247/247, and OOD 1,294/1,294. Receipt SHA256 is
`aea52a94c643906ef822a7c6ddb37a371b4315507a1a0a79acd539a19ae7f5c8`;
it records no model import, checkpoint load, forward call, or outcome access. This
closes the fresh-row/token-label blocker only. E4.1--E4.3 remain unchecked; the next
critical path is the physical candidate dispatcher plus receipt-last streaming scorer.

**10:15 UTC dispatcher-core update — implementation complete, still not an evidence
cell:** `terminal_copy_attention_dispatcher.py` now binds the exact eight frozen
candidates to physical `(layer, head)` sets and computes
`full_native - selected_heads + fit_position_mean` from owned attention adapters. The
L8H3+L8H4 pair is one same-layer transaction; multi-layer candidates preserve the
first-value bus; returned tensors are non-aliasing; invalid candidates, layers,
sequence lengths, and mean-bank topologies fail closed; and the scorer's owned adapter
and fit-mean values are priced. The combined dispatcher/adapter/contract/statistics
suite passes `35/35`. This is not a runner and opened no row, checkpoint, model, or
outcome. Remaining launch blockers are a fit-role per-position head-mean authority and
receipt, production call/support ledgers and source closure, explicit attention-only/
native-MLP omission authority, and the create-only selection/result lifecycle. The
launch gate now requires `physical_candidate_dispatcher` separately from the adapter.
E4.1--E4.3 remain unchecked.

**10:35 UTC fit-mean/owner update — infrastructure only, still not an evidence
cell:** the mean bank is now sparse over exactly the six licensed heads, accumulates
source writes one document at a time on CPU float64 in frozen receipt order, retains a
sealed float64 master and separately hashed float32 runtime cast, and is invariant to
batch partition. A hook-free native collection owner advances all 18 blocks without
calling the unembedding, uses original native writes for the live trajectory, and
independently decomposes only layers 5/7/8/13/14. A separate candidate owner executes
multi-layer candidates sequentially on their live counterfactual states, calls every
MLP natively, and records native/adapter/site integrity. Production rejects short
states, null value buses, non-bfloat16 physical state/adapters, non-float32 means,
wrong model weights, and any population other than 192 ordered 256-token documents.
Hashed banks expose clones only; owners privately clone their instruments and become
permanently poisoned after a partial forward failure. The focused suite passes
`56/56`. This still opens no model or E4 outcome. Remaining launch blockers are the
create-only parent/source authority, model pre/post and row/support ledgers, atomic
mean bundle/manifest/receipt lifecycle, and a selection scorer that reduces and
discards logits without response escape. S1918 owns the GPU, so no launch was attempted.

**07:40 UTC whole-program diagnostic — actual run, not an E1--E4 completion:** the
deployed-scale sweep measured top-1 and permutation-normalized teacher agreement on all
three discovery roles. Scale 0.8 was best for top-1, scale 0.5 best for agreement, and
per-site native-norm scaling was catastrophic. Its advertised CE field was null. The
separate all-position-CE sweep then completed in 214.6 seconds: scale 0.8 was best by
CE on all roles, but improved over 1.0 by only `0.00407 / 0.00992 / 0.00447` nat, and
therefore stayed within the frozen 0.01-nat tolerance. Scale 0.5 was fifth of six by
CE, while per-site native-norm scaling reached `12.81--12.94` CE, worse than uniform.
This closes the cheap scalar diagnostic but does not close an E1--E4 cell or move the
strict ledger.

## Live status update — 08:00 UTC

- Family F itself remains receipt-complete and negative under its registered local
  NRMSE gate. Its most informative successor is the native-Down finite behavioral port.
  That successor remains a NO-GO: its own protocol requires an independent audit before
  publishing the fresh 192-document role, and the CUDA measurement adapter/result
  semantic validator is not implemented. No row or model outcome is claimed.
- E1 is still closed negative. E2.1/E2.2 and the rank-512 hierarchical successor remain
  closed as recorded above. E3.2/E3.3 and E4.1--E4.3 still have no outcome; scaffolds and
  caches are not counted as evidence cells.
- A non-cell CPU mathematical probe completed while S1897 owned the GPU. The frozen
  v3 rank-64 correction projectors are worse than matched Haar projectors at separating
  the native MLP1/MLP2 quadratic tensors; their rank-128 union is worse again. The
  preregistered 64-sample failure survives a labeled 128-sample robustness run. This
  prunes direct-sum/HOSVD canonicalization with these projectors and redirects the
  native-Down successor toward reachable/observable balancing. Static result:
  `FIXED_PROJECTOR_QUADRATIC_CLOSURE_RESULT_2026-08-29.md`.
- S1897/S1898 are measured whole-program mechanism diagnostics, not E cells. At 16,110
  covered types, attention 13 is still the best restoration on all roles and attention
  5 still collapses the program. MLP gains are at most 0.0042 enrichment, but direct
  comparison shows that late MLP restorations change up to 3.66% of predicted tokens;
  attention 5 changes about 96%. Thus MLPs are lower-sensitivity rather than inert.
  The results motivate a fit-only uneven rank allocation but cannot be used as its
  sealed evaluation evidence.

## Live status update — 08:15 UTC

- S1899/S1901 establish covered-token stream/output identity, while S1898/S1900 count
  all positions. A source-closed coverage split resolves the apparent contradiction:
  `1311/1321`, `1342/1350`, and `650/650` changed predictions occur on uncovered current
  tokens. Covered live MLP16 versus table-row error is `3.55e-7--3.57e-7`; uncovered
  live MLP16 versus learned-fallback-row error is `0.328--0.336`. Thus the covered-table
  no-op derivation is correct and its extension to fallback rows was false. S1902 also
  confirms zero restored-arm and all-compiled self-differences on all roles while
  reproducing the between-arm counts. These are measured mechanism diagnostics, not
  E1--E4 cells.
- The source-closed exact-budget uneven table-rank experiment completed in 244.1 seconds.
  The preregistered normalized-energy allocation loses to uniform rank 512 by
  `0.01944/0.02053/0.02309` nat and to its shifted null on every role. A diagnostic raw-
  energy allocation improves over uniform by `0.00772/0.00675/0.00447` nat, missing
  the 0.005 bar on the third role. This prunes normalized local compressibility as a
  rank allocator and leaves a prospective raw/causal-weighted successor. It is a real
  whole-program outcome but, as preregistered, does not complete an E cell because the
  three evaluation roles were already spent.
- Strict cell status is unchanged: E1 is closed negative; E2.1/E2.2 and the large-budget
  hierarchy are closed negative; E3.2/E3.3 and E4.1--E4.3 have no model outcome. Family
  F remains receipt-complete and negative under its registered NRMSE gate, with the
  native-Down successor blocked by independent audit plus missing measurement and
  semantic-validation code.

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

## 11:15 UTC E4 fit-input and transaction update

This is a prerequisite plus a preserved implementation failure, not an E4 evidence
cell. An audit found that a development test had deserialized the combined fit-row
container before fit authority. No model/outcome was opened, but the label-container
access is preserved in an erratum. A separately audited projection now publishes only
`long[192,256]` inputs and ordered document IDs; its receipt is complete, postvalidated,
committed, and pushed.

The hardened fit lifecycle passed 88/88 and received independent GO. Its first
authorized model transaction failed before accepting a document because exact Rotary
identity was compared across CPU/CUDA with device-sensitive `torch.equal`; no bank,
result, manifest, or receipt escaped. V2 is preregistered and pushed, changes only the
cross-device exact-value comparison, and binds v1 authority/failure. It awaits audit
and the GPU used by S1922. E4.1--E4.3 and strict ledgers remain unchanged.

## 11:25 UTC E4 recovery outcome

V2 passed independent audit, froze a committed authority, and launched after S1923
released the GPU. It completed the first batch's native computation and private
accumulation, then failed before accepting the batch because the final-state integrity
hash called NumPy on `bfloat16`. No bank/result/manifest/receipt was published. The
failure is preserved and E4.1--E4.3 remain unchecked.

V3 is a one-change engineering recovery: hash exact raw bytes through a `uint8` view,
with dtype and shape still in the hash domain. Its current focused suite passes
`40/40`; independent audit, source closure, fresh authority, and a clean run remain
required. This is not an evidence outcome.

Current literal cell tally is six measured negatives (E1.1, E1.3, E2.1, E2.2, E3.1,
E3.2), three scientifically pruned cells (E1.2, E2.3, E3.3), and three open E4 cells.
The older 08:00/08:15 statements that E3.2 lacked an outcome are superseded by its
08:39 receipt-backed negative.

## 11:31 UTC v3 fit prerequisite complete

V3 passed independent audit and completed receipt-last on all 192 fit documents.
Semantic bank replay and all authority/bank/result/manifest hashes postvalidate.
Master/runtime mean digests are `3e494ad2...` and `d8b90d58...`; receipt-file SHA256
is `663d1f85...`. No unembedding, logits, losses, labels, copy cells, or selection
outcome were accessed.

This is still not an E4 evidence cell. The critical path has moved from fit collection
to the source-closed eight-candidate selection scorer and its authority. E4.1--E4.3
remain unchecked; strict ledgers remain unchanged.

## 11:45 UTC selection-interface update

No E4 cell is complete. The fit bank is now usable as a semantically replayed
prerequisite, but its receipt correctly does not self-authorize selection. A synthetic-
tested batch owner now executes one shared native arm plus all eight live sequential
candidates and returns only document-cell sufficient statistics. It retains exact
closures and verifies shared native baselines, exact head plans/site calls, document
order, finite recomposition bounds, and logit nonescape.

A schema-only engineering inspection deserialized `selection_natural.pt` before
authority. No values or model/model-outcome were observed; the exposure is preserved
and does not silently receive a pristine-container claim. A prospective execution
ruling also resolves original-preregistration versus screening-amendment conflicts.

Current suite: 25/25. This is CPU implementation evidence, not E4.1. The remaining
NO-GO is the independently audited source-closed selection loader, mask reconstruction,
48-batch authority/lifecycle, bootstrap replay, and receipt-last publisher.

## 12:35 UTC deadline audit and E4 lifecycle update

The eight-hour window is over. Its literal tally remains six measured negatives, three
scientifically pruned cells, and E4.1--E4.3 open. Family F completed receipt-last and
failed its registered NRMSE gate; it is not being counted as a successful composable
port. Plans, cached rows, fit means, and unrun selection code do not count as outcomes.

The E4 selection NO-GO has narrowed substantially. The source-closed lifecycle now
binds 28 exact files and independently reconstructs natural masks plus all 32 reciprocal
synthetic crossover pairs. It freezes 48 natural and 16 synthetic batches, one native
plus eight candidates per batch, exactly 576 outer forwards, and a literal 10,000-draw
shared-document bootstrap over 24 coordinates. It has mutually exclusive passer,
scientific-negative, and failure terminal states; only a passer can open final/OOD.

The previous audit findings have been repaired: synthetic banks must be cross-item
unique and base-row absent, assurance tests are source-closed, mocked full state-machine
paths are exercised, and failure artifacts hash-join partial outputs plus the protected
snapshot. The lifecycle suite passes 17/17 and the full assurance suite 60/60. The
current draft is explicitly nonauthorizing. Independent outcome-blind re-audit is in
progress while S1929 owns the GPU; no E4 selection value or model outcome has been read.

Static strategic interpretation and ranked next actions:
`HOURLY_STRATEGIC_REVIEW_2026-08-29_1235.md`.

## 13:03 UTC final deadline audit

The authoritative cell balance remains:

- measured negatives: E1.1, E1.3, E2.1, E2.2, E3.1, and E3.2;
- scientifically pruned without execution: E1.2, E2.3, and E3.3;
- open without model outcome: E4.1, E4.2, and E4.3.

The deadline replay independently revalidated Family F's saved result and receipt.
Result SHA256 is
`18b03ccf3d6710813375bb7e09b1a3c313d5e7790e2ca3c9a9b683fbf91897c5`;
receipt-file SHA256 is
`e81673095c7b6202fdec293c6ad34924fb9acb15213d02ba4b203d5ff8c65a5a`.
The preserved v1 failure remains at SHA256
`1bb45f2645576fadef564562ef37f98abfb64afb75af8396b882fe63b783f79b`.
The v2 fit is a registered negative: NRMSE `0.78860/0.70275` at K256/K512
misses the `<=0.20` gate, zero validation/final rows were opened, and the receipt
explicitly grants no global-ledger credit. The native-Down K512 diagnostic's better
fit KL (`0.05772`) motivates a fresh prospective experiment but is not promoted.

E4's independent lifecycle audit, canonical audit artifact, and one-shot authority are
now committed and pushed. The corrected launch is queued behind S1930's active GPU
process. Only the authority exists: no lock, selection value, model outcome, ledger,
result, receipt, or failure has been opened. Accordingly all E4 checkboxes remain open.

Weak branches are pruned exactly as stated in
`CURRENT_BEST_UNDERSTANDING_PLAIN_ENGLISH_2026-08-29_DEADLINE.md`. The next full
experiment is the E4 copy transaction and, conditionally on a passer, its registered
final/OOD extraction-removal sequence. The second is a new prospective Family-F K512
native-Down finite behavioral port on fresh documents. Strict ledgers remain 36/36
structural, 5.348245316% storage removed, 10.923302467% named causal CE,
4.72714 nat unexplained, and 0/68 terminal actions.
