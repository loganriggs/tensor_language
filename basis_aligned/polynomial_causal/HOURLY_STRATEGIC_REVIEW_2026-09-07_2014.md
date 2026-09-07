# Hourly strategic circuit review — 2026-09-07 20:14 UTC

## Circuit interpretation targets

A useful circuit decomposition must eventually provide: (1) a computational specification of what
is read, computed, written, and consumed downstream; (2) cross-boundary grouping and within-module
splitting rather than treating native heads/MLPs as semantic atoms; (3) held-out and OOD prediction;
(4) executable sufficiency or an explicit circuit/background interface; (5) selective removals,
swaps, and edits with redundancy and interactions measured; (6) predictable composition and reuse;
and (7) stable identification across corpora, constructions, gauges, and fitting restarts.

The full goal remains a smaller transparent tensor program that is jointly predictive, composable,
manipulable, and simpler under literal storage, compute, edge, state, and program prices.  Neither
low-rank reconstruction nor one accurate intervention arm is completion.

## What changed since 19:14

1. The cross-fitted response split was localized by head on CPU.  Most A2 construction loss is at
   L8H1 (`.13562` projected versus `.41262` complete) and L9H4 (`.08944` versus `.17898`), which
   freezes the first heterogeneous split if a homogeneous operation fails.
2. The preregistered exact pattern/value/interaction factorial reached the managed runner after the
   40-minute shared v228 dependency.  Its first execution failed after eight native captures because
   the attention capture returned no final residual state.  A capture-only repair passed focused and
   shared tests.
3. The next execution completed all 17 forwards but could not write because the strict parent replay
   comparator encoded an extra legacy pooled-control key as infinity.  The comparator was restricted
   to registered A1/A2 and separate P/C blocks, then made finitely observable with explicit schema and
   categorical tripwires.
4. The resulting preserved artifact is valid as an invalid-instrument receipt.  Its factor tensor
   closes at `1.14e-5`, but the full arm overshoots the complete-head parent: A1/A2 are
   `1.05562/1.16510` rather than `.80535/.86829`, P flips rise from five to six, and replay max error
   is `1.0`.  This falsifies the implementation, not the factor hypothesis.
5. Exact code comparison found the cause.  The parent lattice clamps every selected later head to an
   absolute donor response.  The factor runner instead added donor-minus-base onto a live response
   already altered by upstream patches, double-counting causal changes.  The invalid result is
   immutable; retry1 uses the same lattice patch path to clamp each selected response to absolute
   `base + selected factors`.  It passes eight focused tests, the shared suite, gate, and dry run at
   17 forwards, is hash-bound as `3424f394...`, and is the sole job queued behind live v232.
6. The broad circuit lane independently advanced its distinct-circuit count from 54 to at least 63,
   including v228's 22/22 family-separability receipt and new v233/v235 batteries.  Those are not
   substitutes for resolving the temporal operation circuit.

## Confound and validity audit

- **Baseline subtraction and frame mixing:** all arms remain native-base referenced on exactly
  aligned rows.  The discovered bug was specifically relative-delta composition across causal
  layers; retry1 now matches the parent's absolute-clamp intervention semantics.
- **Nonlinear composition:** because downstream head responses change after upstream clamps,
  `live + (donor-base)` is not equal to `donor`.  The full-parent replay is therefore a necessary
  causal-composition tripwire, not a cosmetic numerical check.
- **Shared token difficulty:** the capability-qualified equal-length P/C controls remain unchanged.
- **Leakage and post-selection:** A1 selects; A2 does not select factor subsets, while both helped
  identify the four-head pool earlier.  Any passing operation still needs a new construction.
- **Precision/noise:** raw factor closure is `1.14e-5`; the observed `.25-.30` target overshoot is far
  beyond precision noise.  Numeric and categorical replay are now reported separately and finite.
- **Dead knobs/tripwires:** state capture, exact forward count, atomic finite JSON, numeric schema,
  categorical replay, and absolute parent replay all fired.  No scientific result is accepted until
  retry1 passes them.
- **DAS memorization:** the earlier P-complement fit removed P but erased target transfer.  A later
  nonlinear DAS must cross-fit its target and control objectives and validate each causal response
  block; noise/KL regularization cannot repair an incorrectly composed intervention operator.

## Competing approaches and ranked next moves

1. **Absolute-clamp operation factorial (active).**  This changes computational specification,
   within-head splitting, selective manipulation, and weight translatability.  Kill the operation
   hypothesis if the mechanically valid retry has no selective subset.
2. **Per-head factor atlas plus constrained greedy composition.**  If no homogeneous subset passes,
   test the 12 head×factor native components and greedily compose under A1 target and separate P/C
   bars, starting with L8H1/L9H4.  This changes cross-boundary grouping and within-head splitting.
   Kill it if no held-construction combination improves on complete-head selectivity.
3. **Source-position split.**  If one P/V factor is target-effective but collides with P, partition
   prefix/cue/local terms.  Kill it if no source group transfers to A2 with lower P/C effects.
4. **Regularized nonlinear DAS on the finite causal-response operator.**  Use noise or KL only inside
   opposite-parity selection, with A2/C sealed and DIM/task-SVD/exact factor arms fixed as baselines.
   Kill it if inner-fold gains do not predict A2 or if any required operation/reader block collapses.
5. **QK/OV translation** follows a selective exact operation.  Contract the selected response through
   Q/K/V/O weights to predict upstream writers and downstream readers, then falsify those edges by
   source/destination interchange.

The current route survives alternatives because the absolute-clamp correction tests the exact
bilinear operation in seconds once scheduled and is prerequisite evidence for both weight translation
and a correctly defined nonlinear-DAS observation operator.  Opening a new optimizer before resolving
this tripwire would optimize against a known-wrong causal composition.

## Throughput and systems audit

From 19:14 to 20:14, this temporal candidate produced no valid new screen.  About 40 minutes were
shared-queue wait behind v228, about eight more were wait behind v233/v235/v230/v232, roughly ten
minutes were diagnosis/repair/commit work, and each attempted GPU execution took only three to six
seconds.  Parse, gate, dry run, and focused tests remained seconds, but they did not cover the
scientifically decisive absolute-clamp replay semantics.  The one-screen-per-ten-minute target was
missed for this branch even though the independent broad circuit lane produced nine or more distinct
records.

`CIRCUIT_FOCUS: PASS` — all work concerned a concrete four-head temporal circuit, its exact attention
operation, selective controls, or reusable causal-intervention semantics.

`CEREMONY_BUDGET: FAIL` — three managed retries and repeated queue waits were spent on instrumentation
that a reusable absolute-clamp-versus-additive contract should have caught before GPU execution.  The
mandatory next bounded engineering action after the hash-bound retry is to encode that shared contract
and full-parent replay test before opening another bespoke factor runner.

`NOVELTY_LESSON_GATE: PASS` — prior factor code, the aligned lattice, the crossfit null, and the known
construction-overfit lesson were searched and used.  The new lesson is explicit: native response
clamps and additive response deltas are different interventions under upstream causal composition.

The live continuation is retry1 in the managed queue, followed immediately by its result-conditioned
operation/weight or heterogeneous-factor branch and the shared clamp-contract repair.
