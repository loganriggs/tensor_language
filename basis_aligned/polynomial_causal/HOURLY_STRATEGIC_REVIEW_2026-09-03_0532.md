# Hourly circuit-strategy review — 2026-09-03 05:32 UTC

## Circuit interpretation targets

A decomposition is useful only if it advances these seven targets:

1. **Computational specification:** say what information is read, what operation combines it, what is written, and
   which later computation uses it.
2. **Grouping across modules and splitting within modules:** merge pieces of different heads or MLPs when later
   computation treats them as the same variable, and split one native head or MLP when its pieces do different jobs.
3. **Held-out and OOD prediction:** predict activation and causal effects on unseen documents, task variations, and
   shifted data.
4. **Extraction or sufficiency:** install an isolated executable circuit, or a precisely specified interface plus
   background, that reproduces the target computation or signed causal effect.
5. **Selective manipulation:** removal, swapping, or editing changes the intended behavior while preserving unrelated
   behaviors, with redundancy and interactions measured explicitly.
6. **Composition and reuse:** the same subcomputation serves multiple tasks or modules, and jointly installed pieces
   behave predictably.
7. **Stable identification:** the unit survives document splits, fitting seeds, and allowed changes of basis, or is
   defined by an operational equivalence that later readers cannot distinguish.

The program-level goal remains a smaller transparent tensor program that predicts fresh and OOD text, composes under
joint installation, can be selectively manipulated, and is literally simpler in storage, compute, states, and edges.
Lower rank, reconstruction loss, or CE alone is not a circuit result.

## What changed since 04:32

Rung 521 moved from a preregistration to a fully audited and live Stage-A test.

- The original four-level matched-control rule was infeasible in 8/16 quartet cells when controls were correctly
  restricted to the same FIT half. Matching controls over all FIT rows would have leaked half-1 controls into the
  half-0 power estimate. That shortcut was rejected.
- A deterministic one-to-one matcher now preserves the original four levels and adds terminal fallbacks. Across all
  54,014 pairs, 485 (0.90%) use a new level. Across the 1,886 primary quartet-exclusive pairs, only 16 (0.85%) do;
  none crosses token class. The result is therefore feasible without erasing the original matching semantics.
- Whole-attention donors are coherent row permutations: every position in a recipient sequence uses the same donor
  sequence at the same position. All 24 maps are different-document bijections, use eight distinct donors per
  recipient, and match the decile of row-mean native CE exactly. This is substantially better than the rejected
  circular-shift design, whose mean difficulty gap was about three deciles.
- The create-only preflight receipt freezes 144 control matchings and all donor hashes. The mathematical library and
  Stage-A code pass 19 focused tests, both static gates, dry-run, syntax, and the repository fast suite.
- The managed smoke passed: direct and dispatched logits are bit-identical; a self-donor is an exact no-op; attention8
  is called once; a real donor changes its output by RMS 46.658 and changes logits. The Stage-A liveness floor is
  frozen at 4.665819.
- Stage A is now live in the managed runner. It runs 2,698 inference forwards and zero backwards, then stops before an
  optimizer regardless of outcome.

Claude's independent effect-space analysis also changed the interpretation. The pooled MLP10 source effects contain
a stable approximately three-dimensional shared pattern, concentrated on several block-6 circuits. After removing
that shared pattern, the remaining MLP10 source-specific effect is below its independent-half null. This supports a
shared variable at current sample size but warns that a private residual may be too noisy to identify. It does not
directly prove anything about the larger whole-attention8 activation intervention tested by Stage A.

## Is Rung 521 still the highest-information route?

Yes, through Stage A and then the shared stage if A passes.

The decision is not whether rank four reconstructs attention8. Stage A asks whether the exact causal object that a
future projector must reproduce is stable across two document halves and two independent donor ensembles. A pass
licenses fitting; a failure says the instrument is too noisy and prevents an optimizer from manufacturing a pattern.
That directly protects targets 3, 5, and 7.

Conditional on A, the leave-one-circuit-out shared projector tests targets 2, 3, 4, 6, and 7: one activation subspace
must be learned from two circuits, predict the third, transfer to the historically held-out fourth, stay quiet on
matched controls and other attention8 circuits, and survive seeds and unseen donors. Rank four is merely fixed
capacity shared with oracle and permutation controls.

The private stage is no longer automatically licensed by a shared-stage pass. Before private optimization, the
remaining full-attention8 effect after applying the frozen shared projector must itself reproduce across unseen donor
ensembles and held-out documents. If that residual-power test fails, the correct label is “private target
underpowered,” not “no private subspace.” This is the concrete response to the new MLP10 effect-space result.

## Confound audit

- **Baseline subtraction:** Stage A recomputes native CE from the same model execution used for each edited batch.
  Member and control effects are both edit-minus-native. No stale baseline is mixed in.
- **Frame mixing:** the edited object is the post-output-projection attention8 write in the 1,152-dimensional residual
  coordinates used by the next computation. No head coordinate is treated as the semantic basis.
- **Nonlinear loss composition:** Stage A runs every finite swap through the complete suffix. It never estimates a
  joint CE effect by adding marginal CE changes.
- **Shared token difficulty:** controls match next token, position, and native CE where feasible; only 0.90% use a new
  terminal level. Donor rows match row-mean native-CE decile exactly, but this does not guarantee per-token CE
  matching. The two donor ensembles and matched controls must absorb that remaining nuisance.
- **Leakage:** FIT halves, VALIDATION, and TEST are document-disjoint. Same-half controls are enforced. TEST cannot
  choose rank, seed, threshold, donor rule, or projector.
- **Dead interventions:** exact self-donor no-op and nonzero real-donor edit passed before science. Stage A checks the
  minimum batch edit RMS against the frozen floor.
- **Precision:** the model and captured writes are float32; native replay and self replacement were exactly equal in
  the smoke. No BF16 subtraction is used.
- **Post-selection:** the quartet predates Rung 521; `r.2.0.1` remains excluded from fitting; every other attention8
  circuit is a registered negative or evidence that the shared unit is broader.
- **Power:** the per-source MLP10 fingerprints are known to be weak. Stage A therefore gates the larger exact
  whole-attention8 object before learning. The future private residual needs its own gate.

## Different routes and what would kill them

1. **Current shared activation-DAS route.** Highest information if Stage A passes because it produces an installable
   projector and a direct reuse test. Kill it if whole-attention8 responses fail donor/half reliability, or if the
   shared projector does not beat matched rank-four oracle/permutation controls on held-out circuits.
2. **Pooled downstream-effect basis.** Use the stable three-dimensional MLP10 circuit-effect pattern as an output
   target, then learn an activation intervention that predicts that pattern. This is cheaper and already reliable,
   but it is a lossy readout and may merge distinct activation computations. Kill it if an activation direction that
   fits the pooled pattern fails physical swaps or predicts no held-out circuit effects.
3. **Higher-document source-specific decomposition.** Increase documents by the measured 26--62x range needed for
   individual MLP10 source fingerprints. This could reveal private/reused pieces hidden by attenuation, but is much
   more expensive than testing the large attention8 object now. Kill or defer it if pooled/shared objects already
   explain all reliable residual effects and selective manipulations do not require source-specific units.
4. **A more reliable private scoring basis.** Before abandoning private pieces, score the post-shared residual using
   richer downstream activation/logit responses rather than only 32 sparse circuit masks. Kill it if independent
   document halves still show no residual covariance or if fitted private directions fail owner-specific finite
   swaps.
5. **Exact attention factor vocabulary.** Jointly factor Q/K/Q2/K2/value behavior across heads and identify shared
   inputs or outputs by downstream interchange. This is the structurally different route closest to the user's
   original cross-head proposal. Kill it if held-out attention patterns and downstream effects do not improve over
   native-head or permutation controls, even when factors are allowed to cross heads.

## Next action

Let the already-live Stage-A run finish and score its registered predicates exactly. If A fails, inspect whether the
failure is donor-ensemble disagreement or document-half instability and immediately preregister the corresponding
donor-count or corpus-size repair. If A passes, implement only the leave-one-circuit-out shared projector first;
freeze a separate post-shared residual-power gate before any private optimization. Report the block-6 effect-space
alignment as a secondary comparison, not as a new target chosen after seeing the attention8 result.
