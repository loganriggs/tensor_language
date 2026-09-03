# Hourly strategic review — 2026-09-03 10:21 UTC

## What a circuit result must eventually provide

1. **A computation:** what information is read, what operation combines it, what is written, and which later
   computations read that write.
2. **The right units:** merge pieces of different heads/MLPs when later computation treats them as the same variable,
   and split one head/MLP when its pieces serve different computations.
3. **Held-out and OOD prediction:** predict activations and causal effects on unseen documents, tasks, and shifted
   inputs.
4. **Extraction or sufficiency:** an executable circuit, or an explicit circuit-plus-background interface,
   reproduces the target computation or signed causal effect.
5. **Selective manipulation:** removal, swapping, or editing changes the intended behavior without damaging unrelated
   behavior, while accounting for redundancy and interactions.
6. **Composition and reuse:** shared parts behave predictably when reused by several tasks and when installed with
   task-specific parts.
7. **Stable identification:** the units survive data splits, fitting restarts, and plausible gauges, or are defined by
   downstream operational equivalence.

The program-level goal is a smaller transparent tensor program that is jointly predictive, composable, manipulable,
and simpler under literal storage, compute, edges, states, or program length.  Rank reduction, reconstruction, CE,
or variance preservation alone does not meet these circuit goals.

## What changed since 09:09

- Rung 523 showed that neither fixed-scale nor lower-learning-rate Adam-through-QR made all attention8 projector
  fits healthy.  This was an optimizer closure, not a circuit null.
- Rung 524 then failed on all 15 planted known-answer subspace problems.  The attention8 learned-projector route is
  closed without using its failure as evidence about the real circuit.
- Rung 525 tested exact MLP0 token-by-context operators.  They were stable, but their proposed token groups were
  worse than ordinary token-vector neighbors on held-out documents (`123.5%` of the raw-neighbor distance).
- Rung 526 defined token similarity by first-order effects on 32 real downstream circuits through the complete model
  suffix.  It changed `98.3%` of donors, but the selected groups were document-specific: held-out distance was
  `186.2%` of raw-neighbor distance and selection/score correlation was `0.010`.  The 30 held-out circuits and finite
  swaps correctly remained unopened.
- The 10:20 mathematical review derived an exact, gauge-invariant 20-term split of MLP0's context-only quadratic
  branch once the five source relations are fixed.  It also identified why this is not automatically a semantic
  basis: the five source variables are correlated and their partition was chosen operationally.

These failures rule out the tested token-grouping objects.  They do not license a rank sweep, and the separate
effective-rank measurement changes no circuit ledger by itself.

## Is rung 527 still the best next move?

Yes, for one frozen attempt.  MLP0's context-only branch has a replicated finite causal contribution (`0.3506` nat
FIT and `0.4177` nat SELECT Shapley-average benefit), and rung 517 already exposed a specific accounting defect:
47–52% of the branch energy sat in an unnamed constant because the FIT expectation of the quadratic term was not
assigned to source pairs.  Correcting that creates exactly 20 interpretable operations—linear effects of each source,
self-interactions, and unordered source-pair interactions—and immediately supports finite removals through the whole
model.  It directly tests within-module splitting, held-out prediction, and selective manipulation.

It must stop after one valid null.  Exact reconstruction without stable selective downstream effects is anatomical
bookkeeping, not a circuit and not a reason to tune thresholds or subdivide heads again.

## Confound audit

- **Baseline subtraction:** each circuit score must be the signed difference between a term's finite removal effect
  on registered members and its effect on the circuit's matched controls.  Raw member loss is not enough.
- **Frame mixing:** source means and quadratic expectation terms are fit on FIT only and then frozen.  SELECT and
  held-out rows never redefine the terms.
- **Nonlinear loss composition:** the experiment uses actual finite removals through final logits; it does not infer
  removal effects by adding gradients or singleton CE changes.
- **Shared token difficulty:** matched circuit controls and circuit-label permutations test whether a term merely
  tracks generally difficult tokens.
- **Leakage/post-selection:** the 32 discovery circuits choose eligible source terms; the other 30 circuit families
  remain sealed until all discovery transfer gates pass.  Thresholds and the source vocabulary are frozen first.
- **Dead interventions:** every removed term must change MLP0's write above a numerical floor, while the unmodified
  endpoint must replay exactly.
- **Precision:** compute the semantic terms and FIT expectations in float32/float64, retain a separately measured
  deployed-arithmetic remainder, and require that the newly assigned expectation collapses the old 47–52% unnamed
  energy to a small numerical fraction.
- **Noise floor:** require agreement across disjoint document halves and compare circuit concentration against
  permuted circuit labels.  A large pooled magnitude with unstable circuit direction is a null, as in rungs 519–520.

## Ranked next moves and kill evidence

1. **R527 exact context-source interactions.** Best information per model call and repairs a known causal accounting
   hole.  Kill it if exactness/liveness fails, no term has stable discovery effects, or every stable term is broad
   relative to permuted circuit controls.
2. **Finite predictive-state quotient.** Define prefix states by a battery of actual continuation interventions and
   merge only states with interchangeable held-out suffix behavior.  Kill it if the response table is unstable or
   no finite test battery predicts new continuations.  This is more expensive but changes the object after R526's
   tangent failure.
3. **Shared attention Q/K/output functions across native heads.** Identify pieces by cross-head physical interchange
   and common downstream readers.  Kill it if factor swaps do not transfer across documents or downstream modules.
4. **Move to another MLP.** Do this if R527 closes MLP0's final plausible fixed semantic split; do not respond with
   smaller source bins, more rank, or more token clustering.

R527 remains first because it is exact, cheap, causally material, and directly falsifiable.  Its preregistration and
planted algebra test start immediately.
