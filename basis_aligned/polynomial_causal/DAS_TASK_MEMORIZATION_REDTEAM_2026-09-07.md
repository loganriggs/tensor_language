# DAS task-memorization red team

## Result

The noisy worst-environment rank-one H3 optimizer is a valid negative result. It improved its
odd-row selection objective from a worst-environment loss of `3.0492` to `2.7772`, and two
independently seeded starts converged to nearly the same axis (`|cos| = .9802`). Nevertheless,
the untouched v8/v9 multi-reader score regressed from the frozen pooled-aligned axis's
`.7582` mean / `.9548` worst to `.9493` / `1.2024`. The optimized axis still beats DIM in
every reader family, so the failure is specifically a regression from the stronger pooled
baseline rather than total loss of the circuit.

The selected axis is not a small perturbation of the pooled seed. Its frozen anchor distance
`1 - cos^2 = .3683` implies `|cos| ~= .7948`. The `.9802` stability is between the two
successful optimized restarts. Thus the optimizer reproducibly finds the same attractive
task-specific direction; this is not merely restart noise.

## What this falsifies

Antithetic tangent noise at sigma `.03`, a small projector anchor, multiple causal readers,
and an odd/even lexical-row split do **not** prevent task-family memorization. The row split
holds out examples but not the generative cue/template family. KL is not the missing generic
penalty: the prior frozen tournament already found KL-only and KL+noise axes near DIM and far
behind pooled aligned.

The complement criterion remains scientifically useful as one necessary half of a causal
operator test, but it is not an identifying target by itself and does not become identifying
merely by adding same-family readers. The present optimized loss is therefore rejected as a
model-selection oracle.

## Revised target

Move the holdout boundary from rows to complete environments. The next fit should use
leave-one-cue-family-out selection: optimize on whole v1/v2 families plus one genuinely fresh
construction family, select on a disjoint construction family, and keep another family sealed.
No row from the selection family may enter gradients. The pooled-aligned axis remains the
do-nothing candidate at every checkpoint, so optimization cannot graduate by being worse than
the seed.

The primary score remains the finite causal operator across behavior, L15 H5/H1, and centered
full-vocabulary readers. Noise is retained because it improved prior frozen transfer, but no
additional KL-only arm is licensed. If complete-family validation still selects an axis that
regresses on the sealed family, abandon penalty tuning and optimize a causal-response/Hankel
basis across tasks, using downstream response tensors as the object rather than H3 activation
coordinates.

## Weight-tensor implication

The failure does not weaken the current weight results. Exact weights identify attention15 H1
and MLP shared-Q8 incidence, while activation-conditioned contractions plus finite downstream
sensitivity explain the MLP10-14 propagation chain. Those tensors should now define and audit
reader coverage: an optimized H3 direction must preserve the set of downstream weight readers
that consume the shared state, not only the task outputs used during fitting. This supplies a
task-independent falsifier for future DAS fits and links subspace discovery to actual upstream
and downstream weights.
