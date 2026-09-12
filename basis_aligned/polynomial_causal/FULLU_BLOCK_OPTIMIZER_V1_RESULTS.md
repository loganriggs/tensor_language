# Direct block rotation improves coupling but fails the registered tests

12 September2026. [Registration](FULLU_BLOCK_OPTIMIZER_V1_PREREGISTRATION.md),
[primary result](FULLU_BLOCK_OPTIMIZER_V1_RESULT.json).

The exact full-output optimization completed in1113.56seconds. Numerical checks
pass (A), while separation (B), stability (C) and convergence (D) fail.

| Seed | Initial normalized cut | Final cut | Cross-block energy fraction | Final gradient norm | Stop |
|---|---:|---:|---:|---:|---|
|120423|0.912480|0.839183|41.64%|0.000956|1500step limit|
|120424|0.920960|0.840153|41.68%|0.000893|1500step limit|

The registered cut target was0.1; convergence required gradient norm at most1e-6.
Cross-seed projector overlap is0.717588, below0.9. Both proposed groups have
substantial incident energy (45.65%/54.35% and54.44%/45.56%), so the result is not
explained by quietly selecting negligible directions. Exact initial replay
errors are1.11e-16; orthogonality errors are below3.33e-14; accepted objectives
are monotone under the exact objective. Each start used2380objective evaluations.

Allowing rotations improves on the sketch-fixed basis, but this two-start
budget-limited search does not find independently operating blocks. It is
**unconverged**, not a local-optimality certificate or a global impossibility
result. The earlier planted control additionally showed a strict bad local
minimum, so convergence alone would not settle the search question either.

The prepared [normalized commutant spectral comparison](NORMALIZED_COMMUTANT_NATIVE_V1_PREREGISTRATION.md)
is now submitted after reading this terminal result. It relaxes the projector
search into a matrix-free eigenproblem and checks the resulting rounded
partition using the same exact output objective. Its Ritz values will not be
called certified lower bounds. No native behavioral experiment is warranted
for these failing block candidates yet.

The [shared-parent counterexample](NORMALIZED_COMMUTANT_RELAXATION_V1_MATH.md#independence-is-not-the-same-as-reusable-arithmetic)
also limits the broader conclusion: a cheap reusable DAG can require cross-group
products. The existing shared-parent optimizer has therefore been reused in a
[prepared producer-folded comparison](COMPOSED_SHARED_PARENT_V1_PREREGISTRATION.md),
to be considered after the spectral result. No extra block thresholds or ranks
are selected from these outcomes, and no circuits are promoted.
