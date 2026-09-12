# Exact input-block rotation: registered native comparison

12 September2026. The [previous block search](FULLU_INPUT_BLOCKS_V1_RESULTS.md)
was restricted to sketch-eigenvector coordinates. This run rotates those same
half-dimensional projectors using the exact full-output objective. No data or
output-rank selection is used. It remains an orthogonal two-block hypothesis.

Use both frozen producer projectors from `FULLU_INPUT_BLOCKS_V1_PROJECTORS.pt`,
seeds120423/120424. Reconstruct the same native producer metric and full centered-U
writer Gram. Optimize normalized cut by QR-retracted Grassmann gradient descent,
Armijo coefficient0.0001, initial step100, growth1.5,24backtracks. Each seed has
at most1500accepted steps or600seconds; reaching either is not convergence.
Convergence requires the horizontal reader-gradient Frobenius norm at most1e-6.
There are no redundant scale coordinates in the orthonormal reader representation.
Preserve actual gradient, objective history and stop reason. Total alarm1300seconds.

- A: initial exact cuts replay within1e-8, all accepted objectives nonincreasing
  within1e-12, orthogonality error below1e-8, finite results.
- B: both exact final normalized cuts at most0.1, incident fractions in[0.1,0.9],
  and at least20% below the original matched random-basis controls.
- C: cross-seed projector overlap at least0.9 allowing block exchange.
- D: both starts satisfy the stated gradient convergence condition.

These B/C bars are unchanged from the failed screen. A passing local numerical
optimization does not prove globally recovered blocks. Native extraction, joint
edits, held-out/OOD prediction and explicit implementation pricing remain required.
Background and producer weights are not free; a projector is not a deployed circuit.

## Executed limitation check

The CPU planted4+4block family is recovered from two of three random starts.
All three converge, but the other start stops at normalized cut0.235588. The
16-dimensional local-chart Hessian there has eigenvalues0.0450–1.1775, supporting
a strict local minimum despite exact planted blocks elsewhere. Dense null starts
converge at cuts0.2166–0.2404 and do not produce a false exact decomposition.
See `FULLU_BLOCK_OPTIMIZER_V1_CONTROL.json` (all-start recovery prediction **fails**)
and `FULLU_BLOCK_OPTIMIZER_V1_CURVATURE.json`.

The native comparison is still informative about the previous fixed-basis
restriction, because every accepted step improves the exact objective. Its null
is only failure of this two-start local search to find the registered blocks.
Do not claim structure is absent even if D passes. A native negative needs this
optimization limitation carried forward, and a materially different discovery
assumption or global/lower-bound argument before closing the broader direction.
