# Resume unresolved overlapping blocks with a standard manifold solver

The full-U16-block fit (16input readers and4quadratic outputs per block) never
converged. The already executed QR-gauge repair preserves its function and
component-energy penalty to2.6e-15. This is a solver change on that same model
family and lambda=.01 objective, not evidence from a new structural family.

Use the saved MULTIOUTPUT_BLOCK_ORTHOGONAL_GAUGE_V1_INITIAL.pt after hash binding.
Each block has an orthonormal1152x16 input frame; frames from different blocks
may overlap. Each of64 symmetric16-square cores is represented by136 entries
with off-diagonal1/sqrt(2) weighting and unit norm. Its amplitude lives in its
1152-vector writer. Writers are solved exactly conditionally using the existing
full unembedding metric. Native4608products and all50304output rows enter the
objective implicitly; no data or truncated output metric.

Pymanopt2.2.1 Product(Stiefel^16, Oblique(136,64)), Polak–Ribiere CG, analytic
Torch envelope gradient, QR retraction. Initial native chunk120seconds with
saved state; end-of-chunk is not convergence. Original penalized objective and
representation stay fixed. Canonical manifold gradient threshold1e-5, with
same-objective continuation or restart if the time/step limit wins. Record
the original objective and its saved-state value before any update. A new
coordinate gradient threshold is not retrospectively the old convergence bar.

- pred_a: native initial function/objective replay<=1e-8; conditional writer
  residual<=1e-8; finite differences and independent dense toy contractions<=1e-8.
- pred_b: native final manifold gradient norm<=1e-5. Limits count as a miss.
- pred_c: at least5%relative improvement in coefficient capture over the original
 8.62302%full-U block fit, on the same lambda=.01 objective and literal capacity.

Null: redundancy repair does not materially improve fit or fails to converge.
An unconverged result cannot close this overlapping-block hypothesis. Higher
capture alone does not identify a circuit; a converged family will need stable
block functions/output groups and frozen FineWeb validation before promotion.

Price:16*16*1152 input coefficients +16*4*136 core coordinates +64*1152 writers
=377344 stored coefficients (redundant constrained coordinates are still charged).
Each of16 blocks computes16 readers once and136 unique quadratic monomials,
then4linear core combinations:2176monomials plus output mixing. U and the native
remainder/background remain charged. No claimed whole-model savings.

The differentiable adapter and CPU controls are implemented. Native wrapper,
execution binding and managed queue submission are still pending; do not report
native convergence from the toy control. No duplicate fit may start without
checking the current runner and result receipt.


## Pre-execution correction,11September04:54UTC

Prior-result search found the already completed custom manifoldCG run
MULTIOUTPUT_MANIFOLD_V1_RESULT.json and its checkpoint. This newer source
supersedes the older QR-initial cache for native optimization. Initial loss
.9145139183127093 and residual.9137153091501415 must replay; retain the old
QRbridge as a historical control. Inherited compute540.148+240.027seconds.
This is a standard-library CG/line-search continuation in the same coordinates,
not a novel coordinate repair. Native chunk remains120seconds.

Strengthen convergence before execution: Pymanopt absolute gradient tolerance
1e-7, with the original relative stationarity<=1e-4 and original maximum
absolute gradient<=1e-7 also required at the final point. Relative stationarity
is max(16*norm(bank tangent),8*norm(core tangent))/captured_fraction. Final
canonical gradient and legacy tests are both reported; a library stop cannot
replace them. The original5%capture gain bar remains unchanged.

The original plateau criterion is also retained: relative objective change over
five diagnostics (each five accepted steps, plus final point) <=1e-5. V2adapter
logs scalar diagnostics only, not large iterate arrays. All three legacy
conditions and the absolute1e-7norm are required for pred_b.
