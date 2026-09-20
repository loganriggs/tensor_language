# Optimizer geometry and symmetric block solves — 2026-09-20 16:28 UTC

This is a targeted literature/algorithm follow-up, not a replacement for the scheduled three-hour mathematical review due around16:36.

## Paired parameterization control

The two queued full-input quartic variants represent the same initial function. On a complete small quartic, initial values, the output-weight gradient scaling law, and all other parameter gradients agreed exactly in float64. Yet one Adam step changed the represented function by167% of its small initial norm in direct coordinates versus21% in scaled coordinates; Muon changed it by29% versus4%. This confirms the comparison isolates optimizer geometry. It does not establish which parameterization will optimize better.

The finite-query output-rank bound was also checked against an exact SVD approximation. A root with r output-writing channels makes the query-output matrix rank at most r. This is a lower bound on approximation error for that finite matrix, not a certified bound on the full unsampled tensor.

## Primary literature mapped to the actual object

[O'Leary and Rust](https://www.cs.umd.edu/users/oleary/software/varpro.pdf) study separable nonlinear least squares: linear parameters can be optimized conditional on nonlinear ones. Our output matrix C is linear once input factors A,B are fixed; the exact reduced problem fits this framework. Our numerical ridge changes the exact unregularized elimination slightly. The mapping supplies an algorithmic reduction, not a guarantee of global recovery from our initializations.

[Minster, Viviano, Liu, and Ballard](https://arxiv.org/abs/2112.10855) develop QR/SVD variants of ordinary CP alternating least squares to improve numerical stability relative to normal equations. Our repeated-input polynomial is symmetrized, so ordinary unsymmetrized CP updates are not directly the same objective. We instead build exact symmetric coefficient designs for toy block solves. Native designs would be enormous, motivating an explicitly checked matrix-free normal operator with residual monitoring rather than claiming the paper's QR method was implemented at native scale.

[Hu, Iwen, Needell, and Wang](https://arxiv.org/abs/2505.14037) provide quantitative local convergence results for ordinary CP-ALS under orthogonal or incoherent decomposition assumptions. We have not verified those assumptions for native tensors, and our symmetrized parametrization can be redundant. We therefore do not import their convergence rates or recovery guarantees. Their work reinforces the need to distinguish a successful conditional solve from successful global tensor decomposition.

## Executed symmetric ALS comparison

We ran27 fits on three exactly representable quadratic teachers, with three paired starts and ALS/Adam/Muon. ALS used SVD block solves for500sweeps; gradient methods used1200steps at0.05. Timings and differing work budgets are recorded.

| Teacher | ALS error range | Adam error range | Muon error range |
|---|---:|---:|---:|
|Coordinate-sparse|0.0466–1.03%|0.0928–1.17%|0.0140–19.63%|
|Bilinear CP|0.00389–12.72%|0.208–0.757%|0.0155–2.25%|
|Sparse Tucker|0.00341–0.587%|0.0613–0.144%|0.0318–0.0346%|

All ALS objectives were monotone within tolerance. ALS was faster on these small controls, but two planted CP runs stayed at8.82% and12.72% despite exact block solves. This rules out treating accurate least-squares blocks as sufficient for recovery; it does not invalidate the conditional equations. There is no universal optimizer winner in these settings.

## Matrix-free native consequence

For the symmetric quadratic model, fix C,B and let G=CᵀC. The coefficient-Frobenius normal operator acting on A is

$$
\mathcal H(A)=\frac12\left[(G\odot BB^\top)A+
(G\odot BA^\top)B\right].
$$

For teacher factors $C_t,A_t,B_t$, the right-hand side is

$$
b=\frac12\left[((C^\top C_t)\odot BB_t^\top)A_t+
((C^\top C_t)\odot BA_t^\top)B_t\right].
$$

These formulas are derived for our symmetrized objective. They use matrices of width-by-width and width-by-input size, avoiding a design matrix with one row per tensor coefficient. Operator, right-hand side, and diagonal agree with an independently constructed dense design to at most3.78e-16 relative error. Preconditioned conjugate gradients with a small proximal ridge agrees with a dense solve to1.66e-11.

A native pilot is queued: widths128/512, two starts,30sweeps,30CGsteps per input block. It records actual residuals and rejected objective-increasing sweeps. The proximal term penalizes movement from the current block, rather than shrinking toward zero. This is a computationally feasible alternative to the failed variable-projection settings, not a proof that a low-width native solution exists.

Receipts: `QUARTIC_PARAMETERIZATION_CHECK_V1.json`, `QUADRATIC_ALS_TOYS_V1.json`, `MATRIX_FREE_QUADRATIC_CHECK_V1.json`; plan: `MATRIX_FREE_NATIVE_ALS_PLAN_V1.md`. The full-input quartic runs and this pilot remain in the managed queue behind a verified live shared job. No competing GPU process was started.
