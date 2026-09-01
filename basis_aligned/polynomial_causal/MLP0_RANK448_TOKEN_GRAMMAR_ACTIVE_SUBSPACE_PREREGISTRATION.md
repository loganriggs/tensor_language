# MLP0 rank-448 token-grammar active-subspace screen — preregistration

Date: 2026-09-01 16:04 UTC

## Decision and scope

Rung 404 rejected the claim that the fixed rank-448 MLP0 program has a uniformly interaction-led error: `I` led
two document waves and `T` led two. The robust object is the joint token-dependent grammar `T+I`. This rung asks
whether a global rank-448 input subspace chosen from exact branch derivatives preserves that object better than the
existing normalized-input covariance projection.

This is a single-site tangent active-subspace screen. It cannot license a compressor, whole-model adoption, a new
rank, separate branch storage, a head-labelled basis, or FINAL evaluation.

## Frozen populations and source

- Source weights and program-building dtype: the pinned float32 bilin18 checkpoint used by rungs 328/403/404.
- Program-fitting rows: `.rowcache/fineweb_n192_skip11000.pt[0:24, :257]`, exactly the 24 documents used to build
  the fixed rung-328 covariance program. All 256 MLP0 positions per document enter the metric.
- Product-reference moments: reconstruct the unchanged rung-401 FIT reference from the frozen 96 FIT documents.
- Evaluation: rung-404's exact 384-source-document population, one chunk per source, four contiguous 96-document
  waves, positions `[64:256)`, for 73,728 scored positions total.
- Evaluation model and deployed arithmetic: the unchanged rung-401 BF16 model and physical BF16 MLP0 replacement.
- FINAL remains unopened.

## Mathematical object

Let `L,R` have shape `4608 x 1152` and `D` have shape `1152 x 4608`. Let centered token and attention writes be
`t=e-mu_e`, `c=a-mu_a`, and let `m=mu_e+mu_a`. Ignoring only branch constants, the exact output-space functions are

`T(t) = D[(Lt)*(Rm) + (Lm)*(Rt) + (Lt)*(Rt)]`,

`I(t,c) = D[(Lt)*(Rc) + (Lc)*(Rt)]`,

where `*` is coordinatewise multiplication. Their exact input Jacobians are

`J_T(t)   = D diag(Rm+Rt)L + D diag(Lm+Lt)R`,

`J_It(c)  = D diag(Rc)L    + D diag(Lc)R`,

`J_Ic(t)  = D diag(Rt)L    + D diag(Lt)R`.

For two deterministic seed-405 Rademacher output probes `q` per fitting position, compute input gradients
`g_T=J_T^T q`, `g_It=J_It^T q`, and `g_Ic=J_Ic^T q`. Their empirical Gram matrices are

`M_T = mean(g_T g_T^T)` and `M_I = mean(g_It g_It^T + g_Ic g_Ic^T)`.

Let `Sigma` be the same normalized-MLP-input covariance and eigenvalue flooring used by rung 328. In its whitening
frame define

`K_T  = Sigma^(1/2) M_T Sigma^(1/2)`,

`K_I  = Sigma^(1/2) M_I Sigma^(1/2)`,

`K_TI = K_T/trace(K_T) + K_I/trace(K_I)`.

The T-only, I-only, and joint bases are the top 448 eigenvectors `U` of these three matrices. Every basis produces
the same legal shared-input program

`encoder = U^T Sigma^(-1/2)`, `decoder = Sigma^(1/2) U`,

`L_small=L decoder`, `R_small=R decoder`.

The physical computation is still `D[(L_small encoder z)*(R_small encoder z)] + bias`. It stores exactly
`9,954,432` values and saves `5,971,968` from native MLP0. A seed-1405 Haar rank-448 basis in the same whitening
frame is the negative control. The unchanged rung-328 covariance-RRR p448 is the positive baseline.

This construction is invariant to hidden-unit permutation/rescaling gauges because its Jacobians are taken from the
complete output function. Attention enters through the output-projected residual write `c`, which is invariant to a
within-head value/output change of basis. It does not assume that head labels are canonical.

## Measurements

For each program and each of four waves:

1. physically replace MLP0 and measure CE added above native;
2. reconstruct exact `T` and `I` outputs under the frozen product reference;
3. report branch relative MSE `sum||branch_program-branch_native||^2 / sum||branch_native||^2`;
4. record endpoint identity, state replay, and live forward/hook counts.

Pool waves by equal document/token weighting. No basis, weight, threshold, or population changes after observing an
evaluation result.

## Frozen predictions

### A — instrument, identity, and price

- source checkpoint, row hashes, disjointness, population, four waves, rank, shapes, and literal price are exact;
- all metric entries/eigenvalues are finite, symmetry relative error is at most `1e-6`, and basis orthogonality
  max error is at most `1e-5`;
- the rebuilt covariance baseline reproduces rung 404's pooled and per-wave compact CE damage within `1e-6`;
- native state replay and every compact endpoint have max absolute error `0`, all arms have live calls, and FINAL is
  unopened.

### B — the derivative metrics are branch-specific

- T-only pooled T relative MSE is at most `0.95` times covariance-baseline T relative MSE;
- I-only pooled I relative MSE is at most `0.95` times covariance-baseline I relative MSE;
- the joint basis's geometric mean of pooled T and I relative MSE is at most `0.95` times the covariance baseline
  and at most `0.90` times the random-basis control.

### C — joint T+I weighting improves physical prediction robustly

- pooled joint-basis CE damage is at most `0.85` times covariance-baseline damage;
- joint improves CE damage by at least `0.0002 nat` in at least three of four waves;
- no wave regresses by more than `0.001 nat` against the covariance baseline.

### D — a global joint basis is competitive with specialists and balanced

- pooled joint CE damage is no more than `0.0005 nat` above the better of T-only and I-only;
- joint T and joint I relative MSE are each at most `1.05` times their covariance-baseline values;
- pooled joint CE damage remains in `[0, 0.020] nat`.

## Strong null

The strong null fires if A fails, pooled joint improvement over covariance is below `0.0002 nat`, joint loses to
covariance in at least three waves, or neither specialist improves its own exact branch relative MSE by at least 2%.

## Decision

- A+B+C+D and no null: confirm the fixed joint basis on the established whole-model census/certificate/intervention
  gate before any adoption language.
- A+B with C or D failure: derivative geometry is real but not a predictive compressor; do not tune thresholds.
- Strong null: close first-order active-subspace weighting. Compare direct nonlinear T+I function fitting against a
  document-conditional state hypothesis; do not return to interaction-only fitting or head-label sparsity.
