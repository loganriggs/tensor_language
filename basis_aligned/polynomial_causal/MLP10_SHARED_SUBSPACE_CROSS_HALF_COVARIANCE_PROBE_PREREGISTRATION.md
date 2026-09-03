# Parallel probe: does a NOISE-UNBIASED shared circuit-effect subspace exist at current N?

**Status:** prospectively frozen after §2657 found per-node source-star fingerprints are cross-half noise
(rho=0.016 < permutation null q95 0.077), before any cross-covariance eigenvalue is computed. CPU-only, zero
model forwards, zero deployed parameters. Owner: Claude parallel lane. Three-hourly mathematical review move #1
(2026-09-03 ~04:30). Sign convention §2135 (CE added above native, lower better) — not used here (this is an
effect-covariance statistic, not a frontier claim).

## The mathematics

§2657 correlated each node's 32-circuit fingerprint with ITSELF across document halves and got ~0. Classical
test theory: an observed cross-node cosine is attenuated by the geometric mean of the two fingerprints'
reliabilities, so at reliability ~0.016 no grouping test can detect true grouping regardless of truth — the
R506–R520 grouping nulls are attenuation-bounded. The correct estimator does not correlate raw fingerprints; it
estimates the SHARED signal subspace by pooling nodes, using the fact that the two document halves
(discovery docs 500:624 and 624:748, disjoint, no batch straddle at the 624 bound) have INDEPENDENT sampling
noise. For the node×circuit effect matrices `M0, M1 in R^{83x32}` (material source-star nodes, columns
mean-centred over nodes),

`E[ M0^T M1 ] = Sigma_signal`  (32x32),

because the cross term `E[noise0^T noise1] = 0` under independence. Thus the eigenvalues of the symmetrized
cross-half cross-covariance `S = (M0^T M1 + M1^T M0)/2` are NOISE-UNBIASED estimates of shared circuit-effect
variance: positive eigenvalues are reliable shared directions; noise contributes zero-mean scatter, not a
positive bias. Pooling 83 nodes estimates `Sigma_signal` far better than any single fingerprint (whose
reliability §2657 measured at ~0). This is the cross-half analog of a split-half reliability spectrum and the
signal-subspace estimator behind noise-corrected CCA / reliability-based dimensionality (Cook sufficient
dimension; classical generalizability theory).

## Object (frozen, from the rung520 discovery bundle)

Bundle `mlp10_source_star_causal_quotient_rung520_bundle.pt` (`7838deca…`). Per-node per-half circuit effect
reconstructed exactly as in §2657 (validated: reproduces `material_nodes = 83/88`):
`eff[a,s,h,c] = (memMean[a,s,h,c]-memMean[a,0,h,c]) - (ctrlMean[a,s,h,c]-ctrlMean[a,0,h,c])`, member = mc index
0, control = mc index 1, half axis 0, arm 0 = intact, sources = arms 1..22. Material mask = pooled circuit RMS
>= .0005 nat AND pooled four-task norm (task cols 1..4) >= .00025 nat. `M0 = eff[...,0,:]`, `M1 = eff[...,1,:]`
restricted to the 83 material nodes, each of the 32 circuit columns mean-centred over those 83 nodes.

`S = (M0^T M1 + M1^T M0)/2` (32x32 symmetric). Eigenvalues `lambda_1 >= ... >= lambda_32` (signed).

Null: 200 hash-fixed permutations (seeds 5090..5289) of the 83 half-1 rows (breaking node correspondence);
recompute `S_perm` and its eigenvalues. This is the "any two node-sets over the same circuits" null; under it
`E[S_perm]=0`.

## Frozen predictions (with measured bars)

- **A — instrument exactness.** Bundle SHA256 `7838deca6432f76af14d3ef9f363c5d783bf70490fa199ce00a7b84aa3b19a06`
  and result SHA256 `1c8de74a90ca8eac167274b7fc6b84f6ed3634d5c0baf679d1d457aaf39b2a3b`; reconstruction reproduces
  `material_nodes == 83`; `S` is symmetric to `< 1e-12` relative.

- **B — a reliable shared circuit-effect direction EXISTS (pooled).** `lambda_1(S) > q95` of the permutation
  null's top eigenvalue. (If true, a reusable shared direction is estimable at current N even though per-node
  reliability is ~0 — the estimator, not the data, was the §2657 bottleneck.)

- **C — the reusable subspace is at least 2-DIMENSIONAL.** The number of real eigenvalues of `S` exceeding the
  permutation null's top-eigenvalue q95 is `>= 2` — a genuine low-dim reusable decomposition, not a single
  shared mode.

`strong_null = not (A and B and C)`.

## Reading and routes (frozen)

- A false: repair only the reconstruction clause.
- A true, B false: no reliable shared circuit-effect structure survives even when pooling 83 nodes with a
  noise-unbiased estimator. This REFINES §2657: the grouping absence is not merely per-node power — the shared
  subspace itself is negligible at this N/granularity. Only raising N (or a different object) can help; report
  to Codex as evidence that DAS at this granularity is unlicensed until N rises.
- A,B true, C false: exactly ONE reliable shared direction (rank-1 reuse), consistent with the §2649/§2652
  rank-1 context summary. The reusable "decomposition" is a single shared circuit-effect direction; DAS/reuse
  should target that 1-D object, and multi-component decomposition claims are unlicensed.
- A,B,C true: a >=2-D reliable shared circuit-effect subspace exists and is estimable NOW. Hand Codex the
  eigen-subspace basis (the reusable circuit-effect dictionary) as the object for a GPU DAS + physical-swap
  reuse rung — the pooled subspace, not raw per-node fingerprints, is where grouping should be tested.

Assumptions that may fail: (1) the 83 nodes are 4 actions x 22 sources, not independent samples — action
correlation could inflate apparent structure; the node-permutation null controls it only partially, so a
within-action-permutation null is reported as a secondary diagnostic. (2) Signal genuinely low-rank — if not,
no pooled gain. (3) Half noise independence requires disjoint docs — satisfied (500:624 vs 624:748).

## Literal price

Zero model forwards, zero backwards, zero deployed parameters. One 32x32 cross-covariance, one eig, 200
permutation eigs; CPU, < 1 second.

## Frozen inputs

- rung520 bundle SHA256: `7838deca6432f76af14d3ef9f363c5d783bf70490fa199ce00a7b84aa3b19a06`
- rung520 result SHA256: `1c8de74a90ca8eac167274b7fc6b84f6ed3634d5c0baf679d1d457aaf39b2a3b`
