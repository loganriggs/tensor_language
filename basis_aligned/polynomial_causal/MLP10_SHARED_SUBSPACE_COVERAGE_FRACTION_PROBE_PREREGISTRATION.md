# Parallel probe: what fraction of MLP10's total reliable causal footprint is the shared 3-dim subspace?

**Status:** prospectively frozen after §2658 (reliable ~3-dim shared subspace, dimensionality 3) and §2660 (no
reliable residual), before any coverage fraction is computed. CPU-only, zero forwards, zero deployed parameters.
Owner: Claude parallel lane. Advances the coverage-credit named gap with a concrete number. Not a frontier or
explained-fraction (certificate) claim — this is effect-variance COVERAGE, a distinct metric proposed as a
coverage-credit input; §2135 convention unused.

## The question

§2658 established the shared subspace is 3-dimensional (3 eigenvalues beat the node-permutation null); §2660
showed no reliable residual beyond it. Neither computed HOW MUCH of MLP10's total reliable circuit-effect
variance the 3-dim captures. This probe computes that fraction with a bootstrap CI, giving a quotable
characterization and a coverage-credit input.

## Object (frozen, from the rung520 discovery bundle)

Bundle `mlp10_source_star_causal_quotient_rung520_bundle.pt` (`7838deca…`). Per-node per-half 32-circuit effect
reconstructed exactly as §2657-§2661 (validated: reproduces material_nodes=83). `M0,M1 in R^{83x32}`, circuit
columns mean-centred over nodes. Noise-unbiased cross-half cross-covariance `S=(M0^T M1 + M1^T M0)/2` (32x32),
eigenvalues `w_1>=...>=w_32` (signed). Total reliable variance = `T = sum_i max(w_i,0)` (positive eigenvalues
only, since noise contributes zero-mean scatter and negatives are noise). Top-3 captured = `C3 = sum of the 3
largest positive eigenvalues`. Coverage fraction `f = C3 / T`.

Node bootstrap: 400 hash-fixed resamples (seeds 12000+) of the 83 nodes WITH replacement; recompute `S_boot` and
`f_boot`; report the 2.5/50/97.5 percentiles. Pure-noise baseline: 400 node-PERMUTATION draws (shuffle M1 rows,
E[S]=0) give `f_null`; its q95 is the fraction a noise matrix's top-3 captures of its own positive mass.

## Frozen predictions (with measured bars)

- **A — instrument.** Bundle SHA256 `7838deca6432f76af14d3ef9f363c5d783bf70490fa199ce00a7b84aa3b19a06` and
  §2658 result SHA256 `1e8ade7c9acea5cbc83c1de511aa0b5bad323fea4b681a05dc450bdc6431120b`; reproduces
  `material_nodes==83` and pooled `w_1 ~ 0.00933` (tol 5e-4); `T > 0`.

- **B — the shared subspace captures the MAJORITY of reliable variance.** `f >= 0.50` at the point estimate,
  and the bootstrap 2.5th percentile `f_lo >= 0.40`. (The exact `f` and its 95% CI are the coverage-credit
  deliverable regardless of the pass/fail; the 0.50 bar is the principled majority threshold, not tuned.)

- **C — the coverage is signal, not a noise artifact.** `f` exceeds the node-permutation pure-noise baseline
  `f_null` q95 (a 32-dim noise matrix's top-3 already captures a nonzero fraction of its positive mass; real `f`
  must beat that to mean anything).

`strong_null = not (A and B and C)`.

## Reading and routes (frozen)

- A false: repair only the reconstruction clause.
- A true, B false: the 3-dim shared subspace captures a MINORITY of reliable variance — MLP10's reliable
  footprint is higher-dimensional than the §2658 top-3; report `f` and the effective reliable rank, and revise
  the "one low-dim summary" framing.
- A,B true, C false: `f` is not distinguishable from what noise gives — the coverage number is not
  interpretable; report it as such.
- A,B,C true: the shared 3-dim captures the majority of MLP10's reliable circuit-effect variance and beats the
  noise baseline. Combined with §2660 (no reliable residual), state precisely: at current N, MLP10's reliable
  causal footprint IS one low-dim source-shared summary capturing `f` of the reliable variance. Hand `f` + CI to
  the coverage-credit accounting as the shared-subspace's coverage number.

Assumptions that may fail: positive-eigenvalue truncation of a symmetric cross-cov is a standard signal estimate
but slightly upward-biases `T` from noise leakage (the noise baseline pred_c controls the interpretation);
the 83 nodes are 4 actions x 22 sources (action correlation, noted throughout); effect space is a lossy readout.

## Literal price

Zero forwards, zero backwards, zero deployed parameters. One eig + 800 resample/permutation eigs; CPU, < 2 s.

## Frozen inputs

- rung520 bundle SHA256: `7838deca6432f76af14d3ef9f363c5d783bf70490fa199ce00a7b84aa3b19a06`
- §2658 result SHA256: `1e8ade7c9acea5cbc83c1de511aa0b5bad323fea4b681a05dc450bdc6431120b`
