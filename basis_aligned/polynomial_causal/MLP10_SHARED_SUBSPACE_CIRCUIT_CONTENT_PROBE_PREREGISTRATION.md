# Parallel probe: WHAT does the §2658 shared 3-dim subspace compute — which circuits does it feed, reproducibly?

**Status:** prospectively frozen after §2658 (reliable ~3-dim source-shared subspace exists) and §2660 (no
reliable residual beyond it), before any eigenvector loading or bootstrap is computed. CPU-only, zero forwards,
zero deployed parameters. Owner: Claude parallel lane. This is the first CONSTRUCTIVE characterization of the
recent arc's one positive object: it names the downstream circuits the shared MLP10 context-summary feeds and
checks the naming is reproducible. Hands Codex a labeled target for rung521's shared stage. Not a frontier claim
(§2135 unused).

## The question

§2658/§2660 established that all reliable MLP10 circuit-effect structure at current N lives in a source-shared
~3-dim subspace of the 32-circuit effect space. This probe asks WHAT that subspace is: the eigenvector loadings
of the noise-unbiased cross-half cross-covariance name which of the 32 known downstream circuits the shared
summary perturbs, and a node-bootstrap tests whether that naming is reproducible rather than a single-sample
accident.

## Object (frozen, from the rung520 discovery bundle)

Bundle `mlp10_source_star_causal_quotient_rung520_bundle.pt` (`7838deca…`). Per-node per-half 32-circuit effect
`M0,M1 in R^{83x32}` reconstructed exactly as §2657–§2660 (validated: reproduces material_nodes=83), circuit
columns mean-centred over nodes. Cross-half cross-covariance `S=(M0^T M1 + M1^T M0)/2` (32x32). `v1` = top
eigenvector; `V3` = top-3 eigenvectors (the reliable shared subspace, §2658). "Loading" of circuit c on a mode
= that mode's entry at c; the 32 circuits are the fixed `circuit_tags` axis.

Node bootstrap: 200 hash-fixed resamples (seeds 8000+) of the 83 nodes WITH replacement; recompute `S_boot`,
its top eigenvector `v1_boot` (sign-aligned to `v1`) and top-3 subspace `V3_boot`. Loading stability =
median over bootstraps of `|cos(v1_boot, v1)|`. Subspace stability = median over bootstraps of the mean
top-3 principal-angle cosine between `V3_boot` and `V3` (mean of the 3 singular values of `V3_boot^T V3`).

Nulls: random unit vectors in R^32 for the loading (200, same seeds); random 3-subspaces of R^32 for the
subspace (200). q95 of each null statistic.

## Frozen predictions (with measured bars)

- **A — instrument.** Bundle SHA256 `7838deca6432f76af14d3ef9f363c5d783bf70490fa199ce00a7b84aa3b19a06` and
  §2658 result SHA256 `1e8ade7c9acea5cbc83c1de511aa0b5bad323fea4b681a05dc450bdc6431120b`; reconstruction
  reproduces `material_nodes==83`; `S` reproduces §2658's `lambda1 ~ 0.00933` (tol 5e-4); `V3^T V3 = I_3` to
  1e-10.

- **B — the top shared mode names a REPRODUCIBLE circuit combination.** Node-bootstrap loading stability
  `median |cos(v1_boot, v1)| >= 0.70`, exceeding the random-unit-vector null q95 (a fixed 32-vector's cosine
  with a random unit vector has q95 ~ 0.29).

- **C — the full 3-dim shared subspace is REPRODUCIBLE as a basis.** Node-bootstrap subspace stability
  `median mean-principal-angle-cosine(V3_boot, V3) >= 0.70`, exceeding the random-3-subspace null q95.

`strong_null = not (A and B and C)`.

## Reading and routes (frozen)

- A false: repair only the reconstruction/subspace-pin clause.
- A true, B false: even the leading shared direction is not a reproducible circuit combination — the ~3-dim is
  a stable SUBSPACE but its axes are not individually named (report only the subspace-level circuit energy).
- A,B true, C false: the leading mode is named but modes 2--3 rotate under resampling (close eigenvalues) —
  report the top mode's circuits and the subspace-level energy, not a 3-axis basis.
- A,B,C true: the shared summary is a reproducible ~3-dim object with named circuit content. Report, per mode,
  the top circuits by |loading|, and the circuits with the highest total energy across the 3-dim subspace — the
  "which downstream circuits the shared MLP10 context-summary feeds" list — as the labeled target for rung521's
  shared stage and the seed for a source->circuit reuse map.

Assumptions that may fail: node bootstrap treats the 83 action-by-source nodes as exchangeable (action
correlation, §2658 — the bootstrap mixes actions so stability is if anything conservative); close eigenvalues
2--3 can rotate (handled by the subspace-level pred_c); independent half-noise requires disjoint docs
(satisfied). The circuit loadings are of effect-space coordinates, a lossy readout of activation space.

## Literal price

Zero forwards, zero backwards, zero deployed parameters. One eig, 200 node-bootstrap eigs, 400 null draws;
CPU, < 3 seconds.

## Frozen inputs

- rung520 bundle SHA256: `7838deca6432f76af14d3ef9f363c5d783bf70490fa199ce00a7b84aa3b19a06`
- §2658 result SHA256: `1e8ade7c9acea5cbc83c1de511aa0b5bad323fea4b681a05dc450bdc6431120b`
