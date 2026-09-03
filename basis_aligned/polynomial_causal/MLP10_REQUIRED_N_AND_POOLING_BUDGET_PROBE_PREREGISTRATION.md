# Parallel probe: the document/pooling budget that turns the §2657 null into a testable object

**Status:** prospectively frozen after §2657 (per-node cross-half reliability rho0=0.016) and §2658 (pooling 83
nodes recovers a reliable ~3-dim shared subspace), before any Spearman–Brown multiplier or node-subsample curve
is computed. Math-review move #2 (2026-09-03 0430). CPU-only, zero forwards, zero deployed parameters. Owner:
Claude parallel lane. Turns §2657's power-boundedness into a concrete, actionable budget for Codex's power-gated
rung521 and any higher-N grouping instrument. Not a frontier or compression claim (§2135 convention unused).

## The mathematics

§2657's per-node cross-half correlation IS a split-half reliability rho0. Two established results convert it to
a budget. (1) Spearman–Brown: multiplying the per-node document count by k scales reliability as
`rho_k = k*rho0 / (1 + (k-1)*rho0)`; inverting, reaching a target reliability rho* needs
`k = rho*(1-rho0) / (rho0(1-rho*))`. (2) Pooling m units with independent noise raises the reliability of the
shared-signal estimate like averaging — the cross-half cross-covariance eigenvalue (§2658) becomes detectable
once enough nodes are pooled. This probe measures both levers and checks that §2658's detectability is genuinely
a pooling effect (a falsifiable self-consistency of the independent-half-noise model), not an artifact.

## Object (frozen, from the rung520 discovery bundle)

Bundle `mlp10_source_star_causal_quotient_rung520_bundle.pt` (`7838deca…`). Per-node per-half 32-circuit effect
reconstructed exactly as in §2657/§2658 (validated: reproduces `material_nodes=83`). `M0,M1 in R^{83x32}`, each
circuit column mean-centred over nodes. Per-node reliability `rho0[i] = corr(M0[i,:], M1[i,:])`; `rho0_med` its
median over the 83 nodes (must reproduce §2657's 0.016). Cross-half cross-covariance `S(subset)=(M0s^T M1s +
M1s^T M0s)/2` over any node subset; `lambda1(S)` its top eigenvalue; node-permutation null q95 as in §2658.

## Frozen predictions (with measured bars)

- **A — instrument.** Bundle SHA256 `7838deca6432f76af14d3ef9f363c5d783bf70490fa199ce00a7b84aa3b19a06`,
  §2657 result SHA256 `1bdb425e4da3f85e8da31e701b56dbf51191654f10c3a52c5f58108992532b0d`, §2658 result SHA256
  `1e8ade7c9acea5cbc83c1de511aa0b5bad323fea4b681a05dc450bdc6431120b`; reconstruction reproduces
  `material_nodes==83` and `|rho0_med - 0.016| <= 0.003` (CUDA-wobble tol).

- **B — a single per-circuit fingerprint is an order of magnitude underpowered.** The Spearman–Brown document
  multiplier for the median node to reach reliability `rho*=0.5` is `>= 10x`. The exact multipliers for
  `rho* in {0.3, 0.5, 0.8}` are reported as the budget (not thresholded beyond the 10x floor at 0.5).

- **C — pooling is the cheaper lever and explains §2658.** Via node subsampling (m in {8,16,32,64,83}, 40
  hash-fixed draws each, seeds 6000+), the mean `lambda1` crosses its mean node-permutation null q95 at some
  `m* <= 83`, AND the detectability ratio `mean lambda1 / mean null_q95` is strictly larger at m=83 than at
  m=8. This confirms §2658's reliable subspace is a genuine pooling effect (falsifiable: a flat or decreasing
  ratio, or no crossing by m=83, would mean the independent-noise pooling model is wrong and the required-N
  numbers are untrustworthy).

`strong_null = not (A and B and C)`.

## Reading and routes (frozen)

- A false: repair only the reconstruction/pin clause.
- A true, B false: single fingerprints are not badly underpowered — a small document bump suffices; report the
  (small) multiplier and drop the "raise N a lot" recommendation.
- A,B true, C false: the independent-half-noise pooling model is not self-consistent with §2658 — the required-N
  arithmetic is untrustworthy; flag §2657/§2658 estimation framework for re-examination before Codex relies on it.
- A,B,C true: hand Codex TWO concrete levers for testing finer (source-specific) structure — raise per-node
  documents ~`k(0.5)`x, OR pool `>= m*` nodes (the free lever §2658 already used). Directly parameterises
  rung521's fail-closed power gate and any higher-N grouping rung.

Assumptions that may fail: Spearman–Brown assumes homogeneous added-document noise (iid corpus draws); the
pooling model assumes independent half-noise (satisfied: disjoint docs 500:624 / 624:748); node subsamples treat
the 83 action-by-source nodes as exchangeable (action correlation noted in §2658; the subsample draws mix
actions, so m* is an upper bound on the truly-independent node count needed).

## Literal price

Zero forwards, zero backwards, zero deployed parameters. One reliability vector, closed-form Spearman–Brown, and
5 subsample sizes x 40 draws x (1 eig + node-perm null); CPU, < 2 seconds.

## Frozen inputs

- rung520 bundle SHA256: `7838deca6432f76af14d3ef9f363c5d783bf70490fa199ce00a7b84aa3b19a06`
- §2657 result SHA256: `1bdb425e4da3f85e8da31e701b56dbf51191654f10c3a52c5f58108992532b0d`
- §2658 result SHA256: `1e8ade7c9acea5cbc83c1de511aa0b5bad323fea4b681a05dc450bdc6431120b`
