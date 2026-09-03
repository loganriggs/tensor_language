# Parallel probe: is there reliable circuit-effect structure BEYOND the §2658 shared 3-dim, and is it private?

**Status:** prospectively frozen after §2658 (reliable ~3-dim SOURCE-SHARED subspace) and §2659 (budget), before
any residual eigenvalue is computed. CPU-only, zero forwards, zero deployed parameters. Owner: Claude parallel
lane. A red-team PREVIEW of Codex's rung521 shared-first/private-residual DAS: it answers, in circuit-EFFECT
space and before any GPU spend, whether the private-residual stage has any reliable target. Not a frontier claim
(§2135 unused).

## The question

§2658 found a reliable ~3-dim shared circuit-effect subspace whose top mode is source-shared (fails the
within-action null). Codex's rung521 fits an activation-space shared rank-4 projector PLUS orthogonal private
rank-4 projectors and critiques §2658 as unable to prove sources collinear or to license the private stage. This
probe tests the effect-space precondition: after removing the shared subspace, does reliable structure remain
(pred_b), and is it source-specific/private (pred_c)? A null means the private stage has no effect-space target
and the shared subspace is the whole reliable story; a positive names the sources carrying private structure.

## Object (frozen, from the rung520 discovery bundle)

Bundle `mlp10_source_star_causal_quotient_rung520_bundle.pt` (`7838deca…`). Per-node per-half 32-circuit effect
reconstructed exactly as §2657/§2658/§2659 (validated: reproduces material_nodes=83). `M0,M1 in R^{83x32}`, each
circuit column mean-centred over nodes. Shared subspace `U` = the top `k=3` eigenvectors of the noise-unbiased
cross-half cross-covariance `S=(M0^T M1 + M1^T M0)/2` (k=3 frozen from §2658's `n_real_eigs_above_null_q95`).
Residual projection onto the orthogonal complement: `M0r = M0 (I - U U^T)`, `M1r = M1 (I - U U^T)`; residual
cross-covariance `Sr=(M0r^T M1r + M1r^T M0r)/2`, top eigenvalue `lambda1_r`.

Nulls (200 hash-fixed perms, seeds 7000+), computed ON THE PROJECTED RESIDUALS so the projection is accounted
for: node-permutation null (shuffle M1r rows) and within-action null (shuffle sources within each of the 4
action blocks). q95 of each null's top eigenvalue.

## Frozen predictions (with measured bars)

- **A — instrument.** Bundle SHA256 `7838deca6432f76af14d3ef9f363c5d783bf70490fa199ce00a7b84aa3b19a06` and
  §2658 result SHA256 `1e8ade7c9acea5cbc83c1de511aa0b5bad323fea4b681a05dc450bdc6431120b`; reconstruction
  reproduces `material_nodes==83`; the shared subspace reproduces §2658's `lambda1 ~ 0.00933` (tol 5e-4) so U is
  the same object; `U^T U = I_3` to 1e-10 and the residual has zero projection onto U (`||M0r U|| < 1e-10`).

- **B — reliable structure exists BEYOND the shared 3-dim.** `lambda1_r > q95` of the residual node-permutation
  null. (If false, the shared ~3-dim captures ALL reliable circuit-effect structure; rung521's private-residual
  stage has no effect-space target.)

- **C — the residual structure is SOURCE-SPECIFIC (private).** `lambda1_r > q95` of the residual within-action
  null. (If false, any residual structure is still source-shared, not private — the "private" label is
  unlicensed in effect space.)

`strong_null = not (A and B and C)`.

## Reading and routes (frozen)

- A false: repair only the reconstruction/subspace-pin clause.
- A true, B false: the §2658 shared 3-dim is the entire reliable circuit-effect story at current N. Report to
  Codex: the private-residual DAS stage is unlikely to find reliable effect-space structure without raising N
  (§2659's 26–62x); the shared subspace is the licensed target.
- A,B true, C false: reliable residual structure exists but is still source-shared (additional shared modes, not
  private). Rung521's PRIVATE stage still lacks a target; report the extra shared dimensionality instead.
- A,B,C true: reliable, source-specific (private) circuit-effect structure exists beyond the shared subspace.
  Report which sources load on lambda1_r (the residual top mode's per-source energy) as the priority list for
  rung521's private projectors.

Assumptions that may fail: U is estimated on the same data (projection can bias lambda1_r downward), but the
permutation nulls are computed on the SAME projected residuals so the test is calibrated; the 83 nodes are 4
actions x 22 sources (action correlation), handled by the within-action null; independent half-noise requires
disjoint docs (satisfied, 500:624 / 624:748).

## Literal price

Zero forwards, zero backwards, zero deployed parameters. One eig for U, one projection, one residual eig, 400
permutation eigs; CPU, < 2 seconds.

## Frozen inputs

- rung520 bundle SHA256: `7838deca6432f76af14d3ef9f363c5d783bf70490fa199ce00a7b84aa3b19a06`
- §2658 result SHA256: `1e8ade7c9acea5cbc83c1de511aa0b5bad323fea4b681a05dc450bdc6431120b`
