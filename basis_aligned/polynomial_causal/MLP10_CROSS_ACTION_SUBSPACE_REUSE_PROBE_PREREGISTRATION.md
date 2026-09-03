# Parallel probe: is the block-6 shared summary REUSED across the four score implementations (leave-one-action-out)?

**Status:** prospectively frozen after §2661 (the shared MLP10 3-dim summary is a reproducible object feeding
block-6 circuits) and §2659 (per-node reliability budget: m*=16 nodes for detectability), before any transfer
statistic is computed. CPU-only, zero forwards, zero deployed parameters. Owner: Claude parallel lane. Tests
Logan's compositional-REUSE question directly: is the one shared context-summary the SAME across the four
independently calibrated equality-score implementations, or does each have its own? Not a frontier claim (§2135
unused).

## Why leave-one-action-out (power)

Splitting the 83 material nodes into four per-action subspaces (~20 nodes each) sits right at §2659's m*=16
detectability floor, so per-action eigenvector estimates are noisy and a naive cross-action agreement test would
be attenuation-limited (the §2657 trap). Instead LEARN the shared direction from three actions (~62 nodes, well
above the floor) and TEST it on the held-out fourth via a one-dimensional projection — high power and
non-circular.

## Object (frozen, from the rung520 discovery bundle)

Bundle `mlp10_source_star_causal_quotient_rung520_bundle.pt` (`7838deca…`). Per-node per-half 32-circuit effect
reconstructed exactly as §2657–§2661 (validated: reproduces material_nodes=83, per-action [21,20,21,21]). Nodes
are indexed by (action a in 0..3, source). `M0,M1 in R^{83x32}`, circuit columns mean-centred over nodes.
Cross-half cross-covariance of a node set `X`: `S_X = (M0_X^T M1_X + M1_X^T M0_X)/2` (32x32, noise-unbiased
because the two document halves are disjoint).

For each held-out action `a`: train `S_{-a}` on the ~62 material nodes of the other three actions; `v1_{-a}` =
its top eigenvector, `V3_{-a}` = its top-3 subspace. Test on action `a`'s own `S_a` (~21 nodes):
- transferred top-direction signal `t1_a = v1_{-a}^T S_a v1_{-a}`;
- subspace-captured fraction `f_a = trace(V3_{-a}^T S_a V3_{-a})_+ / sum(positive eigenvalues of S_a)`.

Null: 200 hash-fixed permutations (seeds 9000+) of action `a`'s half-1 node pairing (E[S_a_perm]=0); q95 of the
permuted `t1` and of the permuted subspace trace. Diagnostic (not scored): `cos(v1_{-a}, pooled_v1)` where
`pooled_v1` is the all-83-node top eigenvector (the §2661 block-6 direction).

## Frozen predictions (with measured bars)

- **A — instrument.** Bundle SHA256 `7838deca6432f76af14d3ef9f363c5d783bf70490fa199ce00a7b84aa3b19a06` and
  §2658 result SHA256 `1e8ade7c9acea5cbc83c1de511aa0b5bad323fea4b681a05dc450bdc6431120b`; reconstruction
  reproduces `material_nodes==83` and per-action `[21,20,21,21]`; pooled top eigenvalue reproduces §2658's
  `0.00933` (tol 5e-4); each `V3_{-a}^T V3_{-a}=I_3` to 1e-10.

- **B — the block-6 direction TRANSFERS to every score implementation.** For ALL four held-out actions,
  `t1_a > q95` of the action's node-permutation null (the direction learned from three score implementations
  carries reliable signal in the fourth). A single action failing = not universal reuse.

- **C — the shared 3-dim subspace explains the MAJORITY of each held-out action's reliable signal.** For ALL
  four held-out actions, `f_a >= 0.50` AND the raw captured trace exceeds the permuted-null q95.

`strong_null = not (A and B and C)`.

## Reading and routes (frozen)

- A false: repair only the reconstruction clause.
- A true, B false: the shared summary does NOT transfer to at least one score implementation — reuse is
  partial; report which action(s) fail and their `cos(v1_{-a}, pooled_v1)`.
- A,B true, C false: the top DIRECTION reuses across score implementations but the fuller 3-dim object does not —
  report the 1-D reuse only.
- A,B,C true: the block-6-feeding context-summary subspace is REUSED across all four equality-score
  implementations — one shared summary serves every score variant. This is a concrete compositional-reuse result
  and it supports the premise of rung521's SHARED projector (a single shared object across the a8 cluster);
  report the per-action transfer numbers as the effect-space evidence.

Assumptions that may fail: the ~62-node LOO projector is well-powered (>m*=16) but the ~21-node test set is near
the floor, so the node-permutation null is the calibrated reference (a real null passes conservatively); the 22
sources are common across actions (matched by construction); independent half-noise requires disjoint docs
(satisfied); effect space is a lossy readout of activation space.

## Literal price

Zero forwards, zero backwards, zero deployed parameters. Four LOO eigendecompositions + four test projections +
800 permutation projections; CPU, < 3 seconds.

## Frozen inputs

- rung520 bundle SHA256: `7838deca6432f76af14d3ef9f363c5d783bf70490fa199ce00a7b84aa3b19a06`
- §2658 result SHA256: `1e8ade7c9acea5cbc83c1de511aa0b5bad323fea4b681a05dc450bdc6431120b`
