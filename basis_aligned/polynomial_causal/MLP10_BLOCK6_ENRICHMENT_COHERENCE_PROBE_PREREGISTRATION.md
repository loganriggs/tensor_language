# Parallel probe: is §2661's "feeds BLOCK-6" real enrichment, or just base rate? (self-red-team)

**Status:** prospectively frozen after §2661 reported the shared MLP10 subspace's energy concentrated on block-6
circuits, before any enrichment statistic is computed. CPU-only, zero forwards, zero deployed parameters. Owner:
Claude parallel lane. This RED-TEAMS my own §2661 headline, which both lanes now anchor on as the shared-stage
target — it must be checked before propagating. Not a frontier claim (§2135 unused).

## The concern

The 32-circuit panel contains 12 block-6 circuits (`r.6.*`) = 37.5% of the panel. §2661's "5 of the top-8
energy circuits are block-6" (62.5%) may be genuine enrichment or partly base rate. If it is base rate, the
"feeds block-6" claim is an artifact of panel composition and must be qualified.

## Object (frozen, from the rung520 discovery bundle)

Bundle `mlp10_source_star_causal_quotient_rung520_bundle.pt` (`7838deca…`). Per-node per-half 32-circuit effect
reconstructed exactly as §2657-§2661 (validated: reproduces material_nodes=83). `M0,M1 in R^{83x32}`, circuit
columns mean-centred over nodes. Cross-half cross-covariance `S=(M0^T M1 + M1^T M0)/2` (32x32, noise-unbiased);
`V3` = its top-3 eigenvectors (the §2658/§2661 shared subspace). Per-circuit subspace energy `e_c = sum_k
V3[c,k]^2` (sums to 3 over the 32 circuits). Block-6 index set `B` = the 12 circuits whose tag starts `r.6.`;
base-rate share `|B|/32 = 0.375`, base-rate energy `3*0.375 = 1.125`.

## Frozen predictions (with measured bars)

- **A — instrument.** Bundle SHA256 `7838deca6432f76af14d3ef9f363c5d783bf70490fa199ce00a7b84aa3b19a06` and
  §2658 result SHA256 `1e8ade7c9acea5cbc83c1de511aa0b5bad323fea4b681a05dc450bdc6431120b`; reproduces
  `material_nodes==83`, pooled `lambda1 ~ 0.00933` (tol 5e-4), and `|B|==12`; `V3^T V3=I_3` to 1e-10.

- **B — block-6 is ENRICHED above base rate.** Observed block-6 subspace energy `E_B = sum_{c in B} e_c` exceeds
  the 95th percentile of the null distribution of `E_{B'}` over 2000 hash-fixed random 12-circuit subsets `B'`
  (seeds 11000+). This is the falsifiable enrichment test; a null result means "feeds block-6" is base rate and
  §2661's headline is qualified to "feeds a subspace no more block-6-concentrated than chance."

- **C — the loaded block-6 circuits are COHERENT.** The mean absolute off-diagonal correlation among the block-6
  rows/cols of the noise-unbiased circuit covariance `S` (normalised to correlations) exceeds the 95th
  percentile of the same statistic over the 2000 random 12-subsets. (Do the block-6 circuits co-vary as one
  driven computation, or is the loading spread over unrelated circuits?)

`strong_null = not (A and B and C)`.

## Reading and routes (frozen)

- A false: repair only the reconstruction clause.
- A true, B false: §2661's "feeds block-6" is BASE RATE, not enrichment. Immediately qualify the §2661 ledger
  headline and the board target handed to Codex: the shared subspace is not block-6-specific beyond panel
  composition. Report the actual top-enriched circuit(s) if any single circuit is individually enriched.
- A,B true, C false: block-6 is enriched but not internally coherent — the summary feeds several block-6
  circuits that do not co-vary; report enrichment without a "coherent block-6 computation" claim.
- A,B,C true: §2661's headline stands and strengthens — the shared summary is genuinely block-6-enriched AND the
  loaded block-6 circuits co-vary, i.e. it drives a coherent block-6 computation. Report `E_B`, its null
  percentile, and the coherence margin as the validated target for rung521's shared stage.

Assumptions that may fail: the 12 block-6 circuits are not independent panel members (nested r.6.0.*/r.6.2.*
families) — the random-subset null draws from the same 32 panel so it shares that dependence structure;
effect space is a lossy readout of activation space.

## Literal price

Zero forwards, zero backwards, zero deployed parameters. One eig, one energy sum, one correlation matrix, 2000
random-subset draws x 2 statistics; CPU, < 2 seconds.

## Frozen inputs

- rung520 bundle SHA256: `7838deca6432f76af14d3ef9f363c5d783bf70490fa199ce00a7b84aa3b19a06`
- §2658 result SHA256: `1e8ade7c9acea5cbc83c1de511aa0b5bad323fea4b681a05dc450bdc6431120b`
