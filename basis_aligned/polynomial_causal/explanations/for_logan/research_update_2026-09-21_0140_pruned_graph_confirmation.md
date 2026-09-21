# Confirmation of the 512-product graph — 2026-09-21 01:40 UTC

The graph simplified in [the product-removal experiment](research_update_2026-09-21_0136_product_removal.md) passed its registered checks on another new document panel. This confirms a useful cost–fidelity tradeoff for the full selected folded contribution. It does not establish semantic circuit identity.

## Frozen comparison

We froze both the original corrected 1,024-product graph and the smaller 512-product graph before evaluation. The panel contains 32 new FineWeb documents from skip11000 indices 32:64 and 16 additional top-level Python files from this repository. Source paths and exact token prefixes were checked against prior panels. Code is related local source, not broad external OOD; pretraining overlap is unknown.

| Metric on new documents | Original 1,024-product graph | Smaller 512-product graph |
|---|---:|---:|
| Weight coefficients | 2,671,616 | 1,291,264 |
| FineWeb replacement CE added, nats/token | 0.01178 | 0.01253 |
| Code replacement CE added, nats/token | 0.02561 | 0.02719 |
| FineWeb removal-effect error | 22.09% | 22.64% |
| Code removal-effect error | 15.30% | 15.81% |
| FineWeb same-token swap-effect error | 28.43% | 29.28% |
| Code same-token swap-effect error | 25.09% | 25.98% |

The smaller program uses half the nonlinear products and 51.7% fewer weight coefficients. Means and upstream source computation remain explicit and outside these weight counts.

All registered point-estimate checks passed: frozen hashes/instrument checks, absolute intervention errors below 30% and CE increases below 0.05 on both domains, and no more than 5% relative intervention-error increase against the original graph. The observed relative increases range from 2.50% to 3.55%.

## Uncertainty and limits

A successor paired bootstrap resampled documents 10,000 times, preserving the pairing between the two programs. The descriptive 95% intervals for the smaller graph's swap error are:

- FineWeb: 28.51–30.09%.
- Code: 24.66–27.30%.

The FineWeb interval crosses the absolute 30% threshold. Therefore, the point-estimate test passes, but this panel does not establish a strict population-level 30% bound. The largest individual-document swap errors are 34.02% and 30.16%, respectively.

The relative-preservation result is more stable: bootstrap intervals for the swap-error ratio are 1.0284–1.0316 on FineWeb and 1.0342–1.0367 on code, both below the 1.05 allowance. These are conditional descriptive intervals on the chosen panel, not familywise guarantees or evidence that repository files represent all code.

## Position in the two-stage project

Stage one supplied learned products from the folded tensor. Stage two introduced shared output corrections, shared input projections, then selected fewer products and refitted their writes while counting output storage. The 512-product result is the smallest current candidate that has passed this full set of fresh-panel preservation checks.

What remains: stronger structural simplicity, stable identities for intermediate features, selective behavioral interpretations, general graph search and composition across replacements. No claim of whole-model acceleration follows from these per-section counts. The original goal remains active.

Evidence: `MIDPOINT_PRUNED_CONFIRMATION_V1.json`, the raw per-document records and frozen panel manifest, and `MIDPOINT_PRUNED_BOOTSTRAP_V1.json` under `direct_tensor_match`. See the [overall review](research_update_2026-09-21_0104_full_coverage_and_shared_baselines.md) for the QR/folding and decomposition-to-graph explanation.
