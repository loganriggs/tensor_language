# Separate-role fitting repairs source swaps, but not full-contribution preservation

21 September 2026, 02:44 UTC.

Fitting the shared linear basis to the two input roles separately repairs the narrow source-preservation failure. It does **not** preserve all previously tested interventions. The smaller graph still loses substantial whole-contribution fidelity, so it is not an adopted replacement.

## Matched-width native test

The centered512-product interaction is identical in every candidate. All256-feature variants use2,654,208 total weight coefficients, compared with4,423,680 for exact first-order maps.

| Linear basis | FineWeb source error | Code source error | CE added: FineWeb / code |
|---|---:|---:|---:|
| Exact linear maps | 25.67% | 20.69% | 0.00551 / 0.02367 |
| Paired256 | 27.00% | 22.20% | 0.00921 / 0.04021 |
| Separate-role256 | 26.13% | 21.35% | 0.00911 / 0.04116 |
| Source-only256 | 25.88% | 20.97% | 0.01541 / 0.07720 |

The separate-role candidate passes all registered checks: baseline/context invariance, source errors within5% of exact maps in both domains, and CE increases below0.05. Its source-error increases are1.77% and3.19%. The source-only control improves that one intervention slightly further while damaging full replacement loss, illustrating the objective tradeoff.

These are reused diagnostic panels. No fitting uses these evaluation outputs, but repeated diagnostics are not fresh confirmation.

## Broader preservation audit

A successor CPU audit checks every existing intervention family against the exact-linear version:

| Family | FineWeb: exact → separate256 | Code: exact → separate256 |
|---|---:|---:|
| Full-contribution removal | 17.90% → 22.98% | 11.40% → 14.98% |
| Full-contribution same-token swap | 23.29% → 30.06% | 20.08% → 25.45% |
| Source-only swap | 25.67% → 26.13% | 20.69% → 21.35% |
| Context-only swap | 55.37% → 55.37% | 39.55% → 39.55% |

Removal and full-swap error increase roughly27–31% relatively. The broader5% preservation criterion therefore fails. We retain the source-specific pass and this broader failure as separate facts; one must not substitute for the other.

Context invariance is algebraic: first-order linear terms cancel in that intervention. It is not evidence of preservation of the whole folded function. The absolute context errors also remain high.

## Interpretation and next uncertainty

A covariance objective based on the paired sum can discard directions relevant to changing one role independently. Separate-role fitting helps that problem. However, preserving source-only edits does not establish preservation when both inputs move or when the whole contribution is removed.

The remaining discrepancy may involve first-order midpoint directions, cancellation with the bilinear part, or input-distribution coverage. Calibration uses32documents at64positions, whereas the native panels use256-position contexts. This is a plausible coverage issue, not an established explanation. Position-stratified errors would distinguish an early/late-context gap from a broadly inadequate low-rank approximation without immediately launching another fit.

The research result is therefore a more precise constraint on the desired circuit: it must preserve its allowed families of input changes, not merely one covariance norm or one intervention family. Stable feature identity, semantic selectivity, extraction, composition and broader OOD evidence remain unfinished.

Evidence: `MIDPOINT_ROLE_LINEAR_NATIVE_V1.json`, its raw records, `MIDPOINT_ROLE_SHARED_LINEAR_V1.json`, and successor `MIDPOINT_ROLE_LINEAR_PRESERVATION_AUDIT_V1.json` under `direct_tensor_match`.
