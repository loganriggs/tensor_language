# Weight-ordered greedy prefix: pre-outcome analytic forecast — 2026-09-08 08:39 UTC

## Scope

This is a zero-forward forecast written while the exact prefix runner is still managed-queued and
its result file does not exist.  It uses only the already opened six-head factorial's pooled
singleton signed projections and its P5 union projection.  It cannot compute prefix cosine or
residual because rowwise effect vectors and singleton cross-products were not stored.  It does not
change the frozen selection/validation rules and cannot adjudicate the pending result.

For role order `h_1,...,h_5`, define the additive projection forecast

`r_add(P_k) = sum_{a=1}^k <e_{h_a},g>/<g,g>`.

The P5 projection interaction is the observed simultaneous-union recovery minus `r_add(P5)`.

| population | role | phase | P1 | P2 | P3 | P4 | additive P5 | observed P5 | P5 projection interaction |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| original | temporal | FIT | .0153 | .2555 | .4947 | .4947 | .5217 | .5543 | +.0326 |
| original | temporal | HOLDOUT | .0116 | .2420 | .5004 | .4994 | .5302 | .5623 | +.0321 |
| OOD | temporal | FIT | .0115 | .2303 | .5108 | .5143 | .5367 | .5748 | +.0381 |
| OOD | temporal | HOLDOUT | .0117 | .2370 | .5264 | .5290 | .5525 | .5850 | +.0325 |
| original | is-was | FIT | .1010 | 1.0321 | 1.0025 | 2.0787 | 2.1442 | 2.3048 | +.1606 |
| original | is-was | HOLDOUT | .1164 | 1.1411 | 1.1079 | 2.2184 | 2.2921 | 2.5014 | +.2094 |
| OOD | is-was | FIT | .1343 | 1.5507 | 1.5354 | 2.6160 | 2.7969 | 2.9140 | +.1172 |
| OOD | is-was | HOLDOUT | .1427 | 1.5662 | 1.5522 | 2.6112 | 2.7893 | 2.9009 | +.1116 |

## Prospective interpretation

- **Is-was:** P2 (`L7H7+L9H4`) is the sharp compact prediction.  Its additive original-FIT
  recovery is `1.032`, well inside the selection interval `[.50,1.50]`; its OOD additive forecast
  `1.55-1.57` stays inside the validation cap `1.75`.  L6H7 at P3 is slightly negative, while
  adding L9H1 at P4 creates the already observed severe overshoot.  The pending exact P2 execution
  must still pass cosine, residual, direction, templates, and collateral.
- **Temporal:** P3 (`L7H7+L9H4+L9H1`) is deliberately boundary-sensitive: its additive
  original-FIT recovery is `.4947`, only `.0053` below the frozen `.50` gate.  P4 adds essentially
  zero (`L6H7=-.00005`), while P5's actual union is `.5543`.  A P3 selection would show favorable
  finite prefix interaction; P5 would show that L5H1 or the complete interaction is needed for the
  registered quality level.  A P4 selection would be especially diagnostic because its additive
  increment is null and therefore requires nonlinear prefix interaction.
- P5 projection interaction is positive in all eight pooled cells (`+.0321` to `+.2094`), but this
  does not identify how interaction accumulates at earlier prefixes.  It must not be prorated or
  used to manufacture prefix geometry.
- The forecast illustrates the mathematical review's warning: weight order is not causal marginal
  gain, and more heads are not monotonically better for fidelity.  A passing exact prefix supports
  prospective compactness for this order, not global subset optimality.

The result comparison is fixed: report the actual selected prefix beside this forecast, preserve
any boundary miss without threshold change, and proceed to formula mediation only if the original
greedy A/B/C/E eligibility gates pass.
