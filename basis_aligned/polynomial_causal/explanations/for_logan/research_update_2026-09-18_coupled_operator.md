# Folding the trusted coupled value path into one operator

The current evidence says the direct attention-value term and the MLP8-mediated term should be treated as one coupled object. I therefore folded them together instead of promoting either subterm. The resulting operator computes the generated edited RMS9, the exact MLP8 correction, and the direct attention-value correction from native `z8`, an upstream `delta`, and token IDs. It replays the complete joint correction on both fresh FineWeb panels with maximum relative error `2.2e-5` and runs exactly in isolation on 80 fixtures.

| Property | Evidence | Evaluation | Result | Status |
|---|---|---|---|---|
| Predicts the joint correction | fold | fresh V1 and V2 panels | max error `2.2e-5`; V1/V2 full-suffix screens pass | passes at declared boundary |
| Extracted executable interface | fold + installed/full-suffix replay | 80 fresh fixtures | isolated replay passes; installed readout error ≤ `6.0e-6`, relative ≤ `9e-7`, off-support `0` | established boundary |
| Selective manipulation | edit | V1/V2 full-suffix screens and fresh mediator null | V1/V2 controls ≤ `.25`/`.19`; mediator null 16/16 | passes with subgroup limitation |
| Composition/reuse | edit | V1 and V2 | global joint/parent `.79`/`.80`; V1 reversed subgroup fails, V2 has no reversed rows | candidate for coupled operator; independent pieces unresolved |
| Simplicity | storage accounting | both panels | 16,882,563 stored FP32 values plus native ports | not established |

This changes the object of composition. The V1 panel still falsifies independent direct/MLP8 promotion: reversed-group interaction `.49` and one control `1.12×` the target. V2 reuses the coupled operator globally, but its reversed subgroup is empty, so it cannot close that gate. The package is consequently a compact executable **coupled path**, not a claim that the two internal terms are separately reusable.

The interface has three explicit ports: native unnormalized `z8[1,T,1152]`, upstream `delta[1,T,1152]`, and token IDs. The later suffix remains external, and the program is not token-only. Its larger storage than the supplied-norm mediator is the price of closing normalization; no matched-effect simplicity claim is made.

## Receipts

- [Coupled fold result](../../CITY_MLP8_COUPLED_VALUE_V1_CPU_RESULT.json), [isolated replay](../../CITY_MLP8_COUPLED_VALUE_V1_ISOLATED_RESULT.json), [installed/full-suffix replay](../../CITY_MLP8_COUPLED_VALUE_INSTALLED_V1_RESULT.json), and [package manifest](../../extracted_circuits/city_mlp8_coupled_value_v1/manifest.json).
- [Fresh V1 composition](../../CITY_VALUE_PATH_FRESH_V1_RESULT.json), [fresh V2 reuse](../../CITY_VALUE_PATH_FRESH_V2_RESULT.json), and [fresh same-boundary null](../../CITY_MLP8_NORM_CLOSED_FRESH_V1_NULL_RESULT.json).
- [Coupled operator registration](../../CITY_VALUE_PATH_FRESH_V2_PREREGISTRATION.md) remains scoped to declared native ports and full-suffix evaluation.
