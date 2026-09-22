# Is the error tied to particular input variables?

22 September 2026, 01:21 UTC.

The output-side answer was clear: the smaller output components have much larger percentage errors. The input-side answer is more nuanced. **The residual is sensitive to many input directions, and the answer depends strongly on how we weight those directions.** We have not found a handful of input coordinates responsible for the missing computation.

I computed the exact local derivatives of the native selected quartic function and the two frozen CP approximations. The inputs are the 1,152 coordinates entering MLP16; the outputs are the same 16 projected coordinates of the MLP16→MLP17 quartic path. This is not a derivative of the full transformer or a causal explanation of a token's prediction.

The audit uses one state—position 31—from each of the 256 documents in the newly opened panel. It removes the direction that merely rescales the RMS-normalized input. We then compare how native and approximate outputs respond to infinitesimal input changes.

| Metric | First CP candidate | Second CP candidate |
| --- | ---: | ---: |
| Value error on these selected states | 8.67% | 8.73% |
| Derivative error, treating tangent input directions equally | **26.26%** | **26.64%** |
| Derivative error, weighting directions by calibration covariance | **9.84%** | **10.02%** |

These are different questions. The first derivative metric includes directions that may vary very little in text states. The covariance-weighted metric emphasizes changes typical of the calibration geometry. It looks substantially better, but still cannot replace tests of actual state changes or finite model interventions.

For the isotropic tangent metric, the 64 largest individual input coordinates account for only about **6.9%** of sensitivity-error energy. Sixty-four coordinates are already 5.6% of the 1,152 coordinates, so this is weak concentration. The largest individual coordinate contributes less than 0.3%.

Allowing arbitrary linear combinations gives more concentration, but still no very small residual subspace: the best four input directions capture only about **6.5%**, and the best 64 capture about **35%**. By comparison, the native function's best four directions capture about **76.5%** of its derivative energy on these states. The approximations capture much of the large structure while leaving a more distributed derivative error.

This helps refine the earlier finding that four *output* directions capture most residual energy. A residual can write into only a few directions while the scalar amounts it writes depend on many input directions. A small output basis therefore does not by itself supply a small corrective circuit.

There are two limits to this conclusion. Local derivatives measure sensitivity, not which variables caused an observed value error. And a function with gradients spread across many directions can still have a compact nonlinear arithmetic program. We have evidence against a simple few-coordinate repair, not a proof against compact circuits.

Analytic derivatives passed five toy-family checks against automatic differentiation, the quartic scaling identity, and native finite-difference checks. The [full audit](../../direct_tensor_match/RESIDUAL_INPUT_SENSITIVITY_INTERPRETATION_V1.md) defines the metrics and lists limitations; the [results](../../direct_tensor_match/RESIDUAL_INPUT_SENSITIVITY_V1.json) include all 16 output features.

## Follow-up: do local sensitivity errors predict finite changes?

Not reliably by themselves. On the existing same-token pairs, the smaller original graph has **10.8%** error in the initial derivative along a norm-preserving path, but **15.3%** error in the full endpoint change. The stronger CP candidates have slightly worse local derivative errors, **12.4–13.7%**, but better endpoint errors, **9.7–11.8%**.

An exact quartic expansion shows substantial cancellation between orders. Interpreting each order's error separately would be misleading, especially because a straight line between normalized endpoints changes the input norm in between. Keeping normalization explicit changes the derivative comparison substantially.

Endpoint movements here are large: the median length is about 57% of input norm. Even the native computation is poorly approximated by its starting derivative over that distance. We should keep exact finite-response tests rather than replace them with a derivative loss alone. [Response-order and normalized-path audit](../../direct_tensor_match/QUARTIC_RESPONSE_ORDERS_INTERPRETATION_V1.md).
