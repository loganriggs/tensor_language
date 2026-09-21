**A global tensor penalty limits drift, but neither objective establishes a faithful circuit**

Two100-stepAdam fits use identical3686-product capacity, inherited input directions, covariance parameter coordinates and exact conditional output solves. The mixed objective is normalized covariance coefficient error plus historical response error plus.1times normalized isotropic coefficient error. The other objective is purely isotropic. Their initializer, coordinates and affine mean/tangent repair remain data-informed; only the latter fitting loss is weight-only.

| Fit | Covariance error | Isotropic error | Historical response error |
|---|---:|---:|---:|
|Fixed products, response refit|11.17%|40.24%|3.53%|
|Learned directions, previous unguarded objective|10.57%|55.15%|1.88%|
|Learned directions, mixed objective|10.58%|38.05%|2.99%|
|Learned directions, isotropic objective|30.78%|36.41%|33.59%|

Mixed fitting satisfies the registered global/covariance limits. Its objective decreases8.13%; the isotropic objective decreases only3.04%, so the requirement of5%improvement for both fails. Both best checkpoints occur at step100, and both exports pass FP32 replay (1.13e-6and6.33e-7). These runs do not establish convergence. The actual CPU Pareto audit shows the mixed candidate improves all three fitting measures over both fixed-product candidates at the same14,067,072coefficient budget. This dominance concerns fitting metrics only.

**Native comparisons**

Both frozen alternatives were evaluated under identical rules on the opened32FineWeb/16code documents, with no native-outcome selection between them.

| Primary cohort | Mixed full-path error | Mixed mode error | Isotropic full-path error | Isotropic mode error |
|---|---:|---:|---:|---:|
|FineWeb continuation|14.06%|4.94%|36.49%|62.45%|
|FineWeb spaced word|11.08%|7.75%|14.78%|45.19%|
|Code continuation|11.88%|2.94%|49.53%|63.64%|
|Code spaced word|8.45%|4.63%|14.21%|55.70%|

The mixed program passes the isolated continuation-mode limits but retains three full-path failures. Pure isotropic fitting fails all these intervention limits. The direct path replaces the preceding MLP's source contribution while holding recipient attention fixed, then recomputes the last input state, normalization, bilinear layer and final output. The mode test isolates that mode's scalar change; it is not claimed to explain the whole intervention.

Natural fullMLP effect error is4.21%FineWeb/2.07%code for mixed, versus4.44%/2.12%for the original affine pruning program. Mixed CE added is+.00430/+.00018nats per token. For isotropic, natural effect error is8.41%/6.77%and CE added+.01210/+.02690, failing the code.02CElimit. No new independent panel has been opened for these candidates, and neither passes the combined adoption screen.

A separate CPU mode audit rejects a simple mode-order confusion as the leading explanation for the previous learned-factor regression: that program's continuation-mode cosine remains.99969and its first/second singular-value ratio3.39, versus native3.43. Its covariance mode error nevertheless increases to2.78%, and these close calibration-coordinate modes are not interchangeable under every native intervention.

The result supports explicitly controlling more than the calibration geometry. It does not show that isotropic Frobenius error alone selects useful circuits, that weight-only discovery is impossible, or that the remaining gap can be fixed by more penalty tuning. The full goal still requires computational specification, extraction, OOD and selective manipulation, stable feature identification and reusable composition. The stronger full-layer baseline is useful for judging subsequent decomposition and graph searches; it is not a substitute for those requirements.

A promising distinct comparison is to fit the composed two-MLP path on artificial inputs propagated through the actual preceding weights, retaining normalization explicitly. That would test a weight-defined source/context geometry rather than treating the last MLP's input distribution as either isotropic or supplied solely by text calibration. It is a proposed next experiment, not an executed result.

[Fit and exports](FULL_QUADRATIC_MULTIGEOMETRY_FIT_V1.json) · [Native summary](FULL_QUADRATIC_MULTIGEOMETRY_NATIVE_V1.json) · [Fitting-metric Pareto audit](FULL_QUADRATIC_MULTIGEOMETRY_PARETO_V1.json) · [Mode identity audit](LEARNED_CONTINUATION_MODE_AUDIT_V1.json).
