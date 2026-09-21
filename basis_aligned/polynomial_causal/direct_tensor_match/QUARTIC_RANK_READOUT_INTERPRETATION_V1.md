**Direct rank-constrained fitting improves its joint objective, not every value metric.**

| Output rank | Value error, panel0 / panel1 | Centered derivative error, panel0 / panel1 | Joint fitting objective |
|---|---:|---:|---:|
|8, primary|8.89% / 13.84%|21.78% / 27.66%|0.05533|
|16|8.28% / 13.61%|21.29% / 27.53%|0.05217|
|32|8.07% / 13.57%|21.06% / 27.54%|0.05086|

The primaryrank8 has integrityPASS, valueFAIL and derivativePASS under the preregistered1.1timesparent bars. Its calibration joint objective improves from the same-price prediction-PCA0.05734to0.05533, but values worsen from8.60/13.70%to8.89/13.84%. This is the expected possible tradeoff, not an optimizer bug. The fixed-feature reduced-rank solution is globally optimal for its regularized quadratic objective; neither the feature dictionary nor the whole graph is globally optimized. Runtime2.07s. Spectral objective checks, coefficient compiler replay and FP32export checks pass.

The16-form secondary meets both value and derivative comparison bars. It does not replace the failed primary of this experiment. It is a useful fixed-function input to the subsequent exact graph-rewrite test, whose fidelity and price predictions are separately registered.

[Native records](QUARTIC_RANK_READOUT_V1.json) · [Paired root rewrite](PAIRED_ROOT_INTERPRETATION_V2.md) · [Mathematical mapping](../THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-21_1953.md).
