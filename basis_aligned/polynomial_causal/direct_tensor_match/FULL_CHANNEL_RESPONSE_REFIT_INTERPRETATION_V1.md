**Response-aware fitting improves intervention fidelity modestly, but retains failure**

A fixed-cost conditional writer fit augments covariance coefficient matching with historical source-direction response matching. The product dictionary,3686product count and14,067,072stored coefficients remain unchanged. Exact affine mean/tangent correction is included. The primary weight lambda1was fixed before results; no native48-document outcomes entered fitting or selection.

| Response-loss weight | Covariance coefficient error | Historical response error |
|---|---:|---:|
|0|10.80%|7.45%|
|.1|10.84%|5.67%|
|1, primary|11.17%|3.53%|
|10|12.48%|1.64%|

All registered fitting predicates pass. Normal-equation residuals are below3e-15, with five independent autograd/direct least-squares controls. The strong fitting gain does not establish endpoint improvement: the response metric is pre-final-normalization and uses centered source directions from historical inputs.

The frozen primary was then evaluated under the same direct-source-path donor swaps and original thresholds:

| Primary cohort | Original full-path error | Response-refit full-path error |
|---|---:|---:|
|FineWeb continuation|14.77%|14.59%|
|FineWeb spaced word|12.27%|11.62%|
|Code continuation|13.78%|12.51%|
|Code spaced word|10.49%|9.03%|

All aggregate full-path errors improve, but the all-cells10%criterion still fails (three cells remain above it). Calibration response improvement is52.58%; native relative error gains range1.22–13.90%. These metrics and direction distributions differ, so the comparison demonstrates limited transfer rather than quantifying a single generalization gap.

The isolated leading mode still passes all four primary effect limits. Its errors worsen in three cells and improve for code continuation. Thus the full-layer improvement is not uniformly an improvement of the tracked circuit computation. The CPU paired-document successor also shows only14/29FineWeb continuation documents improve, despite the aggregate gain. No independence-based confidence interval is claimed because donors and recipients are shared.

Natural-input fidelity of this new program has not yet been revalidated. The previous fresh validation belongs to the original program and cannot be transferred to this one. No candidate is adopted.

This tests a useful restriction of the overall search: fixed native products with movable output weights and an explicit response constraint. It does not test whether newly learned product directions, sparse Tucker cores, or cross-layer arithmetic graph edits can represent the desired response more economically. Further lambda tuning on these opened panels is lower priority than addressing that structural restriction and the response metric's downstream sensitivity.

[Fit results](FULL_CHANNEL_RESPONSE_REFIT_V1.json) · [Frozen export](FULL_CHANNEL_RESPONSE_EXPORT_V1.json) · [Native intervention results](FULL_CHANNEL_RESPONSE_NATIVE_V1.json) · [Paired audit](FULL_CHANNEL_RESPONSE_TRANSFER_AUDIT_V1.json).
