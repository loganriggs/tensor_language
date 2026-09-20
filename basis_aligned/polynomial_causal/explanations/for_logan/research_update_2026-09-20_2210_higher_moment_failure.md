# Matching mean and covariance does not fix the wrong correction direction

2026-09-20 22:10 UTC

The frozen quadratic skip was evaluated under Gaussians matched to each actual
text panel's mean and covariance. Its calibration centering was retained;
mean-offset terms were included and independently checked by quadrature.

| Feature1 quantity | Context64 | Context256 |
|---|---:|---:|
| Matched-Gaussian predicted MSE reduction | +4.64 | +8.87 |
| Actual MSE reduction | -14.01 | -15.97 |
| Gaussian residual/correction alignment | +4.93 | +7.18 |
| Actual residual/correction alignment | -4.92 | -5.90 |

Positive reduction means improvement. All registered diagnostic predictions
pass: matching first two moments still predicts the wrong sign. The correction
energy discrepancy is much smaller than the alignment discrepancy. Because
the quartic residual is paired with a quadratic correction, this alignment
requires moments through degree6; covariance alone cannot determine it.
This does not isolate a single cumulant or prove all Gaussian approximations
are useless. It identifies a concrete failure for this proposed graph edit.

We have begun a matched-cost comparison: blend exact Gaussian function moments
with empirical function moments from the ORIGINAL calibration panel only.
This explicitly introduces data-informed polynomial regression beyond an input
covariance. It uses weight-evaluated polynomial targets, no task labels, and
none of the diagnostic/test panels for fitting. All input directions, graph
nodes, centering, output rank and coefficient count stay fixed.

For blend weights0/.25/.5/.75/1, calibration relative error falls from19.94%
to17.27%/16.29%/15.84%/15.52%. The pure empirical endpoint worsens the Gaussian
objective, so the two measures genuinely disagree. Primary weight.5 was fixed
before outcomes. Native transfer is still pending; calibration gains alone
are not evidence of better circuits or generalization.

[Matched-moment result](../../direct_tensor_match/SKIP_MOMENT_MATCH_V1.json),
[alignment decomposition](../../direct_tensor_match/SKIP_GAIN_COMPONENTS_V1.json),
[comparison registration](../../direct_tensor_match/SKIP_METRIC_BLEND_PLAN_V1.md),
[calibration fit](../../direct_tensor_match/SKIP_METRIC_BLEND_FIT_V1.json).
