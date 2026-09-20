# More reuse improves the Gaussian objective but leaves the transfer failure

2026-09-20 22:04 UTC

The mixed skip reuses12 linear-reader activations as well as6 quadratic
products. At output rank2 it adds only24 coefficients over quadratic-only
reuse:21,972 total coefficients, still10 product nodes. Literal DAG replay is
below7e-16. Exact Gaussian moments include the linear/quadratic cross terms.

The primary Gaussian reconstruction error improves from11.53% to11.25%.
Native FineWeb loss damage is+.01330 nats/token. Code feature1 removal-effect
error is41.95%, improving slightly but still FAILING the registered40% target.
All ranks were reported; none was chosen retrospectively. Expanding the same
readout further is now deferred pending a metric diagnosis.

A completed CPU audit gives that diagnosis a sharper question. The exact
calibration-Gaussian objective predicts improvement for every fixed canonical
mode. But on the reused text panels, quadratic-only rank2 increases mode1's
polynomial MSE by21–22%; mixed rank2 increases it by20%. Thus the failure is
not explained by the Gaussian objective knowingly sacrificing that mode.

These same-panel polynomial diagnostics differ from the earlier native removal
screens in both data and metric. Their differing trends must not be attributed
to normalization alone without a matched comparison.

Next, with all programs frozen, recompute the Gaussian prediction using each
panel's actual mean and covariance. Retain the calibration centering and all
mean terms. If the sign remains wrong, the discrepancy requires higher-order
input information, rather than only a different covariance matrix. This test
is registered; it has not yet run.

[Fit](../../direct_tensor_match/MIXED_SKIP_FIT_V1.json),
[native results](../../direct_tensor_match/MIXED_SKIP_NATIVE_V1.json),
[modewise objective audit](../../direct_tensor_match/SKIP_MODE_OBJECTIVE_AUDIT_V1.json),
[next test](../../direct_tensor_match/SKIP_MOMENT_MATCH_PLAN_V1.md).
