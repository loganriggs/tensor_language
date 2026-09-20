# Linear correction helps; normalization changes the fitting question

2026-09-20 20:31 UTC

Updating only the rank8 linear output weights improved the covariance-based
four-product program from **24.16% to 22.00%** relative prediction error on the
reused second panel. Centered variation error is 32.86%. It still costs 34,560
coefficients and four products. The spherical counterpart worsened from 28.05%
to 29.18%, despite improving its own Gaussian objective.

The exact control uses

$$B=\nabla_\mu\mathbb E[F(\mu+\delta)],\qquad
W=BMA^\top(AMA^\top)^{-1},$$

where A is the frozen linear reader and delta has covariance M. No empirical
output targets are fitted. All three registered bars passed; finite-difference
derivative errors are below 8e-8 and independent scalar-DAG replay below 2e-16.

Removing all quadratic products while preserving their Gaussian mean raises
panel2 error from 22.00% to 40.60%. Individual removals give 24.63–35.54%.
The products matter to reconstruction; this is not evidence of semantic
selectivity, stable feature identity or causal circuit identification.

A separate input-only audit exposes a metric mismatch. Actual squared input
length is effectively fixed at 1152 (relative standard deviation about 9e-8).
The calibration Gaussian predicts 11.01% relative standard deviation. This
can matter algebraically: on a sphere of squared radius d,

$$\|x\|^2x_1x_2=d x_1x_2.$$

The two sides are different coefficient tensors but the same restricted
function. A planted control verifies zero sphere error while its standard
Gaussian squared error is 2d+24. This example motivates a native diagnostic;
it does not prove that radial identities explain the native errors.

The next comparison freezes candidates and evaluates raw versus normalized
artificial Gaussian probes before any further fitting. It will test whether
respecting normalization better predicts the observed errors. The quartic
candidate remains better at 18.38%, with 26 products and 47,312 coefficients.
These figures concern a selected folded polynomial contribution, not the full
normalized language model; the fixed vocabulary frame is excluded from prices.

[Linear-control receipt](../../direct_tensor_match/NATIVE_GAUSSIAN_LINEAR_CONTROL_V1.json),
[graph and removal audit](../../direct_tensor_match/GAUSSIAN_LINEAR_PROGRAM_AUDIT_V1.json),
[input-radius audit](../../direct_tensor_match/INPUT_RADIUS_GAUSSIAN_AUDIT_V1.json),
[next diagnostic](../../direct_tensor_match/NORMALIZED_PROBE_PLAN_V1.md).
