**Covariance-weighted improvement concealed substantial global tensor drift**

Exact original-weight comparisons reproduce the three frozen programs' covariance coefficient errors to7.88e-11. Dense mode-Gram controls pass below2.9e-16. No fitting or text forwards were performed; runtime5.496s.

| Program | Covariance coefficient error | Isotropic coefficient error | Error energy in lowest-variance quarter |
|---|---:|---:|---:|
|Original pruned program|10.80%|39.39%|24.55%|
|Fixed products, response refit|11.17%|40.24%|24.53%|
|Learned input factors|10.57%|55.15%|43.44%|

The prediction that learned isotropic error exceeds1.25times the fixed-response error passes. The prediction that more than half its error lies in the lowest-variance quarter fails:43.44%is substantial but not a majority. All four variance bands worsen relative to the fixed-response program. The lowest quarter's band error rises41.00%to74.80%; the highest quarter rises38.41%to43.63%.

These are exact homogeneous quadratic coefficient comparisons in original residual coordinates, with Euclidean unembedding output norm. Affine corrections are excluded. Band energy partitions one input slot of the symmetric tensor's error Gram; cross-band interactions contribute through both slots. This is not a decomposition of behavioral error or proof that low-variance drift caused the native regression.

The historical covariance has eigenvalues from.00355549to147.09978, and none required flooring. A covariance metric weights a coefficient interaction in its eigenbasis by the product of the two input variances. Consequently it can assign little cost to changes that are large in an isotropic metric. Learned factors exploit a freedom that fixed native factors largely lack; that observation is compatible with the measured regression, but not a causal explanation by itself.

An actual CPU successor compares normalized objectives on the already frozen equal-cost candidates:

$$
J_\eta=E_{\mathrm{cov}}+E_{\mathrm{response}}+\eta E_{\mathrm{isotropic}}.
$$

Each term is a squared relative error in its own declared metric. The learned and fixed-response programs exchange ranking at eta=.01544. At eta=.02or.1the fixed-response candidate ranks better; at eta1the original pruning candidate ranks best. This is post-result algebra, not a new optimization, independent selection rule or evidence that a chosen weight repairs behavior.

The next implementation supports exact conditional output fitting across multiple coefficient metrics and optional response constraints. Five independent explicitly lifted least-squares and gradient controls pass below2e-15. It can compare a purely isotropic fitting objective with a covariance/response objective plus an isotropic term at identical product counts. A covariance-informed initializer or affine correction must still be labeled data-informed even when one fitting objective itself is weight-only.

No new program is adopted. The general decomposition/graph objective remains open; this result identifies a concrete weakness of the previous objective rather than ruling out learned-feature simplification.

[Exact geometry audit](FULL_QUADRATIC_GEOMETRY_AUDIT_V1.json) · [Frozen objective tradeoff](FULL_QUADRATIC_METRIC_TRADEOFF_V1.json) · [Multiple-metric solver](multigeometry_quadratic.py) · [Independent controls](MULTIGEOMETRY_QUADRATIC_CONTROLS_V1.json).
