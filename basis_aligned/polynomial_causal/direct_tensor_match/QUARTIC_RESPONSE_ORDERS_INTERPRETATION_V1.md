# Exact response orders expose a path-geometry confound

22 September 2026, 01:27 UTC. CPU audit of feature 1 on the existing 30 directed/20 unordered same-newline-token pairs. Seven frozen programs; no new fitting.

For recipient x and difference d=donor-x, the exact quartic expansion is

$$
F(x+td)=\sum_{k=0}^4 a_k t^k,\qquad a_k={4\choose k}T[x^{4-k},d^k].
$$

Five-node interpolation extracts these coefficients for each executable graph. Native coefficients independently agree with symmetric tensor polarization to 1.31e-14, and the summed response matches archived references to 1.03e-14. This is an algebraic instrument, not fitting a candidate to five training examples.

The native first-order-only ambient response has 49.0% error. Native order norms relative to total response are 1.134, .473, .087, .018. CP512 seed1001 has 31.3/67.7/85.1/101.0% relative error in individual orders, but 9.72% total response error. Squared order-error norms sum to 24.84 times the squared net error (20.15 for seed1002). Signed contributions can be negative; they are not independent percentages of failure.

The straight line passes through unnormalized states. For a norm-preserving chord,

$$
z(t)=\frac{\|x\|}{\|x+td\|}(x+td),\qquad
F(z(t))=\frac{\sum_k a_k t^k}{(1+\alpha t+\beta t^2)^2},
$$

where alpha=2 x^T d/(x^T x) and beta=d^T d/(x^T x). Its initial derivative is a1-2 alpha a0. This retains normalization as a rational operation. Endpoint rescaling differs from the original donor by only about 1.05e-6 relative in this panel. Direct native finite differences verify the normalized derivative below 1e-7.

| Program | Normalized initial derivative error | Full endpoint error |
|---|---:|---:|
| Original384 |10.83%|15.32%|
| CP512 seed1001 |12.39%|9.72%|
| CP512 seed1002 |13.71%|11.78%|
| CP256 seed1001 |18.35%|14.95%|
| CP256 seed1002 |17.35%|14.98%|
| Learned shared1101 |21.36%|25.85%|
| Learned shared1102 |30.02%|36.35%|

CP512 normalized-path error stays roughly 12–14% at fractions .01, .05, .1, .25, .5, ending around 10–12% at 1. Large ambient Taylor errors therefore are not evidence of catastrophic normalized-path failure. The original384 graph has better local derivative fidelity than CP512 yet worse endpoint fidelity; a local metric cannot rank finite fidelity reliably by itself.

The native function's own linearization errors along the normalized path are 0.54%, 2.83%, 6.11%, 19.48%, 62.09%, 322.49% at those fractions. Median endpoint movement is 56.84% of input norm. Path curvature—not necessarily candidate error—makes a starting derivative inadequate over the full change. Intermediate normalized states need not lie on the text-state manifold.

Consequence: retain exact finite responses and normalization-aware interventions. Do not optimize individual ambient Taylor errors assuming their sum must improve. The compressed CP candidates degrade both locally and finitely on these registered changes, despite low parent-Gaussian error.

Files: audit_quartic_response_orders.py and QUARTIC_RESPONSE_ORDERS_V1.json. Includes cross-order Grams, interpolation checks, and normalized step ladders for all seven programs. Taylor orders depend on the anchor; directed pairs are correlated. Neither is an independent circuit component. No semantic or selective causal promotion.
