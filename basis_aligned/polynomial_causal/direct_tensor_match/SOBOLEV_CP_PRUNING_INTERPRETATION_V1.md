Derivative-aware pruning does not repair the component-fidelity gap

22 September 2026, 00:36 UTC. CPU experiment, approximately 2.38 seconds after independent moment controls. The frozen parent is the learned mixed CP512 approximation, not the original model. No native training or new causal validation is claimed here.

The question was whether preserving changes in the polynomial would protect component responses better than preserving its values alone. Write an affine quartic feature in standardized Gaussian coordinates as

$$
\phi_a(z)=\prod_{s=1}^4(\ell_{as}^{\top}z+b_{as}),\qquad z\sim\mathcal N(0,I).
$$

Its exact derivative Gram is

$$
(G_D)_{ab}=\mathbb E[\nabla_z\phi_a\cdot\nabla_z\phi_b]
=\sum_{s,t}(\ell_{as}^{\top}\ell_{bt})
\mathbb E\left[\prod_{u\ne s}(\ell_{au}^{\top}z+b_{au})
\prod_{v\ne t}(\ell_{bv}^{\top}z+b_{bv})\right].
$$

Each expectation has six factors. The implementation shares exact Gaussian subset moments. Because x=mu+S z, this preserves derivatives weighted by S S^T in original input coordinates. It does not include final normalization, the selected same-token context distribution, or finite interventions directly.

Five structural control families compare both Gram values and parameter gradients against independent four-node-per-axis Gaussian quadrature, which integrates degree six exactly. Maximum discrepancies are about1.5e-15. This makes a formula bug unlikely for these checked cases; it does not establish that the metric matches the desired native effects.

The primary metric averages relative parent value error and relative parent derivative error, giving each equal weight. A derivative-only secondary metric is also reported. Supports are selected by the same exact backward-deletion/refit method, with fixed budgets512/384/256/128. Evaluation labels are used only for scoring. The term directions remain fixed.

| Start and budget | Parent Gaussian error | Parent derivative error | Native text error | Native root1 sensitivity error | Native root1 same-token response error |
|---|---:|---:|---:|---:|---:|
|1001,384 terms|0.774%|1.713%|6.606%|12.718%|15.575%|
|1002,384 terms|0.506%|1.188%|6.839%|12.627%|14.916%|
|1001,256 terms|2.411%|4.926%|8.071%|21.973%|26.051%|
|1002,256 terms|2.081%|4.275%|7.649%|17.879%|19.600%|

Integrity passes. At the primary256-term budget, the parent-value requirement, relative native-retention requirement and absolute component requirement all fail. The derivative-only secondary is not a repair either:256-term response errors25.96/19.44%, sensitivity21.17/17.12%. An isolated improvement at128 terms for one start is not a successful joint result.

Compared with value-based pruning, the results are close. The distinction between general Gaussian derivatives and selected native responses matters. This negative result applies to frozen-direction greedy compression of these parents under these metrics; it does not rule out moving term directions, richer graph rewrites, a better identified component, or a different conditional distribution.

Decision: further small reweightings of this fixed-direction pruning objective are lower priority. Continue the queued shared-producer learning experiment, which changes the directions at constant program cost. Retain the unpruned mixed CP model as the stronger fidelity baseline. A possible later compression test would jointly refit the retained directions to the parent, rather than only refit output coefficients; that experiment has not run.

Artifacts: [all results](SOBOLEV_CP_PRUNING_V1.json), [derivative Gram](gaussian_cp_derivative_gram.py), [independent controls](GAUSSIAN_CP_DERIVATIVE_GRAM_CONTROLS_V1.json), [pruning experiment](audit_sobolev_cp_pruning.py). No compressed model from this screen is promoted for adoption.
