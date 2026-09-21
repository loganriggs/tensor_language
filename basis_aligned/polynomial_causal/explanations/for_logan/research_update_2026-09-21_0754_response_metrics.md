# The sensitivity gap is shared by the baseline; the perturbation metric matters

21 September 2026,07:54 UTC. Follow-up to [stable functions and unstable products](research_update_2026-09-21_0748_stable_functions_unstable_products.md).

**The shared graph is more faithful to input derivatives than the separate baseline.** The previous18–20%derivative errors therefore do not show that sharing damaged responsiveness. Both compressed programs have difficulty with unrestricted input perturbations, while errors are much smaller for perturbations shaped by the measured covariance.

## A matched comparison

We differentiate each selected component with respect to the earlier normalized input z, holding the later native input h fixed. The Jacobian J is the linear response to a small change in z. This is a conditional interface test, not the full upstream causal response: it omits how h would change along an actual model path.

| Component | Euclidean perturbations: separate → shared | Covariance-shaped perturbations: separate → shared |
|---|---:|---:|
|1|23.89% → **20.09%**|5.73% → **4.64%**|
|2|19.14% → **17.93%**|4.94% → **4.63%**|
|3|19.40% →19.40%|6.49% →6.49%|

All comparisons satisfy the registered relative limit of1.10times baseline error. The15%absolute diagnostic fails in Euclidean geometry and passes in covariance geometry. Projecting onto the sphere's tangent plane does not remove the larger errors. Independent automatic differentiation verifies the baseline's decoded quadratic program to below2.1e-14.

These percentages use different denominators because the perturbation distributions differ. They cannot be substituted for one another. Covariance weighting emphasizes directions observed in calibration; unrestricted perturbations also test directions of low observed variance. Neither by itself proves OOD fidelity.

## A direct weight-space objective for source responsiveness

For a centered quadratic read error

$$
e(\delta)=\delta^\top\Delta Q\delta,
$$

its gradient is $2\Delta Q\delta$. If the input displacement has covariance $\Sigma$, then

$$
\mathbb E\|\nabla e(\delta)\|_2^2
=4\operatorname{tr}(\Delta Q\Sigma\Delta Q).
$$

This identity needs the second moment, not a Gaussian assumption. Our exact centered linear and mean terms are retained, so the source-gradient error comes from this quadratic residual.

In the existing whitened coordinates $M=\Sigma^{1/2}Q\Sigma^{1/2}$, write $H=\Sigma^{-1}$. We are testing

$$
\mathcal L_\lambda=
\frac{1}{1+\lambda}\left[
\frac{\|\widehat M-M\|_F^2}{\|M\|_F^2}
+\lambda\frac{\operatorname{tr}(\Delta M H\Delta M)}
{\operatorname{tr}(MHM)}\right],
$$

summed over the four source outputs before normalization by their total energies. A tiny ridge term is included when solving the linear readout. The existing per-pair output scales remain fixed. This is a **source-gradient objective**, not direct optimization of the outer component's Jacobian; the outer product and normalization can still change the tradeoff.

## What has completed

A fixed-product-direction control solved the output weights exactly at weights $\lambda=0,0.1,1,10$. At the primary $\lambda=1$, component derivative errors barely changed:20.090%to20.084%, and17.933%to17.918%. The required10%relative improvement fails. Scalar fidelity remains within its existing limits.

Thus readout adjustment alone does not repair the response gap. The product directions themselves are the next variable to test.

Five random loss/gradient checks agree with dense reconstruction and differentiation through the linear solve to floating-point precision. Four of five structurally planted targets recover from the first start. The signed-output case fails at2.55%; restart9473 with4,000steps also fails. A further start,5473 at learning rate0.05 and4,000steps, recovers to6.95e-10. Both seed and rate changed, so this establishes successful recovery without isolating which change mattered. The original failures are retained; gradient correctness does not guarantee optimizer recovery.

## A capacity bound for this particular metric

The256mixed products have at most512distinct input directions. A right-unfolding rank bound puts their normalized source-gradient error at **at least19.08%**; the parent is at29.52%. Thus there is room to improve, but arbitrarily small source-gradient error is impossible at this capacity. This bound applies to the four source quadratic forms under the stated metric—not to the outer components’ Jacobian percentages in the table, and not to general arithmetic DAGs. An explicit small-matrix unfolding check validates the bound calculation. [Bound and scope](../../direct_tensor_match/SOURCE_GRADIENT_BOUND_V1.json).

## Registered joint refit

A managed GPU job is fitting the product directions with four objective balances and two learning rates. Every arm starts from the same frozen parent graph, gets2,000cosine-decayed Adam steps, and keeps512products and897,804floating coefficients. The third component's private branch stays unchanged and remains scored.

The primary balance is $\lambda=1$, selected in advance. Its rate is selected by its weight objective. It must reduce both first-two component derivative errors by at least10% while preserving the existing per-component scalar limits. The $\lambda=0$arm controls for additional optimization budget. Other balances measure the tradeoff, not alternative primary successes chosen after seeing results.

No new success is assumed while the job runs. Even a passing conditional-response screen would leave fresh intervention tests, semantic identity, and native upstream input dependencies unresolved.

## Receipts

- [Matched derivative comparison](../../direct_tensor_match/BASELINE_SENSITIVITY_V1.json).
- [Implicit objective checks](../../direct_tensor_match/SOURCE_SOBOLEV_CHECK_V1.json), [five planted recoveries](../../direct_tensor_match/SOURCE_SOBOLEV_RECOVERY_V1.json), and [retained restart failure](../../direct_tensor_match/SOURCE_SOBOLEV_CASE3_SEED9473_V1.json), and [successful alternate rate/start](../../direct_tensor_match/SOURCE_SOBOLEV_CASE3_SEED5473_V1.json).
- [Fixed-readout control](../../direct_tensor_match/SOURCE_SOBOLEV_READOUT_V1.json).
- [Frozen joint-refit plan](../../direct_tensor_match/SOURCE_SOBOLEV_REFIT_PLAN_V1.json).

All completed response measurements use448previously opened states. The CPU comparisons and the weight-only GPU refit require no new native-model forwards. These are structural and objective diagnostics, not fresh behavioral confirmation.
