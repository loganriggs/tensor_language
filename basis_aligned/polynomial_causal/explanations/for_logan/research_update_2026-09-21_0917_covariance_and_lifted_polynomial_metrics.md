# Why input covariance alone misses the errors that matter

21 September 2026, 09:17 UTC. Follow-up to the [fresh intervention failures](research_update_2026-09-21_0910_fresh_group_and_constituent_interventions.md).

**We found two concrete mismatches between our fitting geometry and component behavior.** Even after matching the observed input mean and covariance, a Gaussian calculation underestimates the actual quadratic-read error. Separately, errors tend to occur where the downstream component is more sensitive to them. Both effects survive explicit formula and execution checks.

This supports changing the error metric, rather than simply running the same coefficient fit longer. It does not establish a successful replacement metric or a new circuit.

## What we measured

For each of six source reads, let its approximation error be

$$
e(z)=z^\top D z+\ell^\top z+b.
$$

D is the fitted-minus-original quadratic matrix; the affine terms include our existing centered mean/linear corrections. We decoded both the earlier partial-sharing graph and the compact399-product graph into this form and checked it against their executable source reads.

We then compared two quantities:

1. The empirical squared error, averaged over the actual cached states.
2. The analytic squared error under a Gaussian with **the same panel mean and covariance**.

For Gaussian z with mean mu and covariance Sigma, define c=2Dmu+ell and m=tr(D Sigma)+mu^T Dmu+ell^T mu+b. The calculation is

$$
\mathbb E[e(z)^2]
=2\operatorname{tr}(D\Sigma D\Sigma)+c^\top\Sigma c+m^2.
$$

Five small Gaussian controls compare this formula with100,000independent probes each; all agree within five Monte Carlo standard errors. Native dense-read replay also passes. An initial lookup used the wrong affine-field names for the older private program; it failed before analysis and was corrected from the actual executor schema.

The diagnostic uses1536training sites and448previously opened sites separately. It does not fit a new student, use the fresh intervention panel, or establish document independence for these historical chunks.

## Covariance closure fails on the measured read errors

Across2programs ×2panels ×6reads, the matched-Gaussian estimate is only **0.325–0.860times the empirical squared error**. All24comparisons fail the registered requirement to agree within10%.

Because each Gaussian uses that very panel's observed mean and covariance, a difference in those statistics cannot explain this diagnostic. The remaining mismatch concerns higher moments of the read errors. This does not say the Gaussian metric is mathematically wrong; it is a different distributional geometry.

For the compact graph's sixth read—the one most implicated in the third-component failure—the ratio is **0.325on training states and0.441on opened states**. Thus the Gaussian estimate understates its empirical squared read error by roughly3.1times and2.3times respectively.

## Downstream sensitivity compounds the mismatch

For one component phi=AB, source-read errors produce first-order terms

$$
\delta\phi\approx
-\frac{B}{2s}\,\delta q_a
+\frac{A}{s}\,\delta q_b.
$$

Here s is the native RMS denominator. Each squared term has the form E[w e^2], with a nonnegative sensitivity weight w. A constant output weighting implicitly risks replacing this by E[w]E[e^2].

The independent approximation is **0.305–0.980times the actual weighted error**;20of24comparisons differ by more than10%. These errors are generally larger where the component is more sensitive.

For the compact graph's sixth read, combining matched-Gaussian closure with sensitivity independence predicts only **23.8%of the measured weighted training error and38.3%of the opened-state error**. This is a diagnostic of one linearized term; signed cross terms and the product of read errors are still needed for the complete component error, and were retained in the preceding exact audit.

Some errors are concentrated. For the compact graph's first read of component3, the top1%of opened sites account for50.3%of that term's weighted error. A few chunks can therefore matter disproportionately; another scalar average is not automatically a reliable fitting target.

## How this relates to your paper's M

The paper defines M as a metric on weight tensors and derives its Gaussian form from moments of lifted polynomial inputs, including self-contractions. It also requires compatible structure for efficient recursive contraction. M is therefore more general than an ordinary input covariance matrix. Its normalized similarity identifies positive proportionality; our reconstruction losses must additionally preserve magnitude. [Paper, §2.2–2.3](https://arxiv.org/html/2605.15183#S2.SS2).

For our quadratic case, write a symmetric quadratic form as

$$
q(z)=\theta^\top\psi(z),\qquad
\psi(z)=(z_1^2,\ldots,z_d^2,\sqrt2z_1z_2,\ldots).
$$

Our concrete data-dependent functional metric would use the **uncentered second moment of these quadratic features**:

$$
M_\psi=\mathbb E[\psi(z)\psi(z)^\top],\qquad
\mathbb E[(q-\widehat q)^2]
=(\theta-\widehat\theta)^\top M_\psi(\theta-\widehat\theta).
$$

Its entries involve fourth moments of z. With a sensitivity weight, the corresponding object is E[w psi psi^T]. Biases and linear terms require extending the feature vector. These are our local constructions; they are not a claim that an arbitrary empirical metric preserves the paper's efficient deep recursion.

## Five controls make the distinction explicit

We checked five distributions in dimension8. All have population mean0and covariance I, yet the same polynomial e(x)=x_1^2-x_2^2 has different expected squared values:

| Input distribution | E[e(x)^2] | Rank of the36-dimensional quadratic-feature metric |
|---|---:|---:|
| Standard Gaussian |4|36|
| Independent random signs |0|29|
| Uniform signed coordinate axes, radius sqrt(8) |16|8|
| Independent sparse entries: zero75%, otherwise +/-2 |6|36|
| Uniform sphere, radius sqrt(8) |3.2|36|

The analytic lifted metrics and empirical checks agree. These are five distribution/metric controls, not five new decomposition-recovery experiments. A final exact-float equality assertion initially failed on the representation of3.2; the corrected test uses a1e-12tolerance and preserves the expected values.

The random-sign example also exposes an OOD danger: this nonzero polynomial vanishes everywhere on that distribution. Perfect empirical functional matching alone need not preserve the global polynomial.

## Executable consequence for the next objective

A possible next objective combines a positive coefficient-space penalty with the data-dependent functional term. We checked that adding0.01I removes the toy lifted-metric nullspaces. That numerical floor is a control, not a tuned native hyperparameter or a guarantee of OOD fidelity.

At native width1152there are664,128symmetric quadratic coordinates. An empirical moment matrix from1536states has rank at most1536. We should therefore use matrix-free contractions and retain a global coefficient constraint rather than materializing this matrix or trusting the empirical term alone. Structured student restrictions may help identification, but the sample count by itself does not provide it.

The next fitting comparison should test this combination against the existing covariance-only objective, preserving all six reads, original coefficient fidelity, every component, and fresh intervention requirements. Prior empirical overfitting results remain a reason for that control. No native fit under this proposed combined metric has yet run.

## Evidence

- [Native value-geometry audit and Gaussian controls](../../direct_tensor_match/VALUE_METRIC_GEOMETRY_V1.json).
- [Five equal-covariance lifted-metric controls](../../direct_tensor_match/LIFTED_METRIC_COVARIANCE_V1.json).

The full circuit goal remains open: this diagnoses the fitting objective, not semantic identity, standalone extraction or successful selective manipulation.
