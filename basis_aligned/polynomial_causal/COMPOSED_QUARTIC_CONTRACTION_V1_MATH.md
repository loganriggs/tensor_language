# A weight-only loss for the genuinely composed two-bilinear function

11September2026. New implementation for the deeper producer/producer route in [Logan's proposal](explanations/for_logan/interaction_path_decomposition_proposal_2026-09-11.md). Existing [joint-router polynomial algebra](joint_router_polynomial_gram_v1.py) already implements dense quartic symmetrization and exact small-matrix inner products; this work reuses it as the independent control. The new part is an efficient factorized contraction for the full two-bilinear composition and its coefficient-loss gradient, without expanding the quartic tensor.

## Why this changes the object we can fit

Treating MLP16's native products as independent input coordinates can obscure relations that hold because those products are all computed from the same residual. Fitting the fully composed function lets those identities and cancellations participate in the objective. It does not automatically preserve arbitrary interventions on every original native product. The chosen extraction/intervention boundary must remain explicit.

Let the symmetric bilinear maps for the two homogeneous MLP numerators be

$$
P(u,v)=\frac{\lambda D_{16}}{2}
\left[(L_{16}u)\odot(R_{16}v)+(L_{16}v)\odot(R_{16}u)\right],
$$

$$
Q(s,t)=\frac{D_{17}}{2}
\left[(L_{17}s)\odot(R_{17}t)+(L_{17}t)\odot(R_{17}s)\right].
$$

Then the pure producer/producer contribution is the quartic function

$$
f(x)=Q(P(x,x),P(x,x)).
$$

Here $x$ is the previous MLP's normalized input and $\lambda$ is the actual block17 residual coefficient. This is the homogeneous degree4 numerator term. Bias produces additional lower-degree terms. Remaining residual, attention, MLP input normalization and the final readout normalization/capping are not included in this polynomial and must remain explicit in a complete executable model.

## Exact four-linear contraction

The unique fully symmetric four-linear coefficient tensor associated with $f$ satisfies

$$
\begin{aligned}
T(x_1,x_2,x_3,x_4)=\frac13\big[&Q(P(x_1,x_2),P(x_3,x_4))\\
+&Q(P(x_1,x_3),P(x_2,x_4))\\
+&Q(P(x_1,x_4),P(x_2,x_3))\big].
\end{aligned}
$$

Each term is a pairing of the four slots. The six previous-layer pair outputs are shared intermediates, and only three final bilinear evaluations are needed. In particular $T(x,x,x,x)=f(x)$. No vocabulary-by-four-input-coordinate array is materialized.

This formula is symmetric across all four copies of the same input. It is different from a separately symmetric query/key tensor, whose two semantic input types cannot be freely permuted. The full-unembedding metric can be incorporated by replacing $D_{17}$ with $JD_{17}$ where $J^\top J=U^\top U$.

## An unbiased coefficient norm objective

Take four **independent** random vectors with identity covariance. For any fixed coefficient tensor difference $E=T-\widehat T$,

$$
\begin{aligned}
\mathbb E\|E(x_1,x_2,x_3,x_4)\|^2
&=\sum_{o,i,j,k,l}\sum_{i',j',k',l'}
E_{oijkl}E_{oi'j'k'l'}
\delta_{ii'}\delta_{jj'}\delta_{kk'}\delta_{ll'}\\
&=\|E\|_F^2.
\end{aligned}
$$

The Kronecker deltas follow from independence across slots and identity covariance within a slot. Standard Gaussian entries and independent random signs both satisfy this condition. Using the same random vector in all four slots instead gives a different moment-weighted function norm; it is not this coefficient estimator.

Consequently, the mean squared difference of factorized contractions is an unbiased weight-coefficient loss for a **fixed** candidate. It remains a stochastic estimate with sampling variance. If a candidate is optimized on a finite fixed probe set, its training estimate is not an unbiased audit of the selected candidate's error; independent probes are required for evaluation. Fresh optimization batches and separate audit seeds avoid silently making probe memorization the objective.

The kernel supports differentiation through the candidate's factors. It does not specify the final sparse model or solve its optimization problem. Its role is to make a genuinely composed-function comparison possible when exact full tensor contraction is too expensive.

A single pair-partition tensor, $Q(P(x_1,x_2),P(x_3,x_4))$, is not yet fully symmetric. Symmetrizing is an orthogonal projection in coefficient space, so its coefficient norm cannot increase. A measured norm reduction is mathematical redundancy removal, not on its own evidence of learned sparsity or semantic circuits.

## Completed controls and native price check

[CPU controls](COMPOSED_QUARTIC_CONTRACTION_V1_CONTROL.json) compare the implementation against an explicit small symmetric tensor, all ordered standard-basis contractions, sequential diagonal execution and gradients through all six factor matrices. Maximum execution error8.64e-16; coefficient norm error2.22e-16; maximum gradient error4.53e-15. The8192-probe Gaussian illustration falls0.665estimated standard errors below its known exact norm; this is a finite-sample diagnostic, not a probabilistic guarantee.

[Native preregistration](COMPOSED_QUARTIC_NATIVE_V1_PREREGISTRATION.md) uses4096Gaussian and4096random-sign probes to measure full-U coefficient norm uncertainty and actual evaluation cost. It is queued separately behind the live shared/independent sparse path fit. Check the runner or result for its latest status. It is not a native sparse fit and does not advance behavioral certification by itself.

[Kernel](composed_quartic_contraction_v1.py) · [CPU control](check_composed_quartic_contraction_v1.py) · [Native runner](../bilinear_quotient/ops/run_composed_quartic_native_v1.py).

## Native oracle result and interpretation

[Native check](COMPOSED_QUARTIC_NATIVE_V1_RESULT.json) completed22:25:30, A/B/C held. Gaussian/random-sign4096-probe estimates are3.6549e20/3.6482e20, with estimated relative standard errors0.291/0.255%. Each distribution plus its unsymmetrized pair reference takes about1.45seconds. Full run3.61seconds, peak allocatedGPU memory493MB. Native diagonal error3.74e-16; permutation replay0. These measurements price the reference objective, not the backward pass or a full sparse optimization.

The observed fully symmetric/pair-partition norm-squared ratios0.33481/0.33432 are near1/3. For exchangeable pairings with common norm-squared $N$ and pair inner product $C$, the expectation ratio is $(1+2C/N)/3$. [Executed descriptive accounting](COMPOSED_QUARTIC_PAIRING_ACCOUNTING_V1.json) gives inferred $C/N$ around0.00221/0.00149. Ratio covariance was not retained, so no sign/significance claim follows. Generic near-orthogonality can explain the reduction; this is not a discovered sparse circuit or evidence that the learned composition has a special cancellation.
