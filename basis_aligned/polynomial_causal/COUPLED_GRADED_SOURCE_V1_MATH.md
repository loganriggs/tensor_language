# Coupled products and explicit source-degree weighting

12 September 2026, 08:24. This follows the [shared MLP15 interface and its failed joint fidelity test](COMPOSED_WEIGHT_COMPARISON_V1_RESULTS.md#a-common-mlp15-interface-is-useful-but-not-yet-sufficient-0812). The new computation is weights-only. Text remains reserved for validation of frozen frames.

## Why change the objective?

The previous spectral solution approximated three quadratic readers separately. Actual branches multiply a shared reader by a private partner. Exact error accounting showed that different reader errors dominate different task failures. Reweighting the readers from those validation outcomes would break the weights-first discovery rule. Instead, preserve the two complete products, with their actual full-unembedding writers, in an explicitly stated coefficient metric.

Let $A_0,A_3,A_8$ be the frozen symmetric input forms, $H$ the MLP15 nonconstant producer coefficient Gram, and $S=H^{1/2}$. Write the **formal** source as $y=(b,z)$, with $b$ the background including producer bias and $Sz$ the nonconstant producer output. Define

$$
T=[I\ S],\qquad B_j=T^TA_jT,
\qquad \mathcal F=\sum_{j\in\{3,8\}}Uw_j\otimes\operatorname{Sym}(B_0\otimes B_j).
$$

The metric uses the full $U$, including common vocabulary components. Its two-by-two writer Gram is $W^TU^TUW$. This keeps complete branch rescaling symmetries intact. Native normalization and residual background remain explicit execution ports.

## Exact projection identity removes the huge tensor

For $P^TP=I$, let $J=\operatorname{diag}(I,PP^T)$. Using the shared producer interface replaces the coefficient tensor by $J$ applied to every source-input slot:

$$
\widehat{\mathcal F}=\mathcal F\times_1J\times_2J\times_3J\times_4J.
$$

Since $J$ is an orthogonal projector, so is this four-slot operation. Therefore

$$
\langle\mathcal F,\widehat{\mathcal F}\rangle
=\|\widehat{\mathcal F}\|_F^2,
\qquad
\|\mathcal F-\widehat{\mathcal F}\|_F^2
=\|\mathcal F\|_F^2-\|\widehat{\mathcal F}\|_F^2.
$$

All contractions reduce to $K=I+SPP^TS$. With $C_j=A_jK$ and $t_{ij}=\operatorname{tr}(C_iC_j)$, the branch coefficient Gram is

$$
G_{ij}(K)=\frac{t_{00}t_{ij}+t_{0i}t_{0j}
+4\operatorname{tr}(C_0C_0C_iC_j)}6.
$$

Contract this with the writer Gram to obtain the retained norm. This uses ordinary $1152\times1152$ matrices instead of a $2304^4$ source tensor. The existing quartic symmetrization identity is reused; the new reduction concerns the common source projector. Dense tensor, projection, loss and tangent-gradient controls pass below $10^{-9}$. The gradient comparison is on the Grassmann tangent space: the simplified norm extension off orthonormal frames need not have the same ambient derivative as a direct tensor implementation.

The [native price check](COUPLED_SOURCE_PROJECTION_NATIVE_V1_RESULT.json) passes all bars: approximately 0.085 seconds per gradient evaluation, 0.76 GiB peak allocation, full-rank replay $1.25\times10^{-15}$, tangent finite differences below $6.13\times10^{-9}$. A small descent step improves the coupled objective. None of these checks is a fitted circuit result.

## The raw metric has a consequential scale bias

The native source Gram has a large coefficient scale. The full raw norm is $2.729\times10^{38}$, compared with $3.984\times10^{17}$ for background alone. Merely subtracting the background constant does not balance mixed interactions. We measured every degree group before launching optimization.

Partition the quartic coefficient tensor by the number $k$ of producer slots, $k=0,\ldots,4$. These groups are orthogonal in formal coefficient space. If

$$
N(t)=\|\mathcal F\text{ with producer scaled by }\sqrt t\|_F^2,
\qquad N(t)=\sum_{k=0}^4c_kt^k,
$$

then $c_k$ is the squared norm of degree group $k$. The [native graded check](GRADED_SOURCE_PROJECTION_NATIVE_V1_RESULT.json) finds raw norm fractions approximately

$$
(1.46\times10^{-21},\ 9.32\times10^{-16},\ 2.25\times10^{-10},\ 2.44\times10^{-5},\ 0.999975623).
$$

Thus 99.9976% is pure producer. This is not evidence that the mixed terms are unimportant on text; the source-intervention results already show substantial mixed computation. It is a consequence of this coefficient metric's units and inductive bias.

The new, separately declared objective is

$$
L(P)=\frac14\sum_{k=1}^4\left(1-\frac{\widehat c_k(P)}{c_k}\right).
$$

It gives each producer-dependent degree group equal **relative coefficient-error** weight. Pure background is preserved exactly. Rescaling the producer coordinates multiplies numerator and denominator of each ratio by the same factor, so the objective is invariant to that rescaling. Equal degree weight is an explicit modeling assumption, not uniquely prescribed by the model or selected from validation performance.

The implementation computes the five polynomial coefficients directly from matrix products; it does not estimate small coefficients by subtracting enormous norm evaluations. Dense per-degree norms, tangent gradients and source-scale checks pass below $1.6\times10^{-15}$. Native finite-difference error is $3.78\times10^{-9}$ and a gradient evaluation takes 0.127 seconds. Spectral128 starts at loss0.718990; spectral512 at0.275747. These are measured initializations, not optimized outcomes.

## What is being optimized, and what remains unproven

The [registered fit](GRADED_SOURCE_FIT_V1_PREREGISTRATION.md) uses rank128, spectral and independently random initializations, horizontal Grassmann gradients, QR retraction and Armijo descent. Each arm has540seconds and an explicit final gradient check; local convergence requires norm at most $10^{-5}$. A time limit is not convergence, and convergence is not global recovery. The managed job records terminal frames for further work. Inspect its live log/progress or terminal result before claiming a fit has finished.

Only after both frames are frozen will the runner evaluate original signed swaps and removals for both branches. It preserves the failed earlier thresholds and reports the quoted control. Its circuit-level decision is whether a common upstream interface preserves both computations. A lower coefficient loss alone does not pass that test.

Finally, exactness here is for a quartic on **independent formal background and producer coordinates**. The Gram $H$ represents quadratic producer functions through a paired coefficient metric. This is not the fully symmetrized eighth-order norm obtained by substituting the same native MLP15 input into every producer occurrence, nor a probability model of reachable text states. The native model ties these sources and includes RMS denominators. Those distinctions remain material even if this fit converges. The broader OOD, extraction, selective removal and composition/reuse goal stays open.
