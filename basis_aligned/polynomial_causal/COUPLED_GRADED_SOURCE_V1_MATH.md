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


## Executed local-minimum control and a way to test escape (08:31)

The [landscape control](GRADED_SOURCE_LANDSCAPE_CONTROL_V1_RESULT.json) gives an exact counterexample to interpreting local convergence as global recovery for this very objective. In two dimensions choose

$$
A_0=\operatorname{diag}(1,-\sqrt\alpha),\quad
A_1=\operatorname{diag}(1,\sqrt\alpha),\quad H=I,
\qquad 0<\alpha<1.
$$

The scalar branch is $x_1^4-\alpha x_2^4$. For rank-one producer direction $p=(\cos\theta,\sin\theta)$, degree-balanced retained fraction is

$$
1-L(\theta)=\frac{
\sum_{k=1}^3(\cos^{2k}\theta+\alpha^2\sin^{2k}\theta)
+(\cos^4\theta-\alpha\sin^4\theta)^2
}{4(1+\alpha^2)}.
$$

Each numerator term is at most1, so $p=e_1$ attains a global minimum, with loss $\alpha^2/(1+\alpha^2)$. But $p=e_2$ is also a strict local minimum whenever $\alpha^2>1/10$:

$$
L(e_2)=\frac1{1+\alpha^2},\qquad
L''(\pi/2)=\frac{10\alpha^2-1}{2(1+\alpha^2)}>0.
$$

At $\alpha=.8$, the losses are0.390244 and0.609756. Both stationary gradients vanish; the bad minimum's curvature is1.64634. The same QR/Armijo rule as the native job reaches the corresponding basins from two nearby starts. The bad-basin run reaches gradient2.43e-8 but misses its deliberately stricter1e-8 stopping target within1000iterations; the analytic stationary point itself has zero gradient. Both are below the native1e-5 criterion. This is a proved limitation of stationarity, not evidence that the live native fit is in this particular trap.

A constructive follow-up is [one-plane search](GRADED_PROJECTION_PLANE_V1_CONTROL.json). Replace one retained direction $v$ by $v\cos\theta+w\sin\theta$, where $w$ lies in the excluded orthogonal space. The projector depends linearly on $\cos2\theta$ and $\sin2\theta$. Its quartic coefficient norm therefore has Fourier frequencies at most4 in $\phi=2\theta$. Nine equally spaced evaluations recover those coefficients. Multiplying the derivative Laurent polynomial by $z^4$, $z=e^{i\phi}$, yields a polynomial of degree at most8; unit-circle roots give stationary-angle candidates.

The executed CPU control starts at the bad minimum and finds the good one, reducing loss by0.219512. Thirteen additional direct evaluations agree with the interpolation within4.44e-16. This is global search on a **specified two-dimensional rotation plane**, not over all rank128subspaces. Floating-point roots remain candidates requiring interpolation and direct-objective verification; the helper is not a general interval certificate. There are many possible native planes, so failing to find an escape in a small panel would not certify native global optimality. A weight-only plane panel is a concrete post-fit red-team option, preserving the original fit and behavioral verdicts rather than silently replacing them.


## A bound for every frame at the fixed rank (08:39)

The [CPU rank-bound audit](GRADED_SOURCE_RANK_BOUND_CPU_V1_RESULT.json) distinguishes representation capacity from optimization, without text fitting. For degree group $k$, let $N_k(Q)$ be the retained norm polynomial with a symmetric producer-slot operator $Q$ in every producer slot. At $Q=I$,

$$
R_k=\frac1k\nabla_QN_k(I),\qquad
S=\frac14\sum_{k=1}^4\frac{R_k}{N_k(I)}.
$$

$R_k$ is the covariance from contracting every index except one producer index. It is positive semidefinite, with trace $N_k(I)$; consequently $S$ is positive semidefinite with trace1. For an orthogonal projector $Q=PP^T$, projecting every producer slot retains no more norm than projecting only one slot. Hence

$$
1-L(P)\leq\operatorname{tr}(P^TSP)
\leq\sum_{i=1}^{128}\lambda_i(S),
\qquad L(P)\geq1-\sum_{i=1}^{128}\lambda_i(S).
$$

One weighted reverse-mode derivative computes $S$ without a large tensor. The2.28second CPU audit gives native rank128loss lower bound0.480319, trace error below1.2e-14, and positive minimum eigenvalue. On the analytic two-dimensional trap, the same bound equals the known global minimum0.390244. For the native model, the gap between this lower bound and current~0.684fits is unresolved: the bound does not prove that a0.480solution exists, nor certify current near-optimality. It applies only to this fixed common-projector representation and coefficient metric, not arbitrary tensor programs or causal-effect error.

The [native plane audit](GRADED_NATIVE_PLANE_AUDIT_V1_PREREGISTRATION.md) is implemented but must await terminal fit interpretation. It tests four random and four weight-gradient-selected planes per arm, preserving original frames and verdicts. Separately, the [existing LBFGS adapter control](GRADED_LBFGS_ADAPTER_CONTROL_V1_RESULT.json) passes on a planted common subspace: coefficient loss8.91e-15, gradient2.00e-7, orthogonality4.97e-16 in12updates. This reuses the project's existing limited-memory manifold optimizer, with no new optimizer framework. It supports a same-objective convergence continuation, not a guarantee against local minima.


## First two-start fit is terminal; convergence remains unresolved (08:43)

[GRADED_SOURCE_FIT_V1_RESULT.json](GRADED_SOURCE_FIT_V1_RESULT.json) records1082seconds of managed execution. Both540second arms hit their time limits: spectral loss0.68397897/gradient1.558e-4, independent loss0.68396756/gradient2.740e-5. A/B/C all fail as registered. The spectral arm improves its coefficient objective about4.87%, below10%. Both frames preserve active-family signs, but fail joint magnitude fidelity. Branch3 adverb/progressive errors are17.54/23.04%in the first arm and15.64/20.95%in the second; branch8 gerund/progressive also miss. Full write errors remain only2.7–2.8%forbranch3 and5.2–6.6%forbranch8, again showing why write reconstruction alone is insufficient.

The original results and frames remain immutable. A [same-objective LBFGS continuation](GRADED_SOURCE_LBFGS_V1_PREREGISTRATION.md) has been submitted, with300seconds perarm, exact final gradients, the original loss baseline and original behavioral thresholds. The separate plane audit is held pending its outcome. Small differences between unconverged objective values do not certify unique computation, native stability or absent structure.
