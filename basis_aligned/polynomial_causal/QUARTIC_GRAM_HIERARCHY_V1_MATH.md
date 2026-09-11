# Finding quadratic intermediates without mistaking a matrix gauge for structure

11 September 2026. Follow-up to the [sparse composed quartic fit](SPARSE_QUARTIC_CORE_V1_MATH.md). Weights only; no native text fitting or semantic circuit claim.

## First resolve the sparse fit's limitation

[Native pilot](SPARSE_QUARTIC_NATIVE_V1_RESULT.json) finished23:19:13 after596seconds. The two fits capture0.124815% and0.130388% of the estimated full quartic coefficient energy. Initial-to-final gains are1142 and2854, reflecting weak initialization; they are not evidence of high final coverage. Instrument and gain predictions pass, the original absolute-gradient convergence prediction fails.

The [completed fixed-bank ceiling test](SPARSE_QUARTIC_CEILING_V1_RESULT.json) passes all three predictions. Keeping every quartic interaction in the learned sixteen-dimensional input spaces would capture only0.137852% and0.148640%. The selected128edges already retain90.54% and87.72% of those projected energies. Estimated error outside each input space is7660 and5470 times its discarded inside-space energy. These are fixed-bank ceilings, not a theorem about all16-dimensional subspaces. The full coefficient norm remains an estimate with0.291% estimated relative standard error.

This argues against spending the next major fitting budget merely adding monomials within these banks. Larger, overlapping, or output-specific input spaces are materially different alternatives. A small coherent component could still be a circuit despite accounting for little global coefficient energy; that requires native behavioral evidence.

The [separate scaling check](SPARSE_QUARTIC_SCALE_V1_RESULT.json) shows that the convergence miss arose from the absolute objective scale. The original objective divided energy by its tiny initial value. Replacing that fixed divisor with the final selected energy changes only scale, not the function, support, or maximizers. Gradient rescaling replays within2.36e-15. Both unchanged fits meet a unit-energy tangent threshold of1e-7, at5.13e-8 and4.54e-8; no update is taken and the top128supports remain unchanged. Thus we have local stationarity under an explicit meaningful normalization. The original failed predicate remains in its receipt; no global recovery claim follows.

## Why a quartic's pair matrix is not unique

Set $a=B^\top x\in\mathbb R^r$ and form coefficient-normalized quadratic features

$$
z_{ii}(a)=a_i^2,\qquad z_{ij}(a)=\sqrt2\,a_i a_j\quad(i<j).
$$

A scalar quartic can be written $f(a)=z(a)^\top A z(a)$, with $A$ a symmetric $n_2\times n_2$ matrix, $n_2=r(r+1)/2$. Its eigendecomposition gives

$$
f(a)=\sum_j\lambda_j\big(q_j^\top z(a)\big)^2.
$$

These are quadratic intermediates followed by squares and signed output weights. An intermediate can combine many input products. Positive and negative $\lambda_j$ are both allowed; token-writing functions need not be nonnegative.

The monomials in $z$ are algebraically dependent. [Parrilo's Gram-matrix formulation, section3.2](https://web.mit.edu/~a_a_a/Public/Publications/refs_for_seb_blog/Parrilo_MathProg.pdf) makes this explicit: coefficient matching defines an affine family of Gram matrices, and positive-semidefinite feasibility characterizes a sum-of-squares representation. Our task uses the same coefficient-matching family but permits indefinite matrices and seeks economical signed intermediates. A failed positive-semidefinite search would not rule out such a representation, and nuclear minimization is not the paper's sum-of-squares feasibility theorem.

Let $\mathcal L(A)$ be the vector of normalized quartic coefficients. If quadratic pairs $p,q$ together have sorted four-tuple $\alpha$, their contribution is

$$
[\mathcal L(A)]_\alpha
=A_{pq}\frac{\sqrt{m_p m_q}}{\sqrt{M_\alpha}},
\qquad m_{ii}=1,\quad m_{ij}=2.
$$

Here $M_\alpha$ is the quartic multiplicity from the sparse-core note, and the sum includes ordered matrix entries. Our [map and adjoint](quartic_gram_map_v1.py) satisfy

$$
\mathcal L\mathcal L^*=I,\qquad
A_{\mathrm{canonical}}=\mathcal L^*c,\qquad
\{A:\mathcal L(A)=c\}=\mathcal L^*c+\ker\mathcal L.
$$

The canonical fully symmetric matricization is the minimum-Frobenius-norm representative. It need not have minimum rank. At$r=16$, symmetric Gram matrices have9316 coordinates but quartics have3876; the nullspace has5440 dimensions.

[Executed example](QUARTIC_GRAM_MAP_V1_CONTROL.json): for$f(a)=a_0^2a_1^2$, canonical Gram eigenvalues are$(-1/6,1/6,1/3)$, rank3. The alternative matrix having only$A_{01,01}=1/2$ has rank1 and exactly the same polynomial. Thus a canonical pair unfolding can look unnecessarily complicated. Coefficient, adjoint, norm and diagonal replay tests pass within2.59e-16. This invalidates a proposed rank lower bound; it does not prove that the native quartic has a small hierarchy.

## A convex search over equivalent representations

We implement the surrogate

$$
\min_{A=A^\top}\|A\|_*\quad\text{subject to}\quad\mathcal L(A)=c.
$$

For symmetric matrices the nuclear norm is$\sum_j|\lambda_j|$. Minimizing it favors economical signed-square representations in this fixed quadratic coefficient metric. It is not minimum rank or minimum arithmetic cost, nor does it force multiple output functions to reuse the same quadratic dictionary.

The affine projection is exact:

$$
\Pi_c(V)=V+\mathcal L^*(c-\mathcal L(V)).
$$

The nuclear proximal update soft-thresholds signed eigenvalues. Alternating these in scaled ADMM follows the standard splitting method described by [Boyd et al.](https://web.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf). Both terms are closed convex, the affine set is nonempty because the canonical representative exists, and finite-dimensional nuclear-norm level sets are bounded. A fixed positive penalty gives the usual convex optimization setting; this does not provide a finite-iteration rate or low-rank recovery theorem for our native functions.

Our dual check is explicit:

$$
\max_y\langle c,y\rangle\quad\text{subject to}\quad\|\mathcal L^*y\|_{\mathrm{op}}\le1.
$$

The current dual estimate is rescaled to satisfy this spectral bound. A feasible primal nuclear norm minus this dual value bounds remaining objective error. We also require primal and dual residuals to be small; a nominal iteration count is not convergence. Inputs are normalized to unit coefficient norm during solving and rescaled afterward.

[Toy controls](QUARTIC_GRAM_NUCLEAR_V1_CONTROL.json) recover the rank1 example in41iterations and a signed rank2 example in81iterations, from canonical ranks3 and10. Relative dual gaps are below3.7e-8. These positive controls do not guarantee native low rank.

## Native mode experiment and its current status

Before looking at its outcomes, we selected the first two **centered** output modes of each dense projected quartic, four cases total. This tests quadratic hierarchy inside those frozen weight-selected scalar functions, not all native computation. [First native solver run](NATIVE_QUARTIC_GRAM_NUCLEAR_V1.json) is feasible to about2.5e-15 but fails both registered effective-rank and convergence criteria. After4000iterations, relative dual gaps are0.00117–0.00198, and the exactly feasible matrices retain134–135 effective eigenvalues out of136 at the1e-6 cutoff.

There is nevertheless approximate structure: their nuclear norms fall to36–39% of canonical values, and a descriptive eight-eigenvector truncation has about0.8–1.3% coefficient error. This does not repair the original rank criterion, which also required error<=1e-4. Exact feasible rank is sensitive to the many small entries left before convergence; an approximate-error curve is more informative than pretending those entries are either zero or semantically meaningful.

[V2 solver](quartic_gram_nuclear_v2.py) keeps the same convex objective, warms the primal from V1, and balances the penalty against residuals for at most1000iterations before holding it fixed. It permits up to40000iterations per case, with the original prediction thresholds. This is a warm-primal continuation, not restoration of the unsaved V1 dual state. [Adaptive-branch control](QUARTIC_GRAM_ADAPTIVE_BRANCH_V1_CONTROL.json) triggers penalty changes and recovers the same rank1 optimum. The [native V2 receipt](NATIVE_QUARTIC_GRAM_NUCLEAR_V2.json), when present, gives the terminal result; do not infer completion from this plan.

Any extracted quadratic intermediate still needs a cross-start comparison in the original input space, prior-art/dossier checks, native execution with declared background, and fresh/OOD intervention tests. Neither a matrix factor nor a shared algebraic node is automatically one of the four-property circuits.

## Converged native result

[V2 native continuation](NATIVE_QUARTIC_GRAM_NUCLEAR_V2.json) completed with all three registered predicates passing. The four cases converge in roughly6900–9800iterations after finite penalty adjustment, with relative primal-dual gaps below1e-7 and exact coefficient feasibility near2e-16. The near-exact effective ranks are67,68,65,66 rather than canonical136; truncation at the registered eigenvalue cutoff gives coefficient error below4e-8. These ranks minimize neither component count nor arbitrary arithmetic complexity: convergence certifies the stated nuclear surrogate.

More useful for compact interpretation is the matched-rank approximation comparison:

| Seed / centered output mode | Canonical eight-term coefficient error | Optimized eight-term coefficient error |
|---|---:|---:|
|11511 /0|26.10%|0.827%|
|11511 /1|36.47%|1.255%|
|11512 /0|30.65%|1.026%|
|11512 /1|37.57%|1.320%|

Each term is a squared scalar quadratic with a signed output weight. Changing only the polynomial-equivalent Gram representation makes a fixed eight-term truncation substantially better. The optimized full matrices represent exactly the same projected scalar functions; their eight-term truncations are approximations, not exact rewrites. The V1 failure remains useful: stopping the solver early made effective rank look almost full, even though approximate structure was already visible.

At sixteen input readers, one such eight-term scalar program stores136quadratic coefficients per intermediate and8signed weights, then uses a single physical output writer; shared input readers and their native dependencies remain required. Quadratic matrices may themselves admit cheaper bilinear products, and different output modes may share intermediates, but neither saving is assumed here. Approximation errors in this table are coefficient norms, not token-logit or causal-effect errors. Cross-start correspondence and native validation are the next identification tests; there is no new promoted circuit.

The first [direct cross-start correspondence screen](QUARTIC_INTERMEDIATE_CORRESPONDENCE_V1.json) is now executed. It compares the sixteen terms per seed in the original ambient input space, including signed physical output writing under the centered unembedding metric. The total sixteen-term function cosine is0.91282 and passes its0.9screen, but only two one-to-one terms exceed0.9, below the registered four-match threshold. Instrument error is8.88e-16. Thus a compact group representation is more reproducible than its individual quadratic-square factors; these factors do not yet support stable circuit names. No text outcomes were used to choose them. This direct check also completed the hourly review's immediate workflow repair using existing artifacts and contraction algebra.
