# Fixed-value cubic paths reduce to a weighted matrix SVD

12 September 2026. This follows the requested equation-specific ROI review.
It uses the already identified value factor of a producer; it is not automatic
value-factor discovery and does not replace the original QK normalizers.

## Exact support delivered an actual CPU gain

[Cubic kernel benchmark](CUBIC_SUPPORT_KERNEL_V1_BENCHMARK.json) compares the
same rank16 candidate and existing objective in the full versus exact union
support of heads8.2/9.8. Source coordinates520, query512. Value and gradient
relative errors are3.5e-16 and2.8e-15. Five alternating two-thread CPU trials
have median gradient costs69.06ms versus9.27ms, a7.45×ratio. Full execution
includes lifting latent readers; reduced execution excludes one-time basis
construction. No GPU speedup or smaller extracted package is inferred.

## A structure with a globally solvable restricted fit

For a fixed source value reader v, a producer pair numerator has the form

$$
p(q,s)=(v^Ts)(q^TAs)(q^TBs),
$$

where source s is in its exact reduced support, q is in its reduced query
support, and A,B include the two score maps and a fixed relative rotary
position. For the known selective reflection-even component, sum two such
terms, using inside/inside and outside/outside key maps. The same v is shared.

Instead of three freely optimized source readers per component, try

$$
\widehat p(q,s)=(v^Ts)\sum_{r=1}^R
(q^TU_rq)(s^TX_rs),
\qquad U_r=U_r^T,\quad X_r=X_r^T.
$$

This permits a quadratic to read multiple directions. Rank R counts separated
query/source quadratics, not CP rank. These matrices cost more than vectors;
a rank-matched CP comparison is not parameter matched.

The coefficient norm induced by multiplication with v must be retained. For
symmetric X define its cubic coefficient tensor

$$
L_v(X)_{ijk}=\frac{v_iX_{jk}+v_jX_{ik}+v_kX_{ij}}3.
$$

Direct expansion of the nine products gives

$$
\langle L_v(X),L_v(Y)\rangle_F
=\frac{\|v\|^2}{3}\langle X,Y\rangle_F
+\frac23(Xv)^T(Yv)
=\langle X,M_v(Y)\rangle_F,
$$

$$
M_v(X)=\frac{\|v\|^2}3X+
\frac{(Xv)v^T+v(Xv)^T}3.
$$

Thus a plain QK matrix SVD is generally the wrong coefficient weighting once
the value factor is included. Let u=v/||v|| and P=uu^T. The three orthogonal
parts of a symmetric matrix are

$$
X_\parallel=PXP,\qquad
X_\times=PX+XP-2PXP,\qquad
X_\perp=(I-P)X(I-P).
$$

M has eigenvalues ||v||²,2||v||²/3,||v||²/3 on these three spaces. Consequently
M^(1/2) and M^(-1/2) are explicit inexpensive operations; no large metric matrix
needs to be built. The nonzero-v assumption is required.

Define the query/source quadratic operator

$$
\mathcal T(X)=\operatorname{sym}(AXB^T),
\qquad \operatorname{sym}(Z)=(Z+Z^T)/2.
$$

In orthonormal symmetric-matrix coordinates the weighted operator is

$$
\mathcal A=\mathcal T M_v^{1/2},
\qquad
\mathcal A^*(Y)=M_v^{1/2}
\operatorname{sym}(A^T\operatorname{sym}(Y)B).
$$

Take a matrix SVD of $\mathcal A$. If W_r is a right singular matrix and V_r is its left
singular matrix, use X_r=M^(-1/2)(W_r) and U_r=sigma_r V_r in the polynomial
above. Truncating the SVD is a global best rank-R approximation **within this
fixed-value separated-quadratic family**, measured in symmetric cubic source
and quadratic query coefficient norm. The sum of discarded squared singular
values is the coefficient error. This does not optimize the value reader,
identify semantic units, or solve the general CP/DAG problem. Repeated singular
values identify a subspace rather than unique individual matrices.

## Executed discriminating control

The [independent dense control](FIXED_VALUE_QUADRATIC_OPERATOR_V1_CONTROL.json)
uses5source dimensions,3query dimensions and rank2. It constructs every cubic
coefficient explicitly, separately from the implicit operator.

- Metric identity relative error1.95e-16.
- Operator/adjoint inner-product absolute discrepancy3.55e-15.
- Actual truncated cubic error versus singular-value tail discrepancy1.49e-16
  relative to target squared norm.
- Polynomial factor execution versus explicit tensor execution error4.04e-16.

All registered control bars hold. No native spectral convergence or behavior
result has been measured for this new family yet.

## Native application and decision

Start with head9.8's known current-value sector and its complete or already
selective joint-key numerator. Its source support includes K1,K2 and v, at
most257 dimensions; query support at most256. An operator on symmetric matrices
has n(n+1)/2 source and m(m+1)/2 query coordinates. At257/256 these are
33,153 and32,896. Materializing that matrix would use about8.72GB inFP64.
Use matrix-vector products and an iterative singular solver, with explicit
singular-pair residuals and an honest unconverged label on time limits.

For new relative rotary positions keep the frozen source quadratics and
recompute their query coefficient matrices. Original native denominators remain
supplied at execution. A fit at one relative position does not establish all
position fidelity; a held-position check and physical intervention screen are
required. The useful circuit decision is whether stable quadratic blocks can
replace the identified joint-key/value computation and preserve its selective
removal or swaps. Better coefficient error alone is not promotion.
