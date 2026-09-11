# Sparse interactions in the composed two-bilinear function

11 September 2026. Implements the next stage of [Logan's composed-path proposal](explanations/for_logan/interaction_path_decomposition_proposal_2026-09-11.md). The first native optimization is managed-live; no final capture or circuit claim yet.

## What is being factored

The [existing quartic oracle](COMPOSED_QUARTIC_CONTRACTION_V1_MATH.md) supplies the fully symmetric tensor $T$ for

$$
f(x)=UQ(P(x,x),P(x,x)).
$$

$P$ is MLP16's homogeneous bilinear numerator including the block17 residual coefficient; $Q$ is MLP17's homogeneous bilinear numerator. The input $x$ is the normalized MLP16 input. The target is one degree-four interaction path. Bias, remaining residual, attention, downstream input RMS and final RMS/tanh remain explicit dependencies in a complete model. They have not been absorbed into a globally polynomial approximation.

The new fit preserves the fact that MLP16's product coordinates all arise from the same $x$. Earlier formal-port fits did not optimize this quartic object. That distinction permits cancellations and identities in the composition to affect the loss.

## Exact sparse core on learned input directions

Let $B=[b_1,\ldots,b_r]\in\mathbb R^{1152\times r}$ with $B^\top B=I$. For each sorted four-tuple $\alpha=(i,j,k,l)$, define multiplicity

$$
M_\alpha=\frac{4!}{\prod_a n_a!},\qquad
\psi_\alpha(x)=\sqrt{M_\alpha}\,(b_i^\top x)(b_j^\top x)(b_k^\top x)(b_l^\top x).
$$

The associated coefficient tensor is the sum of its distinct ordered outer products divided by $\sqrt{M_\alpha}$. These tensors are orthonormal in coefficient Frobenius norm. For example, an all-distinct tuple has24 ordered copies, a two-pair tuple has6, and four identical indices have1. This normalization is essential; treating all monomials equally would optimize a different norm.

Let $J^\top J=U^\top U$ and let $T_J$ have its physical output multiplied by $J$. Its exact coefficient along this feature is

$$
c_\alpha=\sqrt{M_\alpha}\,T_J(b_i,b_j,b_k,b_l).
$$

For a fixed support $S$, the optimal output coefficients are exactly these $c_\alpha$ and

$$
\left\|T_J-\sum_{\alpha\in S}c_\alpha\otimes\Psi_\alpha\right\|_F^2
=\|T_J\|_F^2-\sum_{\alpha\in S}\|c_\alpha\|^2.
$$

Thus fixed-bank support selection is exact: keep the largest $K$ coefficient-vector energies. Only the input bank requires continuous optimization. Physical writers are $J^{-1}c_\alpha$; vocabulary coordinates are never enumerated per monomial during fitting.

The [kernel](sparse_quartic_core_v1.py) computes all $r(r+1)/2$ previous-layer quadratic pair outputs once, reads those through both MLP17 factors, and combines the three pairings of each four-tuple. It reuses the fully symmetric oracle's algebra without constructing a $1152^4$ input tensor. During a fixed-support gradient step it forms only selected quartic coefficient columns; full columns are needed for support selection.

At $r=16$ there are3876 possible monomials. The pilot keeps128, with165888 fitted floats. The restriction is strong: all interactions must lie within one sixteen-dimensional input subspace, and only128 monomials in its chosen orthonormal coordinates survive. This is an initial sparse symmetric Tucker-style core with dense output vectors. It does not cover arbitrary low-rank quadratic factors, nonorthogonal overcomplete dictionaries, overlapping input blocks, or general arithmetic DAGs. A poor fit cannot reject those alternatives.

## Optimization and controls

[Preregistration](SPARSE_QUARTIC_NATIVE_V1_PREREGISTRATION.md) fixes two weight-informed starts, exact conditional output updates, alternating support selection and Stiefel CG, stopping criteria, cost and interpretation. The objective is deterministic coefficient energy, not a fit to sampled activations. The previously measured native norm is used to report a capture fraction; its sampling uncertainty remains. It is not needed to select coefficients or compute gradients.

[Controls](SPARSE_QUARTIC_CORE_V2_CONTROL.json) compare coefficients and gradients through all six native factor matrices plus the input bank with the independent four-linear contraction, then check exhaustive small-tensor norms, diagonal execution and sparse projection error. Maximum discrepancy is $1.72\times10^{-15}$. A syntax error in the first control script occurred before execution and is [preserved](SPARSE_QUARTIC_CORE_V1_CONTROL_FAILURE.json); V2 fixes that expression without changing the kernel.

Both near-planted starts recover100% of the known two-term function. Both cold starts finish at68.75%, one meeting the tight local gradient criterion. This is a demonstrated local optimization trap, not evidence that the planted sparse structure is absent. The near starts' $10^{-8}$ stopping test misses despite exact function recovery at displayed precision; that numerical stopping limitation is also retained. The native pilot uses its independently preregistered $10^{-7}$ normalized tangent threshold and a relative stationarity test.

Early native progress shows large gains over initialization and multiple support changes. These are intermediate observations, not final results. Because the objective is scaled by initial energy, large gains also enlarge its absolute numerical gradient floor; the relative gradient and line-search stopping reasons must be inspected before interpreting a failed convergence predicate. Do not rewrite the running experiment's criteria.

## An executable shared-product DAG

For a fixed monomial $(i,j,k,l)$, multiplication can be scheduled as any of

$$
(a_i a_j)(a_k a_l),\qquad
(a_i a_k)(a_j a_l),\qquad
(a_i a_l)(a_j a_k),\qquad a=B^\top x.
$$

Different monomials may reuse the same quadratic pair. The [compiler](quartic_pair_dag_v1.py) finds the minimum number of distinct quadratic pairs among these $2+2$ schedules. Let $z_p$ indicate that pair $p$ is computed and $y_{e,s}$ indicate the schedule selected for monomial $e$. It solves

$$
\min\sum_p z_p,\qquad
\sum_s y_{e,s}=1,\qquad
y_{e,s}\le z_p\quad\text{for each pair used by schedule }s,
\quad y,z\in\{0,1\}.
$$

There is one further multiplication per quartic monomial. The solver uses SciPy's [HiGHS MILP interface](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html), records its bound/gap, and distinguishes an optimal result from a time-limited incumbent. This is a finite combinatorial restriction; it does not establish the minimum arbitrary arithmetic circuit or search distributive rewrites involving sums.

[Exhaustive planted controls](QUARTIC_PAIR_DAG_V1_CONTROL.json) confirm that five monomials can share three quadratic nodes: eight products instead of fifteen independent products. The MILP optimum equals exhaustive enumeration. Signed output execution and shared-node deletion agree with direct monomial/incident-edge evaluation within $2.40\times10^{-16}$. Two-node removal uses inclusion–exclusion, preserving overlap exactly; a shared descendant is not counted twice.

This supplies an explicit reuse/removal interface for a future fitted program. Its quadratic nodes are learned-reader products, not yet semantic computations. Removing a node zeros its descendants while holding the declared source/normalization background fixed. Replacing source activations would be a different intervention and must recompute affected native dependencies.

Native graph construction awaits the completed frozen fit. Subsequent coefficient stability and fresh native intervention tests must establish whether these computational nodes predict, extract, selectively remove, or compose across behaviors. Algebraic execution alone proves none of those behavioral properties.

## Completed pilot and representation diagnosis

[Pilot](SPARSE_QUARTIC_NATIVE_V1_RESULT.json) finished23:19:13:0.124815/0.130388%estimated full coefficient capture. Original absolute-scale convergence misses; a separate [unit-energy gradient check](SPARSE_QUARTIC_SCALE_V1_RESULT.json) establishes local stationarity without moving factors or support. The [dense fixed-bank ceiling](SPARSE_QUARTIC_CEILING_V1_RESULT.json) is only0.137852/0.148640%; current128edges retain88–91%of that. These particular input spaces, not their sparse support alone, omit most native coefficient structure. Broader input representations remain open.

[Gram hierarchy method and results](QUARTIC_GRAM_HIERARCHY_V1_MATH.md) develops an additional route: optimize equivalent quadratic-pair matrices before extracting signed squared-quadratic intermediates. All four selected centered native output modes converge under the convex surrogate. Eight-term errors improve from26–38%with canonical matricization to0.8–1.3%with the optimized representation. This is structure inside projected scalar functions, not global model coverage or circuit certification.
