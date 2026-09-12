# A spectral relaxation of input-block independence

12 September2026. CPU control executed while native direct rotation is running.
This addresses the demonstrated local-minimum problem; it does not change the
running experiment or its registered bars.

## Why this is a different optimization problem

The exact block objective can have bad strict local minima. The
[previous control](FULLU_BLOCK_OPTIMIZER_V1_PREREGISTRATION.md) stopped at0.2356
despite perfectly independent planted blocks. Its gradient vanished and its
local Hessian was positive. More steps at that point would not find the blocks.

We can relax the search for a projector into a generalized eigenvalue problem.
This preserves the exact normalized-cut value for every legitimate projector,
but permits additional matrices during the search. It is an application of the
commutant formulation of simultaneous block diagonalization described in
[Maehara and Murota](https://epubs.siam.org/doi/10.1137/090779966), with the
normalization and projector mapping derived below for our weight object.

## Exact mapping

Use the [full-output contraction](FULLU_INPUT_BLOCKS_V1_MATH.md):

$$
\Phi(X)=\sum_v S_vXS_v,\qquad K=\Phi(I),\qquad
\mathcal L(X)=KX+XK-2\Phi(X),\qquad \mathcal M(X)=KX+XK.
$$

Assume $K$ is positive definite; otherwise restrict to its supported subspace
and account separately for the excluded null directions. For a nontrivial
orthogonal projector $P$, define

$$
t=\operatorname{tr}K,\quad a=\operatorname{tr}(PK),\quad
c=a-\operatorname{tr}(P\Phi(P)),\quad X=P-\frac at I.
$$

$X$ is orthogonal to the trivial identity in the $\mathcal M$ inner product:

$$
\langle I,\mathcal M(X)\rangle_F=2\operatorname{tr}(KX)=0.
$$

Because $\mathcal L(I)=0$, $P^2=P$, and $\Phi$ is self-adjoint,

$$
\langle X,\mathcal L(X)\rangle_F=2c,\qquad
\langle X,\mathcal M(X)\rangle_F=2a\left(1-\frac at\right).
$$

Therefore the generalized Rayleigh quotient is exactly the registered cut:

$$
\frac{\langle X,\mathcal L(X)\rangle_F}
{\langle X,\mathcal M(X)\rangle_F}
=\frac{tc}{a(t-a)}.
$$

Minimizing this quotient over **all symmetric** $X$ orthogonal to identity
relaxes the projector constraint. The true smallest generalized eigenvalue
$\lambda_\perp$ is consequently a lower bound on every projector's cut,
including unbalanced ranks. This bound concerns orthogonal blocks in the
specified producer metric, not overlapping DAGs, nonorthogonal decompositions
or every possible polynomial representation.

The relaxed solution need not have the two eigenvalues of a centered projector.
Diagonalizing it and selecting half its eigenvectors gives a candidate, whose
actual cut must be recomputed. A low relaxed eigenvalue does not guarantee a
good rounded partition. A reliably established high lower bound would be more
decisive than repeated local-optimization failures.

## Executed evidence

[Code](normalized_commutant_relaxation_v1.py) and
[receipt](NORMALIZED_COMMUTANT_RELAXATION_V1_CONTROL.json) use the same8D planted
and dense families as the optimizer control, with no new favorable example.
They construct the entire36-dimensional symmetric operator, remove identity,
and solve the reduced generalized problem through symmetric whitening.

- Planted relaxation eigenvalue: $1.12\times10^{-16}$. Rounding recovers the
  exact4+4blocks, with cut within $1.12\times10^{-15}$ of zero. The negative
  sign at this numerical scale is roundoff, not negative physical coupling.
- Dense-family relaxation minimum:0.187618. The rounded cut is0.233657,
  compared with0.216647 for the best previous local fit. The relaxation bound
  is lower, as required; its rounding is not automatically the best fit.
- Projector/Rayleigh identity error is at most $1.09\times10^{-15}$;
  generalized eigenpair residual is at most $2.16\times10^{-16}$.

These checks solve the previously trapped planted example without restarting
local descent. They establish an executable alternative, not native structure.

## Native implementation and the certification boundary

Native symmetric dimension is664,128. Do not construct its dense operator.
If $K=V\operatorname{diag}(k_i)V^T$, the Sylvester square root is cheap:

$$
\mathcal M^{-1/2}(Y)=V\left[
\frac{(V^TYV)_{ij}}{\sqrt{k_i+k_j}}
\right]V^T.
$$

Combine this with the existing exact $\Phi$ kernel and remove the transformed
identity $\mathcal M^{1/2}(I)$. This gives a symmetric matrix-free standard
eigenproblem suitable for a Krylov method. Reuse the existing symmetric packed
coordinates; check their Frobenius weighting and the transformed-identity
projection against this dense control before native use.

**An iterative Ritz minimum is generally an upper bound on the true relaxed
minimum, not a certified lower bound.** A small residual locates a nearby
eigenvalue but does not prove that smaller eigenvalues were not missed. Do not
report a native impossibility theorem from one converged iterative estimate.
Record independent starts, residuals, orthogonality, stability and rounding
quality; any global certificate needs additional validated spectral bounds.

The current local run remains authoritative for its own registered comparison.
After its terminal receipt, a native spectral candidate can audit local-search
failure without fitting text or changing the original separation criteria.
Behavioral extraction, removal, OOD prediction and composition remain untested
for this proposed block representation.
