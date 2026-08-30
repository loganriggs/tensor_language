# MLP0 token/context tensor-factorial discovery

**Frozen:** 2026-08-30 03:52 UTC, before the real-model branch CE or branch-energy
outcome was computed.

## Question

Can MLP0's exact bilinear computation be separated into structurally different fixed
tensor branches—token-derived, token-by-context interaction, and context-derived—so
that later compression can give each branch its own grammar without a TopK router?

This is a decomposition/census, not yet a compressed replacement. It earns no storage,
causal-ledger, terminal, semantic, OOD, extraction, or removal credit.

## Exact object

At layer 0, let $e_t$ be the residual state before attention 0 and let $a(t,c)$ be
attention 0's write. RMSNorm supplies one scalar $\rho(t,c)$ shared by all coordinates:

$$
z=\operatorname{RMSNorm}(e_t+a)=\rho(t,c)(e_t+a).
$$

For the bias-separated bilinear MLP

$$
F(z)-b=D((Lz)\odot(Rz)),
$$

define scaled token and context inputs $p=\rho e_t$ and $q=\rho a$, so $z=p+q$.
Then

$$
F(z)=b+T(p,p)+T(p,q)+T(q,p)+T(q,q),
$$

where $T(u,v)=D((Lu)\odot(Rv))$. The frozen three branch groups are

$$
\mathrm{TT}=T(p,p),\qquad
\mathrm{X}=T(p,q)+T(q,p),\qquad
\mathrm{CC}=T(q,q).
$$

`TT` has token-specific tensor direction with a shared context-dependent RMS scale;
`X` is bilinear token/context interaction; `CC` is a continuous context quadratic.
All are fixed contractions. There is no input-dependent component selection.

## Numerical implementation and toy gate

The observed bf16 normalized state is authoritative. The scalar multiplying $e_t$ is
the least-squares collinearity scalar between the reconstructed pre-normalization sum
and the observed normalized state; $q$ is then defined as $z-p$, making $p+q=z$
in real arithmetic and to measured float32 roundoff in the implementation. The runner
reports both this numerical reconstruction error and the residual non-collinearity.

Branch contractions use float32 copies of the pinned $L,R,D$ weights. The analytical
sum must match the corresponding float32 full quadratic to relative MSE at most
$10^{-10}$ in the planted toy and at most $10^{-8}$ on real states. The difference
between that analytical sum and the deployed bf16 native MLP write is reported as a
finite-precision residual, not assigned to a semantic branch.

Before model execution, tests must verify exact branch reconstruction for random
indefinite bilinear weights, invariance under token/context rescaling that preserves
their sum, exact three-variable Möbius/Shapley accounting, and exact reconstruction of
a token table as mean plus an overlapping lexical-DAG term plus token-private residual.

## Frozen rows and arms

Reuse the already-opened P512 FIT and SELECT row roles only because this is a
no-fit descriptive diagnostic. Each has 96 source documents; score positions 64--255.
FINAL is not requested or opened. No result chooses a rank, class assignment, branch,
threshold, row, or hyperparameter.

For each role, run all eight subsets of `{TT, X, CC}`. Native bias and the measured
bf16 finite-precision residual remain in every arm by returning the native MLP0 write
minus the omitted analytical branches. Thus the full subset replays native exactly;
the empty subset is native bias plus numerical residual, not an empirical mean.

Report:

1. per-arm pooled and per-document CE;
2. the three branch Shapley contributions to full-minus-empty CE benefit;
3. all Möbius interaction dividends of the performance function $-CE$;
4. the $3\times3$ empirical branch-write Gram matrix and normalized correlations;
5. FIT/SELECT sign and rank-order transport;
6. observed-state collinearity and analytical reconstruction errors;
7. native MLP0 Left/Right/Down and all other component call censuses; and
8. exact checkpoint, row, runner, and preregistration hashes.

## Frozen interpretation

- If `TT`, `X`, and `CC` effects are mostly additive and one branch dominates, factor
  branches independently, using a lexical/DAG table for `TT`, block terms for `X`, and
  data-weighted quadratic rank for `CC`.
- If interaction dividends are large, the branch split is algebraically exact but
  independent simplification is not causally composable. Fit a joint block-term model
  with shared downstream constraints.
- If a low-energy branch has a large Shapley/ablation effect, Frobenius/activation
  energy is again the wrong simplicity metric for that branch.
- This experiment cannot establish lexical classes. The next `TT` experiment must
  compare overlapping fixed lexical/DAG incidence, tree incidence, token-private
  residual, and matched continuous controls on fresh roles.
