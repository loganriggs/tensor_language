# Three-hour mathematical review — 2026-09-04 00:30 UTC

## Goal and exact object

The project goal is a smaller transparent tensor program that predicts fresh and out-of-distribution text, composes
when several recovered computations are installed together, supports selective removal/swap/edit interventions, and
is simpler under literal storage, compute, edges, states, and executable-program price. Lower rank or reconstruction
error alone does not identify a circuit.

bilin18 has residual width $D=1152$, 18 blocks, 9 attention heads per block, head width $d_h=128$, and bilinear-MLP
product width $H=4608$. With the normalized residual state denoted $z\in\mathbb R^{1152}$, a bilinear MLP is

$$
M(z)=W_D\big[(W_Lz)\odot(W_Rz)\big]+b.
$$

Holding normalization fixed, this is degree two in $z$. One attention head forms two query/key score branches and
multiplies them. Ignoring rotary position notation and fixed scaling,

$$
s_{qk}=\langle W_Qz_q,W_Kz_k\rangle\,
       \langle W_{Q'}z_q,W_{K'}z_k\rangle,
\qquad
o_q=W_O\sum_{k\le q}s_{qk}W_Vz_k.
$$

With normalization fixed this write is degree five across query/key/value states. RMS normalization makes the full
network rational rather than a global polynomial. Parameters are tied across token positions within a layer and not
across layers.

R592 studies a much smaller mediator. With four sites $h$, two semantic source roles $r$, scalar attention
coefficients $e_{h,r}$, and projected content vectors $u_{h,r}\in\mathbb R^{1152}$,

$$
B_h(E,U)=\sum_{r=1}^{2}e_{h,r}u_{h,r}\in\mathbb R^{1152}.
$$

The contraction graph has one scalar-vector multiplication per role, a role sum within a site, then sequential
addition at the four native head-write locations. It is bilinear in $(E,U)$. Allowed inputs are the frozen R578
recipient/donor pairs on FIT and conditionally SELECT. Outputs to preserve and measure are full-vocabulary logits,
registered target margins, and cross-entropy. Exactness uses maximum absolute logit/vector error $10^{-5}$; scientific
thresholds and grouped bootstraps are inherited from R585. The new experiment costs 639 FIT and 322 SELECT forwards,
961 maximum, with no backward pass or update.

## Symmetry and gauge

Inside a value/output channel, $v\mapsto Gv$ and $W_O\mapsto W_OG^{-1}$ preserve the projected vector $u=W_Ov$.
The operational factor $B(E,U)$ is therefore invariant to that internal coordinate change. Query/key coordinates have
paired inverse symmetries only within transformations compatible with the model's rotary position operation; arbitrary
cross-head mixing is not an architectural gauge because different heads use different nonlinear attention patterns.
The two score branches may be exchanged because their scalar outputs are multiplied.

There is also a semantic non-identifiability: different sites or factors may be equivalent if every allowed downstream
reader and intervention treats them identically. R592 does not resolve that quotient; it tests one fixed factorization.
The later decomposition should group factors by held-out downstream behavior rather than by head label or variance.

## Exact finite-difference decomposition

For recipient $x$ and donor $y$, define the two-variable finite differences

$$
\begin{aligned}
\Delta_E&=B(E_y,U_x)-B(E_x,U_x),\\
\Delta_U&=B(E_x,U_y)-B(E_x,U_x),\\
\Delta_{EU}&=B(E_y,U_y)-B(E_y,U_x)-B(E_x,U_y)+B(E_x,U_x).
\end{aligned}
$$

Then exactly

$$
B(E_y,U_y)-B(E_x,U_x)=\Delta_E+\Delta_U+\Delta_{EU},
$$

with

$$
\Delta_{EU}=\sum_r(e^y_r-e^x_r)(u^y_r-u^x_r).
$$

This is Möbius inversion on the two-mediator Boolean lattice. It is an exact decomposition of the chosen intervention,
not a regression and not a rank approximation. It directly addresses the multiple-mediator problem: the joint term is
not silently credited to either single factor.

## Relation to existing exact methods

Distributed alignment search learns a rotated activation subspace whose interchange reproduces a specified high-level
counterfactual. Its formal target is causal abstraction under interchange, which matches our eventual semantic goal,
but it assumes that we have already supplied meaningful counterfactual labels and it does not by itself map a found
activation subspace to the model's bilinear weights. See Geiger et al.,
<https://arxiv.org/abs/2303.02536>, and the broader causal-abstraction formalization,
<https://arxiv.org/abs/2301.04709>.

Block-term tensor decomposition has a closer exact algebraic match to the crossed immediate-output tensor. For donor
index $i$, recipient index $j$, output functional $w_k$, and site/role $a$,

$$
\mathcal T_{ijk}=\sum_a e_{i,a}\langle w_k,u_{j,a}\rangle.
$$

Each $a$ contributes a multilinear-rank $(1,L_a,L_a)$ block. Domanov and De Lathauwer give checkable generic and
deterministic uniqueness conditions and an eigenvalue-based recovery algorithm for this decomposition
(<https://doi.org/10.1137/18M1206849>). The mapping is exact for a complete donor $\times$ recipient $\times$ output
tensor. Our current data violate the theorem's clean setting because semantic pairs are sparse rather than a complete
Cartesian grid, factors are highly structured rather than generic, and downstream layers act nonlinearly on the
inserted write. Thus block-term decomposition can propose cross-head groups, but causal interchange and held-out
prediction must still decide whether they are circuit variables.

The newer general block-term work extends recoverability beyond $(1,L,L)$ terms and can recover block sizes
algebraically (<https://doi.org/10.1137/23M1557246>). It does not remove the same sparse-observation and downstream-
semantics gaps.

## A new executable consequence: test whether a partial coefficient swap is locally realizable

R592 deliberately swaps two output-factor coefficients while leaving the rest of the attention pattern untouched.
That is not generally the output of any query/key state. The distinction can be tested, rather than left verbal.

For one endpoint and site, let $F(z)$ be the vector of all causally allowed attention coefficients at the query, as a
function of a chosen local state $z$. Let $d$ be the desired coefficient change: the registered equality coordinates
change to donor values and every unregistered coordinate remains zero. Compute the Jacobian

$$
J=\frac{\partial F}{\partial z}\bigg|_{z_x}.
$$

The minimum-norm first-order state change is $\delta z=J^+d$, and the unavoidable tangent residual is

$$
\rho_{\mathrm{tan}}=\frac{\|J\delta z-d\|_2}{\|d\|_2}.
$$

- If $\rho_{\mathrm{tan}}$ is large across held-out endpoints, the partial coefficient intervention is not even locally
  realizable by the chosen query/key state. It remains useful as a controlled output-factor intervention, but cannot be
  described as a QK swap.
- If $\rho_{\mathrm{tan}}$ is small, perform a nonlinear constrained solve for $F(z_x+\delta z)=F(z_x)+d$, validate on
  untouched coefficients and downstream logits, and test the resulting state interchange. Passing only the tangent
  test is a screen; passing the nonlinear held-out interchange is identification.

The same computation can compare candidate state spaces: query only, each key only, both QK branches, or an upstream
residual subspace. Its literal price is one Jacobian-vector interface plus a least-squares solve per endpoint; it does
not change model storage. The route is killed if the residual is large, the nonlinear solve cannot preserve
unregistered coefficients, or a realizable state swap loses the selective behavioral effect.

## Decision

The fixed-geometry centered R592 experiment remains higher information than immediately fitting a new decomposition:
it first establishes whether the proposed selector/content variables exert the predicted causal effects without the
known numerical confounds. The exact finite-difference identity justifies its interaction accounting. The tangent
realizability test is the next mathematical discriminator if and only if R592 identifies the output factor; it will
separate a useful surgical intervention from a native QK-computable variable.

The block-term route remains the best candidate for grouping pieces across heads after valid factor capture. It is not
licensed as a compression result: recovered blocks must predict held-out tensors, survive gauge-aligned resampling,
and then pass selective interchange/removal tests. The immediate concrete step is therefore to freeze and independently
review R592's exact preregistration before implementing it.
