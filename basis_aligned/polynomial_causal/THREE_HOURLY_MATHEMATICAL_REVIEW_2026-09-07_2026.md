# Three-hour mathematical tensor-network review — 2026-09-07 20:26 UTC

## Decision

The exact attention experiment exposes a mathematical distinction that the earlier DAS formulation
blurred: an absolute causal clamp and an additive response perturbation are different maps once an
upstream intervention changes the input to a downstream module.  The correct finite observation
operator must therefore index not only a subspace or factor, but also its installation semantics and
causal order.  The current retry tests an exact absolute factor clamp.  If it fails selectivity, a
regularized nonlinear DAS is justified only as a constrained search over this correctly defined
operator—not as another target-minus-complement scalar objective.

The user's noise/KL proposal is mathematically useful but insufficient by itself.  Small Gaussian
training noise produces a local Jacobian/Tikhonov penalty; it discourages sample memorization but does
not guarantee target preservation or OOD invariance.  The executable objective should instead make
A1 causal transfer a hard cross-fitted feasibility condition, minimize worst-group P no-effect plus
the noise-induced sensitivity penalty among feasible projectors, choose every hyperparameter on
opposite A1/P parity, and open A2/C exactly once.  This prevents the trivial zero-effect complement
solution and makes failure diagnose the objective rather than optimizer capacity.

## Explicit model, tensors, contraction graph, and interventions

Bilin18 is an 18-block decoder with vocabulary `V=50,304`, residual width `d=1,152`, nine heads,
head width `p=128`, and bilinear-MLP width `m=4,608`.  For a batch row `i`, token positions
`s,q <= T_i`, layer `l`, and head `h`, normalized residuals produce

\[
 Q^{(1)}_{ilqha}=X_{ilqd}W^{Q1}_{lhda},\quad
 K^{(1)}_{ilsha}=X_{ilsd}W^{K1}_{lhda},
\]

with analogous `Q2,K2`, and effective values

\[
 V_{ilsha}=(1-\lambda_l)X_{ilsd}W^V_{lhda}+\lambda_l V^{(1)}_{isha}.
\]

For ordinary bilinear attention the pattern is a difference of two causal softmax contractions,

\[
 P_{ilhqs}=\operatorname{softmax}_s(Q^{(1)}K^{(1)\top}/p)
 -\beta_l\operatorname{softmax}_s(Q^{(2)}K^{(2)\top}/p),
\]

while squared-attention blocks replace this with the masked product of the two score tensors.  The
head response is `H_{ilqha}=sum_s P_{ilhqs}V_{ilsha}` and the nine heads contract through
`W^O_l in R^{1152 x 1152}`.  The MLP remains

\[
 M_l(x)=D_l[(L_lx)\odot(R_lx)],\quad
 L_l,R_l\in\mathbb R^{4608\times1152},\ D_l\in\mathbb R^{1152\times4608}.
\]

Thus an MLP is degree two in its normalized local input, but the full model is not polynomial because
RMS normalization and softmax remain data dependent.  Embedding/output weights are tied.  Residual
basis changes induce inverse gauges on adjacent weights; a rank-`k` projector `UU^T` is invariant to
`U -> UG` for `G in O(k)`, so its literal degrees of freedom are `pk-k(k+1)/2`, not `pk`.

For native base/donor patterns and values, each selected head has the exact three-vector factor tensor

\[
 \Delta H = (P'-P)V + P(V'-V) + (P'-P)(V'-V).
\]

Across the four heads this is a response tensor
`F[i,l,h,q,a,f]` with `f in {pattern,value,interaction}` and `a=1..128`.  The current action for a
factor subset `S` is an **absolute clamp**

\[
 H^{\mathrm{live}}_{ilhq:}\leftarrow H^{\mathrm{base}}_{ilhq:}
       +\sum_{f\in S}F_{ilhq:f},
\]

executed in increasing layer order, plus an absolute complete attention-15 clamp.  An additive action
would instead set `H_live <- H_live + sum_f F_f`; after an earlier clamp these are unequal.  The
invalid run measured that difference directly: the additive full arm overshot parent A1/A2 by about
`.25/.30` despite raw factor closure `1.14e-5`.

For environment/panel `e`, row `i`, test block `o`, and ordered action `a`, define the finite causal
response operator

\[
 \mathcal R_{(e,i,o),a}=o(F(x^{base}_{ei};a))-o(F(x^{base}_{ei})).
\]

The tests include answer-minus-foil margin, final residual in `R^1152`, vocabulary KL/top-one, and
registered downstream reader blocks.  Allowed inputs are capability-qualified equal-token-length
paired texts.  Approximation is measured blockwise by signed projection, relative squared error,
direction fraction, KL, and discrete flips; scalar averaging cannot certify identification.

Literal price for the exact factor screen is 17 forwards over 64 rows and zero fitting.  A rank-`k`
head-local DAS projector stores `128k-k(k+1)/2` real degrees of freedom per independently fitted head,
plus the declared intervention edge and dose; four separate rank-one heads therefore store 508
Grassmann degrees of freedom.  Noise-regularized fitting multiplies intervention forwards by noise
replicates and folds and is justified only after exact factors fail.

## Neighboring results and exact assumption audit

### Noise as Tikhonov/Jacobian regularization

Bishop shows that, under small additive input noise and a sum-of-squares loss, expected noisy training
is equivalent to a positive-semidefinite generalized Tikhonov regularizer involving first derivatives
of the learned mapping ([Neural Computation 1995](https://doi.org/10.1162/neco.1995.7.1.108)).  Map
the learned mapping to the intervention response `g_U(x)=R(x;UU^T)` and perturb only training
activation rows: for `xi ~ N(0,sigma^2 I)`,

\[
 E_\xi\|g_U(x+\xi)-y\|^2
 \approx \|g_U(x)-y\|^2+\sigma^2\|J_xg_U(x)\|_F^2.
\]

This gives an executable sensitivity penalty and directly addresses memorization.  Its assumptions do
not hold exactly for our KL loss, finite interventions, RMSNorm/softmax nonsmooth regimes, or top-one
flips.  Noise is therefore a local regularizer, not a proof that a subspace is task invariant.

### Worst-group objectives and invariance

Sagawa et al. show that naive group DRO can still fail in overparameterized networks and that stronger
regularization or early stopping is essential for worst-group generalization
([arXiv 1911.08731](https://arxiv.org/abs/1911.08731)).  The exact mapping is to treat construction,
direction, and parity as groups and minimize the worst control/target constraint violation.  Their
guarantees and experiments concern supervised prediction under predefined group shifts, not causal
subspace interventions, so they motivate the max-group form but do not identify our circuit.

IRM asks for a representation admitting the same optimal classifier across environments
([Arjovsky et al. 2019](https://arxiv.org/abs/1907.02893)); later analysis constructs simple cases where
the practical linear form fails even at population level
([Kamath et al. 2021](https://proceedings.mlr.press/v130/kamath21a.html)).  This maps to demanding a
stable intervention rule across A1 groups, but warns against treating one scalar complement loss as
the invariant mechanism.  Our full nonlinear suffix, finite rows, and task-defined causal actions
violate the linear-predictor assumptions.

### Singular-subspace stability and minimal realization

Wedin bounds singular-subspace rotation by perturbation size divided by a retained/discarded singular
gap ([BIT 1972](https://doi.org/10.1007/BF01932678)).  Applied to fold-specific causal response
matrices, it provides a valid stability test for a *block* of response directions; it does not make an
unstable or gapless individual DAS axis identifiable.

Ho and Kalman's construction recovers a minimal LTI realization from a block Hankel matrix of Markov
parameters under linearity, time invariance, controllability, and observability
([1966 record](https://ntrs.nasa.gov/citations/19670049337)).  Interventions can be viewed as inputs and
downstream tests as outputs, but Theseus is layer-varying, input-dependent, nonlinear, and uses finite
clamps rather than LTI impulses.  The theorem is exact only for a depth-tied local-linear restriction.

Oseledets' tensor-train decomposition constructs stable low-rank approximations from unfolding SVDs
and prices their storage and contractions efficiently
([SIAM J. Sci. Comput. 2011](https://doi.org/10.1137/090752286)).  It can compress the already measured
`F[i,l,h,q,a,f]` tensor after circuit identification, but TT rank neither chooses causal factors nor
resolves the absolute/additive action semantics.  It is demoted under the anti-rank-drift gate.

## Executable constrained-DAS consequence

If the exact absolute factor retry has no selective subset, fit `U in St(128,k)` separately or shared
only where the per-head factor atlas supports it.  Split A1 and P by frozen group parity.  On each
outer fold, select `k,sigma,lambda_J` using only the opposite inner A1/P fold and solve lexicographically:

\[
 \text{find }U\text{ such that }\min_{g\in A1_{train}}
   \operatorname{proj}(R_g(U),R_g^{target})\ge .75,
 \quad \min_g\operatorname{dirfrac}_g\ge .875;
\]

among feasible projectors minimize

\[
 \max_{g\in P_{train}} D_{KL}(p^{base}_g\|p^U_g)
 +\lambda_J E_\xi\frac{\|R_g(U;x+\xi)-R_g(U;x)\|^2}{\sigma^2}
 +\lambda_S\|U_1U_1^T-U_2U_2^T\|_F^2.
\]

Use a fixed unit intervention dose, orthonormal columns, no learned classifier that can answer “is
this the requested subspace?”, and no outcome-conditioned row dropping.  Hyperparameters are chosen
by inner worst-group target/control feasibility, not average loss.  A2 and C remain sealed until one
choice is frozen.  Final acceptance requires A2 target bars, zero P/C top-one flips, median KL
`<=.02` separately, stable fold projectors under a Wedin gap report, and preservation of every
registered reader block.  The zero projector is infeasible by construction.

Opposing predictions are explicit.  If noise/KL regularization reduces fold sensitivity and P loss
while A2 remains high, the earlier constrained DAS overfit its finite task rows.  If inner folds pass
but A2 fails, the task objective still identifies construction-specific geometry.  If target
feasibility itself is empty, the chosen head/subspace family is wrong.  If exact factors outperform
all learned projectors, optimize no further and translate those factors through QK/OV weights.

The empirical route still dominates an exact tensor-train or Ho–Kalman construction because the
current uncertainty is causal operation semantics and selective transfer, not compression.  The
immediate safe action remains the already queued absolute-clamp factorial; its result determines
whether the next executable object is weight translation, a 12-component factor atlas, or this
cross-fitted constrained-DAS protocol.
