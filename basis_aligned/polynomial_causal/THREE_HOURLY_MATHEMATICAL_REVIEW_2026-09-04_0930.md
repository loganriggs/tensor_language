# Three-hour mathematical review — 2026-09-04 09:30 UTC

## Circuit target and current boundary

The target is a smaller executable tensor program that specifies what each circuit reads, computes, writes, and feeds
downstream. Its units must be allowed to cross native attention-head/MLP boundaries or split one native module. The
program must predict held-out and out-of-distribution behavior, be sufficient when extracted, support selective swaps
and removals despite redundancy, compose across circuits, and remain stable across data splits, fitting runs, and
internal gauges. Literal storage and compute are eventual prices, not substitutes for these circuit properties.

The repaired task-14 subject–verb-agreement authority passed native FIT capability. The first localization
preregistration at commit `7986557` is nevertheless blocked by independent review. In particular, its ordinary
A1/A2/P rows perfectly confound complete-subject number with the morphology of one head token, while its coordinated
C rows test only invariance to an attractor edit. A one-dimensional morphology coordinate could therefore pass every
gate while assigning a subject such as “the key and the dog” the singular state. No localization implementation or
model run is authorized. A prospective successor is now materializing exact manifests and adding affirmative
coordinated-subject transfer.

## 1. The actual Theseus contraction

Let the prompt length be $T$, the residual width be $D=1152$, the number of blocks be $L=18$, the number of attention
heads be $H=9$, the head width be $d_h=128$, and the bilinear-MLP product width be $M=4D=4608$. At a block boundary,
the residual is

$$
R_l\in\mathbb R^{T\times D}.
$$

Each block first mixes the previous residual with the normalized token-embedding residual $R_0$ using two learned
scalars, then applies RMS normalization. In the configured double-bilinear attention, for head $h$ and positions
$i,j$,

$$
a^h_{ij}
=\frac{\langle q^h_i,k^h_j\rangle}{d_h}
 \frac{\langle \tilde q^h_i,\tilde k^h_j\rangle}{d_h},
$$

with a causal mask. The value is a learned mixture of the current block's value projection and the block-0 value
projection, and the head output is

$$
o^h_i=\sum_{j\le i}a^h_{ij}v^h_j.
$$

Concatenated head outputs are mapped back to $D$ dimensions and added to the residual. The normalized MLP input
$z_{l,i}\in\mathbb R^D$ is then processed as

$$
F_l(z)=W_{D,l}\left[(W_{L,l}z)\odot(W_{R,l}z)\right]+b_l,
$$

where

$$
W_{L,l},W_{R,l}\in\mathbb R^{M\times D},
\qquad
W_{D,l}\in\mathbb R^{D\times M}.
$$

Conditional on already normalized inputs, one MLP is degree two and one double-QK attention output is degree five:
two degree-two QK contractions are multiplied and then multiplied by a linear value. The complete model is not a
polynomial in the raw residual because every block contains RMS normalization and the output applies a bounded `tanh`
to logits. Any theorem that assumes a globally polynomial tensor network therefore applies only to a declared
normalized local contraction, not silently to the whole transformer.

The currently allowed inputs are only the frozen FIT prompts. The target observable is

$$
m(x)=\ell_{\texttt{ are}}(x)-\ell_{\texttt{ is}}(x),
$$

plus finite changes in $m$ under registered natural-donor interventions. SELECT, TEST, and OOD are closed. There is no
current approximation norm for a replacement model: the live question is causal identification, scored by signed
interchange, necessity, invariance, and reader-handoff effects. The successor compiler must state physical calls,
backward calls, updates, retained arrays, bytes, and GPU time before any execution; those prices are not yet frozen.

## 2. Symmetries and what can actually be identified

For a candidate subspace $U\in\mathbb R^{D\times k}$, the intervention depends only on

$$
P=UU^{\mathsf T}.
$$

Thus $U$ and $UO$ are identical for any orthogonal $O\in O(k)$; for rank one, $u$ and $-u$ are the same projector.
The sign may be oriented for exposition using plural-minus-singular discovery means, but the physical object is the
projector.

The bilinear hidden factors have further gauges. Product units may be permuted; a row of $W_L$ and a row of $W_R$ may
be reciprocally rescaled with the corresponding $W_D$ column adjusted; and the two branches may be exchanged. Within
attention, reciprocal query/key basis changes preserve a QK form, value/output bases have a corresponding reciprocal
gauge, and heads may be permuted. Therefore a raw product index, head coordinate, or factor vector is not a semantic
unit. The stable object must be an intervention projector or an equivalence class of weight terms with the same
registered downstream response.

## 3. Exact match to causal-abstraction and DAS mathematics

[Distributed Alignment Search](https://proceedings.mlr.press/v236/geiger24a.html) defines an interchange intervention
by rotating a neural representation, replacing selected orthogonal coordinates by their values under natural source
inputs, rotating back, and comparing the result with a high-level causal model. It optimizes the rotation while the
neural and high-level models remain frozen. This maps directly to task 14:

| causal-abstraction object | task-14 object |
|---|---|
| high-level variable | complete grammatical-subject number $s\in\{-1,+1\}$ |
| low-level state | one residual vector $r_{b,p}\in\mathbb R^{1152}$ |
| alignment | projector $P=UU^{\mathsf T}$ |
| base/source settings | target prompt and natural donor prompt |
| high-level intervention | replace subject number while preserving the registered nuisance variables |
| output comparison | signed change in the are-minus-is logit contrast |

The theorem-level promise is conditional: exact equality under all aligned interventions gives a causal abstraction;
[approximate causal abstraction](https://proceedings.mlr.press/v115/beckers20a.html) provides a graded relation when
effects differ. DAS itself does not guarantee that gradient descent finds the global optimum or that the optimum is
unique. Its own paper assumes that valid high-level counterfactuals can be sampled. The blocked v1 task-14 design shows
why that assumption is substantive: cross-noun and cross-syntax swaps did not distinguish complete-subject number from
single-token morphology. Adding C only as a no-change control did not complete the high-level causal model.

The v2 successor therefore needs bidirectional coordinated-plural/ordinary-singular swaps at the prediction position,
an affirmative plural-state test for every C endpoint, and same-state coordinated/ordinary-plural controls. These
counterfactuals are not extra data for a rank fit; they are what makes the proposed high-level variable identifiable.

## 4. A closed-form local solution: the causal-response matrix

There is an exact solution to a useful restriction of the DAS objective. For registered target/donor pair $n$, define

$$
\delta_n=r(d_n)-r(x_n),
\qquad
g_n=\nabla_r m(x_n),
\qquad
\sigma_n=\frac{s(d_n)-s(x_n)}{2}.
$$

For a rank-one intervention $P=uu^{\mathsf T}$ and a locally affine downstream model,

$$
\begin{aligned}
E_n(u)
&\approx \sigma_n g_n^{\mathsf T}uu^{\mathsf T}\delta_n\\
&=(u^{\mathsf T}\sigma_ng_n)(u^{\mathsf T}\delta_n)\\
&=u^{\mathsf T}A_nu,
\end{aligned}
$$

where

$$
A_n=\frac{\sigma_n}{2}
\left(g_n\delta_n^{\mathsf T}+\delta_ng_n^{\mathsf T}\right).
$$

For nonnegative preregistered pair weights $w_n$, the mean local effect is the Rayleigh quotient

$$
J_{\mathrm{local}}(u)=u^{\mathsf T}Au,
\qquad
A=\frac{\sum_nw_nA_n}{\sum_nw_n}.
$$

Subject to $\lVert u\rVert_2=1$, its global maximizer is a top eigenvector of $A$. If the top eigenvalue is simple, the
local projector is unique up to sign. This is **not** PCA: $A$ combines actual donor changes with the signed gradient
of the declared behavior; activation variance never appears.

The matrix need not be stored. A Lanczos iteration can apply

$$
Av=\frac{1}{2\sum_nw_n}\sum_nw_n\sigma_n
\left[g_n(\delta_n^{\mathsf T}v)+\delta_n(g_n^{\mathsf T}v)\right]
$$

in $O(ND)$ time per iteration and $O(ND)$ stored discovery values. A randomized CPU calculation verified the quadratic
identity to $5.6\times10^{-17}$ and the eigenvector optimum against 100,000 random unit directions.

This gives the future instrument a deterministic local null model:

- if finite DAS and the spectral projector agree and finite effects match the Taylor prediction, the state is locally
  affine and a nonlinear optimizer was unnecessary;
- if finite DAS transfers but the spectral projector fails, curvature or a genuinely finite interchange is essential;
- if both succeed on A1/A2/P but fail coordinated-subject transfer, they found head morphology rather than the declared
  grammatical state; and
- if both fail despite a valid full-state ceiling, the linear rank-one causal abstraction is unsupported at that site.

Absolute leakage penalties are not quadratic, so this spectral object is an initializer/diagnostic for the signed
answer-changing part, not a replacement for the finite preregistered controls.

## 5. Multiple mediators become a finite Hessian test

[Vaidyanathan et al.](https://arxiv.org/abs/2606.27510) show that ordinary activation-patching effects mix a mediator's
effect with interactions, that the interaction is negligible for a locally affine model, and that multi-component
effects decompose into pairwise and higher-order finite differences. For two residual interventions $\Delta_i$ and
$\Delta_j$, the exact finite interaction is

$$
I_{ij}
=m(r+\Delta_i+\Delta_j)-m(r+\Delta_i)-m(r+\Delta_j)+m(r).
$$

At second order,

$$
I_{ij}\approx \Delta_i^{\mathsf T}H_m(r)\Delta_j.
$$

Thus the spectral local model and the two-site experiment test one another. Near-zero finite interaction supports the
locally affine restriction. Strong interaction says that one-at-a-time effects cannot be added; weak singleton
necessity plus strong joint necessity may reflect mutually covering routes. It does not by itself prove that the two
sites represent the same variable. The blocked v1 review correctly requires explicit baselines, positive finite
denominators, fixed aggregation, and distinct terminal labels before reader or redundancy claims can be emitted.

## 6. Exact translation from an identified state to bilinear weight terms

Once a normalized MLP-input direction $q$ and a downstream output/read direction $v$ have passed finite causal tests,
define

$$
c=W_D^{\mathsf T}v\in\mathbb R^{4608}.
$$

Writing $l_j^{\mathsf T}$ and $r_j^{\mathsf T}$ for row $j$ of the two branches, the scalar computation is

$$
v^{\mathsf T}F(z)=\sum_{j=1}^{4608}c_j(l_j^{\mathsf T}z)(r_j^{\mathsf T}z)
=z^{\mathsf T}Q_vz,
$$

with

$$
Q_v=\frac12\left[
W_L^{\mathsf T}\operatorname{diag}(c)W_R+
W_R^{\mathsf T}\operatorname{diag}(c)W_L
\right].
$$

For a finite state change $z\mapsto z+\beta q$, product term $j$ contributes exactly

$$
\Delta_j(z,\beta)=c_j\left[
\beta(l_j^{\mathsf T}q)(r_j^{\mathsf T}z)
+\beta(l_j^{\mathsf T}z)(r_j^{\mathsf T}q)
+\beta^2(l_j^{\mathsf T}q)(r_j^{\mathsf T}q)
\right].
$$

The 4608 contributions sum exactly to the MLP's change along $v$. A randomized CPU check reproduced both this hidden-
term sum and the equivalent self/mixed/context quadratic decomposition to machine zero.

For each term, form its **causal response signature** across all registered examples,

$$
\rho_j=(\Delta_j(z_1,\beta_1),\ldots,\Delta_j(z_N,\beta_N)).
$$

Terms from different physical products or modules can be grouped when downstream interventions cannot distinguish
their summed signatures; one native MLP is split when its signatures serve different counterfactual families. This is
the user's requested interaction-determined basis in an exact weight representation. Grouping raw $j$ indices by
weight cosine would not be gauge-stable; grouping their registered response functions is operationally meaningful.

The bilinear inverse-problem literature formalizes identifiability only up to transformation groups under subspace or
sparsity assumptions ([Li, Lee, and Bresler](https://arxiv.org/abs/1501.06120)). It does not directly solve this task:
our weights are already known, while the unknown object is the semantically correct $q,v$ pair; our measurements are
structured prompt interventions rather than generic bilinear measurements; and RMSNorm plus downstream attention
violates a single bilinear observation model. Its useful lesson is to state the gauge and recover an equivalence class,
not to expect raw factor uniqueness.

## 7. Executable consequence and decision

The next highest-information step remains the repaired task-14 causal experiment, but the mathematics changes its
implementation requirements:

1. Freeze materialized v2 split/donor manifests and the affirmative coordinated-subject semantics before any compiler.
2. In the later compiler, use the already required discovery residual differences and gradients to compute the local
   causal-response operator $A$ without a dense $1152\times1152$ allocation. Freeze this as a diagnostic; it must not
   select on validation or replace finite controls.
3. Report projector distance, signed-effect correlation, and finite-effect residual between the spectral local solution
   and each finite-DAS seed. These values decide whether curvature/interaction, rather than initialization luck, is
   doing work.
4. Only after the Q complete-subject state and an ordered reset/rescue handoff pass should the reader's exact MLP or QK
   contraction be opened and per-term causal response signatures computed.
5. Group or split physical terms by held-out downstream response equivalence, then test necessity, sufficiency,
   selectivity, and composition. Do not award circuit credit for a small rank or short parameter list alone.

The exact DAS/causal-abstraction literature solves the search formulation, not our counterfactual validity or
uniqueness. The local Rayleigh construction exactly solves the affine rank-one restriction and provides a cheap,
falsifiable baseline. The bilinear contraction exactly solves weight translation after causal axes are known. Neither
mathematical route licenses skipping the repaired coordinated-subject interventions.

Concrete continuation is already active: the blocked v1 is preserved with its independent review, and a CPU-only v2
successor is building exact manifests and complete-subject counterfactuals. No GPU localization job is authorized or
queued from this checkpoint.
