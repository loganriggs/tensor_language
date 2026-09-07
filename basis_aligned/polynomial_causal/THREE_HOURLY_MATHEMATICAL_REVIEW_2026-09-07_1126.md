# Three-hour mathematical tensor-network review — 2026-09-07 11:26 UTC

## Decision

The corrected weight-edge atlas and the joint source factorial rule out a linear singleton ledger and
task-specific upstream writer sets for the temporal/is-was rank-four programs. The useful mathematical
object is now a **set-valued causal response function** on patched physical components, with its
higher-order interactions measured by a Möbius transform. The next experiment should not fit another
subspace or enumerate the 12-source union. It should measure all singleton and pairwise Möbius terms on
the already sufficient shared eight-MLP intersection, then use those terms only as a prospective order
for a frozen pooled greedy deletion. This separates redundant parallel writing from ordered interaction
at a cost of 38 source subsets rather than `2^12 = 4096`.

The DAS consequence is equally sharp. A complement penalty on the same task family is a training loss,
not an identification result: the optimized direction can memorize a family selector and satisfy that
loss without isolating the intended causal variable. Noise and KL are useful stability regularizers, but
the decisive constraint is held-out multi-environment invariance with the target and complement tests
both sealed out of sample. The existing comparison supports this reading: complement-constrained DAS
was worse than difference in means; KL improved its complement score from about `.4485` to `.2445`, near
the difference-in-means `.2385`, while noise alone left it at `.4486`. An aligned objective reached
`.2334` only by losing the target bar. Optimization is finding solutions; the scalar objective does not
identify the desired operator.

## Explicit causal tensor object

Let the frozen rank-46 physical source support be `V`, the two tasks be
`t in {temporal,iswas}`, and the eight response sites be

\[
R=\{\mathrm{MLP1,MLP3,MLP4,MLP6,L8H1,L9H1,L9H4,L11H3}\}.
\]

For response site `r`, its ambient width is 1152 for an MLP output and 128 for a head output. Let
`Q_r` be the frozen response basis and `M_{r,t}` its frozen task-specific rank-four mode, so
`B_{r,t}=Q_r M_{r,t}`. For a physical source subset `S subseteq V`, full head/module patching defines

\[
F_{t,r}(S)=B_{r,t}^{\mathsf T}
  \left(y_r(\operatorname{patch}_S(x,\tilde x))-y_r(x)\right)\in\mathbb R^4.
\]

The combined response is the direct sum `F_t(S)=oplus_r F_{t,r}(S)`. Its empirical inner product is the
sum over held-out rows, sites, and four coordinates. The behavior output is a scalar signed answer-margin
response; the control output is the full-vocabulary distribution summarized by KL and top-one flips.
Thus a proposed circuit must preserve a vector response, a scalar behavior, and distributional controls;
none can be silently substituted for another.

The Boolean-lattice Möbius/Harsanyi coefficient of a component set `A` is

\[
\Delta_A F_t=\sum_{B\subseteq A}(-1)^{|A|-|B|}F_t(B).
\]

For a singleton `i`, `Delta_i F=F({i})-F(emptyset)`. For a pair,

\[
\Delta_{ij}F=F(\{i,j\})-F(\{i\})-F(\{j\})+F(\varnothing).
\]

The v2 atlas's additive prediction summed singleton terms and obtained signed projections `2.4422`
(is-was) and `3.0584` (temporal), with relative squared errors `2.5044` and `4.7094`. Therefore the
degree-at-least-two remainder is not negligible. The sign and size are consistent with redundant or
saturating writes: individual interventions each receive credit for response that is not additive when
installed together.

## New causal evidence and symmetry

The joint factorial authority passed exactly and used at most 32 forward calls with no fitting or model
updates. Each task's atlas-selected top-80% set was sufficient. More importantly, their shared set

\[
I=\{\mathrm{MLP0,MLP1,MLP2,MLP3,MLP4,MLP6,MLP7,MLP8}\}
\]

was itself sufficient for both tasks: behavior projections were `.8827` temporal and `.8955` is-was,
while aggregate task-rank-four response projections were `.99998` and `.999996`. The task-specific
extras had effectively zero causal advantage: temporal own-minus-cross was `-2.8e-7` and is-was was
`+1.7e-7`. This is a permutation symmetry of the task labels at the upstream-source-set level. Task
typing seen in seven of eight exact weight-edge incidence maps is therefore downstream/contextual
typing of a shared writer bank, not evidence for distinct upstream writer circuits.

The symmetry is not a complete circuit claim. The physical complement of the temporal top-80 set still
carried `.5545` of temporal response, and the is-was complement carried `.3194`; both retained behavior
near `.96`. The 12-source union also changed controls (median KL `.0536`, maximum `.3637`, top-one flip
fraction `.125`) and its temporal control-margin RMS fraction was only `.0578`. Sufficiency and
necessity remain separated by redundancy and background-state dependence.

The tensor contraction graph is therefore:

`patched source modules -> native residual recurrence -> response-site task modes -> final margin/readout`.

Exact weights identify which downstream modes can read each write, but static incidence does not commute
with jointly patching multiple nodes through RMS normalization, attention softmaxes, and later bilinear
MLPs. The full graph has no global linear superposition symmetry. The only safe gauges are orthogonal
changes of basis inside each frozen response subspace and the induced inverse change in its coordinate
map; scalar edge scores discard information and are not gauge-complete.

## Theorem and algorithm mapping

For a scalar coordinate whose Boolean-lattice Möbius expansion is exactly `s`-sparse and bounded to
degree `d`, adaptive sparse Möbius-transform algorithms can recover it with query complexity on the
order of `s d log(n/d)` under their stated oracle and noise assumptions. Erginbas et al. give adaptive
and passive sparse Möbius algorithms and emphasize that the coherent AND basis prevents a direct appeal
to vanilla compressed sensing ([Adaptive Sparse Möbius Transforms for Learning Polynomials](https://arxiv.org/abs/2602.06246)).
Kang et al. connect interaction identification to Möbius coefficients and give sparse/bounded-degree
recovery results under explicit assumptions ([Learning to Understand: Identifying Interactions via Möbius Transform](https://arxiv.org/abs/2402.02631)).
Fast exact zeta/Möbius transforms on a finite poset can exploit its DAG or chain structure once all
required function values exist ([Fast Möbius and Zeta Transforms](https://arxiv.org/abs/2211.13706)).

Those guarantees do **not** yet apply here. We have vector-valued responses over repeated examples;
sparsity and bounded interaction degree are unmeasured; interventions occur inside a nonlinear recurrent
network; observations have deterministic numerical rather than ideal noiseless-oracle error; and a full
12-source lattice costs 4096 subsets before task/control replication. The theorems motivate a pilot for
low-degree sparsity. They do not license calling the circuit sparse.

## DAS as finite causal operator identification

Write a learned intervention as `T_theta` and environments as lexical family, construction, task,
reader, direction, and held-out row split. A same-family complement objective minimizes an empirical
risk such as

\[
L(\theta)=L_{\mathrm{target}}(\theta;E_{train})+
\lambda L_{\mathrm{complement}}(\theta;E_{train}).
\]

This admits family-selective solutions whenever activations encode environment identity. Adding
isotropic noise controls local sensitivity, and KL controls collateral output drift, but neither alone
forces the same finite operator across environments. A valid next DAS objective must impose hard
difference-in-means-level target feasibility and minimize the worst held-out complement/collateral loss:

\[
\min_\theta\max_{e\in E_{train}}L_{off}(\theta;e)
\quad\text{subject to}\quad
\min_{e\in E_{train}}R_{target}(\theta;e)\ge R_{DIM},
\]

followed by a sealed evaluation on unseen families and readers. KL and injected source noise belong in
`L_off` as regularizers, not as evidence that the intended subspace has been identified. This explains
why unconstrained optimization need not beat difference in means: DIM has the correct cross-example
inductive bias, whereas the previous loss rewarded an easier family-specific separator.

## Executable consequence, price, and stop rules

The immediate experiment uses only the shared eight-MLP set `I`. It evaluates the empty set, all eight
singletons, all 28 unordered pairs, and the full intersection: 38 source subsets. For each task it stores
the vector `F_t(S)`, behavior response, and frozen controls. It reports pair coefficients
`Delta_{ij}F_t`, their norm relative to the donor response, cosine with the residual left after singleton
addition, and cross-task similarity. No parameters are fitted.

The pilot has opposing outcomes:

1. Sparse large pair terms concentrated between ordered layers support a serial interaction quotient;
   the subsequent greedy deletion must preserve those pairs as atomic groups.
2. Broad negative pair terms with singleton overcount support redundant saturation; pooled backward
   deletion over `I`, scored jointly on both tasks and controls, is the efficient circuit finder.
3. Small pair terms despite the large global remainder falsify a degree-two approximation; stop pairwise
   expansion and test ordered layer-band conditional increments, because degree-three-or-higher or
   normalization-mediated interactions dominate.

The fixed cost ceiling is 38 patched subset evaluations plus native base/donor/control captures, with
zero gradient steps and zero model updates. The input population remains the sealed fresh temporal and
is-was rows; the outputs and norms are exactly those defined above. The experiment is not allowed to
relax thresholds after seeing results. It is a diagnostic for the deletion order, not a released circuit
by itself.

Next mathematical review due around **2026-09-07 14:26 UTC**.
