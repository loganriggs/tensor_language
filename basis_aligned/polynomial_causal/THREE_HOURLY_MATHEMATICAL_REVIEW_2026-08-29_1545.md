# Three-hour mathematical review — 2026-08-29 15:45 UTC

## Decision summary

The newest local evidence changes the mathematical priority.  The copy-specific
payload and edge are now localized, the contextual gate has a shared rank-256 HOSVD
program, and that program composes with the C512 MLP0 `Down` replacement.  On the
same changed trajectory, C512 preserves the gate's downstream state with
$R^2=0.9955$ and the two replacements have only `0.00064` nat aggregate interaction.

The bottleneck is therefore no longer “find any low-rank object.”  It is:

> select and compile MLP1/MLP2 mechanisms in a way that is invariant to their exact
> factor gauges, accounts for joint effects, and can be verified as the same causal
> mechanism under interchange.

A current primary-literature result provides a direct formulation of this problem:
**causal mechanism reduction** (CMR).  It treats internal replacement, exact folding,
compression, and causal-abstraction verification as one object rather than four
separate stages.  The highest-priority move below adapts CMR to bilin18's exact
bilinear product variables.

Strict whole-model quantities have not changed from the balance sheet merely because
the local copy composition passed: exact typed algebra covers every site, but named
causal paths cover `10.92%` of global ablation headroom and the current shipped
composite remains `0.8976` nat above its paired clean model.  The held-out residual is
dominated by the joint MLP0--2 group (`0.728` of `0.873` global Shapley nats), with
interaction fractions of 43--64% across registered cells.  This is why a diagonal
per-unit pruning score cannot be trusted without a joint diagnostic.

## Ranked move 1: gauge-invariant causal mechanism reduction

### Exact object in bilin18

For one bilinear MLP, define the physical product variables

$$
a_j(x)=(L_jx)(R_jx),
\qquad
y(x)=\sum_{j=1}^{4608}D_{:j}a_j(x)+b.
$$

Each product channel has the exact two-scalar gauge

$$
L_j\mapsto s_jL_j,
\quad
R_j\mapsto t_jR_j,
\quad
D_{:j}\mapsto (s_jt_j)^{-1}D_{:j}.
$$

This changes the coordinate value $a_j$ but not the function.  Any unit-importance or
simplicity score that changes under this transformation is measuring a parameterization,
not a mechanism.

For a replaced set $S$, CMR defines the perturbation
$\delta=\phi(a_K)-a_S$ and the local replacement risk

$$
Q_S(\phi)=\mathbb E\left[
g_S^\top\delta+\frac12\delta^\top H_S\delta
\right].
$$

With affine downstream logits and constant mean replacement, its per-channel
logit-distortion specialization is

$$
s_j=\operatorname{Var}(a_j)\lVert D_{:j}\rVert_2^2.
$$

Unlike variance alone, this score is exactly invariant under the bilinear gauge.
Constant replacement compiles by deleting $D_{:j}$ and adding
$D_{:j}\mathbb E[a_j]$ to the bias.  An affine replacement
$a_S=\beta+Ba_K$ compiles by

$$
D'_K=D_K+D_SB,
\qquad
b'=b+D_S\beta.
$$

There is no runtime mask and no surviving call to the removed mechanism.

The relevant primary source derives the general second-order replacement risk, exact
constant/affine folding, a reparameterization-invariant logit score, and a margin
certificate connecting logit distortion to interchange agreement
([Asiaee, 2026, arXiv:2602.24266v2](https://arxiv.org/html/2602.24266v2)).  It is a
recent preprint, so its general empirical claims are not treated as settled; the
algebra used here is independently checked below.

### Theorem/operational consequence beyond reconstruction

For an interchange intervention $I$, let $m_I(x)$ be the native intervened model's
top-1 margin and let $D_2$ be expected squared logit distortion between the native and
compiled intervened models.  The CMR certificate gives, for every $\epsilon>0$,

$$
\Pr(\text{top-1 disagreement})
\leq
\Pr(m_I\leq2\epsilon)+\frac{D_2}{\epsilon^2}.
$$

Thus this simplification criterion buys three things that local activation MSE does
not: an exactly smaller executable map, gauge-invariant selection, and a falsifiable
lower bound on interchange accuracy.  It also naturally exposes the problem with
independent scores: the joint distortion contains off-diagonal covariance/curvature
terms.  Those terms must be measured because the current early-stack residual is
strongly interactive.

### Assumptions that may fail

- The second-order score is local; replacing many product channels can leave its
  regime.
- Per-channel additivity requires diagonal or block-diagonal curvature.  Existing
  MLP0--2 interactions make this doubtful.
- Exact folding applies at the immediate affine `Down` interface.  Interchange
  fidelity through RMSNorm, attention, and later residual additions is a separate
  empirical obligation.
- A low logit-risk mechanism may be predictive but not semantically nameable.
- Calibration and verification distributions may disagree, especially on code/OOD.

### CPU proof executed in this review

New implementation:

- `bilinear_causal_mechanism_reduction.py`
- `test_bilinear_causal_mechanism_reduction.py`
- `bilinear_cmr_mlp0_proof.py`
- `bilinear_cmr_mlp0_proof_results.json`

Five known-answer tests pass.  The create-only proof then loaded the exact pinned
2.068-GB checkpoint on CPU and operated on actual MLP0 weights with shapes
$L,R\in\mathbb R^{4608\times1152}$ and
$D\in\mathbb R^{1152\times4608}$.  On 256 synthetic full-support states, independent
product gauges used log-scales in $[-3,3]$.

| Check | Result |
|---|---:|
| physical MLP write error after exact gauge | relative RMS `3.45e-7` |
| raw-variance top-512 Jaccard after gauge | **`0.0723`** |
| CMR-score top-512 Jaccard after gauge | **`1.0000`** |
| maximum relative CMR-score error | `2.11e-7` |
| compiled constant-replacement error | relative RMS `2.01e-7` |
| compiled affine-replacement error | relative RMS `2.43e-7` |

The result is deliberately scoped: this proves the algebra and demonstrates that raw
variance rankings are almost arbitrary under a function-preserving gauge on the real
weights.  It does **not** show that CMR selects a faithful natural-text MLP program.
The synthetic top-64 off-diagonal fraction was only `0.0347`, but that number has no
natural-text authority and is not used to claim additivity.

### Cheapest falsifying model experiment

On a frozen natural-text discovery/evaluation split, rank fixed product groups by:

1. raw activation variance;
2. invariant tensor mass;
3. individual CMR logit distortion;
4. joint/block CMR risk including off-diagonal terms;
5. matched random groups.

Install constant and affine folds at MLP1 first, on the already composed C512 plus
HOSVD background.  Score held-out CE/KL, physical MLP1 response, the layer-8 copy
state, exact copy-edge behavior, and interchange swaps.  Falsify the move if joint
CMR does not beat the invariant-mass and random controls at equal executable price,
or if the margin certificate is vacuous because almost all intervened margins are
small.  Recompute scores after every promoted replacement; the theorem is local and
the trajectory changes.

## Ranked move 2: response-conditioned multi-view tensor identification

### Exact object in bilin18

Earlier weight-action SAEs found sparse, CE-efficient reconstructions, but individual
atoms were unstable across seeds.  Raw tensor CP/Kruskal work also encountered the
generic identifiability ceiling.  The new proposal is not another decomposition of
the raw MLP tensor.  It defines an atom by three *downstream views* of the same early
state:

$$
u=\text{MLP1 Left response},\qquad
v=\text{MLP1 Right response},\qquad
w=(z_{\mathrm{copy}},\text{MLP2 response}).
$$

After centering/whitening and randomized projection, form the cross-moment

$$
T=\mathbb E[u\otimes v\otimes w].
$$

Under a multi-view latent-component model this has a CP form

$$
T=\sum_{j=1}^{r}\kappa_j a_j\otimes b_j\otimes c_j.
$$

Kruskal's condition $k_A+k_B+k_C\geq2r+2$ makes the factors unique up to permutation
and scaling; robust versions give approximate recovery under perturbation
([Bhaskara, Charikar, and Vijayaraghavan, 2014](https://proceedings.mlr.press/v35/bhaskara14a.html)).
Spectral moment methods exploit exactly this three-view structure in identifiable
latent-variable models
([Anandkumar et al., 2014](https://jmlr.csail.mit.edu/papers/v15/anandkumar14b.html)).

### Measurable consequence beyond reconstruction

If the unstable SAE atoms are rotations of genuine shared causes, downstream views
should pin a reproducible basis.  Its factors must recur across data halves and must
predict which MLP0 write edits jointly alter MLP1's two product branches and the copy
consumer.  That gives canonical nodes for the typed sparse graph and predicts
cross-consumer edits, not only reconstruction.

### Assumptions that may fail

- The three deterministic views are not conditionally independent given one atom.
- Multiple sparse atoms coactivate; a continuous distributed code need not be a
  finite mixture with nonzero third cumulants.
- Kruskal's condition is sufficient, not necessary, and may fail at useful ranks.
- Whitening and finite third moments can be unstable for rare language states.
- A stable moment component can still be causally inert.

### Cheapest falsifying experiment

Collect the three views once on 64 fit and 64 held-out documents; project each to 32
dimensions before forming the $32^3$ tensor.  Fit ranks 8, 16, and 32 on each half.
Require factor matching materially above random and above the historical SAE atom
stability, held-out third-moment prediction, and a causal swap/knockout match.  If
split-half factors are unstable or the causal response does not follow the matched
factor, prune multi-view CP rather than increasing rank.

## Ranked move 3: empirical balanced realization at the validated copy cut

### Exact object in bilin18

Treat depth as a finite, time-varying nonlinear dynamical system.  At the early-stack
cut, admissible inputs are physical MLP0/1/2 write changes; state is the residual
stream; outputs are the validated copy state, H3/H4 edge scalars, registered circuit
responses, and final logits.  Estimate:

- a controllability covariance $W_c$ from actual admissible write perturbations;
- an observability covariance $W_o$ from Fisher-weighted downstream JVP/VJP responses.

Balanced directions are ordered by the singular values associated with $W_cW_o$.
A direction is retained only when upstream mechanisms can move it *and* downstream
consumers can observe it.  This is the nonlinear empirical analogue of balanced
truncation; empirical controllability and observability Gramians for nonlinear,
time-varying systems are developed by
[Condon and Ivanov (2004)](https://arrow.tudublin.ie/scschmatart/70/).

### Measurable consequence beyond reconstruction

The balanced tail should predict error for a *family* of component edits and should
compose across the MLP0/1/2 boundary.  It can reduce the 256-dimensional HOSVD copy
input to the part both built by the early stack and read by registered consumers.  A
successful rank is a causal interface dimension, not a variance rank.

### Assumptions that may fail

- The transformer is nonlinear and trajectory-dependent; classical infinite-horizon
  LTI error bounds do not transfer automatically.
- Local tangent responses may not predict finite deletions or large interchange swaps.
- Empirical Gramians depend on the chosen edit and output batteries.
- Rare OOD directions may be absent from both empirical covariances.

### Cheapest falsifying experiment

The repository already contains a proof-checked finite-horizon cut implementation and
frozen MLP0--2 geometry, but no scored Fisher outcomes.  Reuse it on the newly
validated output battery: $z_{\mathrm{copy}}$, copy scalars, and final Fisher probes.
At ranks 16/32/64/128, compare prediction of held-out C512, MLP1, and MLP2 interventions
against activation PCA and the existing HOSVD basis.  Stop if the balanced spectrum
or response prediction is unstable under document doubling, or if it does not beat
PCA at matched rank.  This is a newly useful application of existing infrastructure,
not a claim that the old generic Hankel probe succeeded.

## Pruning across the requested mathematical families

| Family | Current ruling |
|---|---|
| Tensor/arithmetic-circuit rank | Exact multiplication count remains an upper bound and flattenings a weak lower bound. Do not optimize raw rank; use response-conditioned rank inside moves 2 or 3. |
| Simultaneous factorization/shared dictionaries | Shared HOSVD already paid off for the copy gate. Generic joint diagonalization concentrated weights without causal gain. Retain only consumer-conditioned multi-view factorization. |
| Polynomial invariants/gauge quotient | A validity condition, not a standalone search. The new CPU proof shows why: raw variance selection collapses under an exact gauge while CMR remains invariant. |
| Algebraic complexity | Useful for executable multiplication/storage bills after a mechanism passes; insufficient to decide which products matter. |
| System identification/minimal realization | Promote only at the validated copy/MLP2 output cut via empirical balancing. Generic sequence-state splicing was OOD and failed its low-rank test. |
| Hankel/automata | Defer. The existing prefix/suffix probe did not beat its additive baseline and incurred `+3.54` nat splice shift. |
| MDL/prequential coding | Final selection currency among verified programs. It cannot discover a causal basis, but should price graph structure, constants, and residual errors after moves 1--3. |
| Causal abstraction/bisimulation | Use interchange commutativity and the CMR margin certificate as acceptance tests. Do not accept high IIA from an unconstrained nonlinear alignment. |
| Information bottleneck | Prune as a discovery objective: average mutual information can discard rare intervention-critical states and does not guarantee composition. |
| Sparse program synthesis | Promote in the restricted CMR form: constant/affine folds over physical product groups with joint risk and exact price. Generic SAE atoms remain proposals only. |
| Approximation certificates | Promote the margin certificate because it predicts interchange top-1 from logit distortion and margins. Existing empirical CE remains necessary when the bound is loose. |
| Norm minimization before HOSVD | Closed locally: canonicalization added only `0.93` recovery point and missed its registered two-point bar. |

## Ranked execution plan

1. **Natural-text bilinear CMR discriminator at MLP1**, crossed with C512 plus the
   HOSVD copy gate; use group/off-diagonal scoring and sequential recomputation.
2. **If atom identity remains unstable, run the 32-cubed multi-view moment pilot**
   before investing in a larger sparse-transcoder graph.
3. **Run empirical balancing on the existing frozen tangent geometry** with the
   validated copy and final-Fisher output battery.

The first move is now implemented and algebraically verified.  Its next step is a
prospectively frozen natural-text experiment, not more synthetic proof work.
