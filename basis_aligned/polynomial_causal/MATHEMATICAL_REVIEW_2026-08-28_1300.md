# Mathematical review: physical actions after the local-frame failure

Date: 2026-08-28 13:00 UTC

Status: literature-backed design, CPU implementation, proof tests, and prospective
preregistration. This review opens no model outcomes and grants no GPU authority.

## Evidence that changes the question

The previous cutwise tangent-state proposal has now had its cheapest falsifier. At
MLP1, all 32-by-32 response halves had numerical support 32 and 95%-energy ranks
between 10 and 17, but none had the preregistered twofold spectral gap. Even at a
forced rank 16, same-context split-half projector distance was 0.5621 against the
0.15 ceiling. The result is **no admitted repeatable local response frame**. This does
not imply that MLP1 is intrinsically high-rank; it says that independently fitting a
small coordinate frame at each context is not presently a stable compiler interface.

The global balance sheet is unchanged:

- all 36 native modules have exact tensor-network formulas;
- the executable rank640 attention program stores 516,707,766 values, a 5.3481%
  reduction from the 545,904,054-value standalone model;
- its strict named causal recovery is 0.57968 / 5.30682 nat = 10.923%, leaving
  4.72714 nat unexplained in that ledger;
- the semantic program covers 32.1% +/- 6.4% in its own evaluation currency;
- dense MLP banks still store 286,675,200 values, 52.51% of the model, with no
  admitted gate removal.

Thus “fraction explained” has no honest scalar answer: structural knowledge is
36/36, executable parameter removal is 5.35%, and strict named causal recovery is
10.92%. The largest live gap is an executable, composition-safe account of the MLPs.
The prior local-frame result specifically directs us toward interventions on physical
multiplication gates and away from another local PCA or response-rank sweep.

At review time the GPU was idle (1 MiB reported), the checkpoint, FineWeb caches,
rank640 parent, and authority machinery were present, and no `rspd`, data, checkpoint,
or compute blocker existed. Unrelated working-tree artifacts were left untouched.

## Ranked move 1: trajectory-complete physical-gate response

### Exact object in bilin18

MLP1 has 4,608 native scalar multiplications

$$
h_n(z_{c,q})=(\ell_n^\top z_{c,q})(r_n^\top z_{c,q}),
\qquad
M(z_{c,q})=b+\sum_n d_n h_n(z_{c,q}).
$$

Insert one scale $\alpha_n$ on gate $n$ **shared at every token position**. For a
registered downstream score $s_{c,a}$, measure

$$
E_{(c,a),n}
=\left.\frac{\partial s_{c,a}}{\partial\alpha_n}\right|_{\alpha=1}
=\sum_q h_n(z_{c,q})d_n^\top g_{c,a,q}.
$$

The position sum is essential. A gate is shared over the sequence, so a gradient at
only the final position is not the derivative of the executable edit. This response
also survives the native scale gauge $\ell_n\mapsto t\ell_n$,
$d_n\mapsto d_n/t$ and is equivariant to gate permutation.

### The mathematics and operational definitions

There are two distinct problems:

1. **Response column subset selection (CSS):** choose physical columns of $E$ whose
   span approximately preserves all measured gate-response directions. Ridge leverage
   uses

   $$
   \lambda=\|E-E_r\|_F^2/r,
   \qquad
   \tau_n=e_n^\top E^\top(EE^\top+\lambda I)^+Ee_n.
   $$

   Published CSS algorithms can select actual columns with near-best low-rank or
   projection-cost guarantees for the measured matrix
   ([Boutsidis, Drineas, and Magdon-Ismail 2014](https://arxiv.org/abs/1103.0995),
   [Cohen, Musco, and Musco 2017](https://arxiv.org/abs/1511.07263)). The deterministic
   top-score pilot implemented here is a falsifiable selector, not itself a claim to
   every guarantee of those randomized algorithms.

2. **Sparse all-on approximation:** choose a common support $S$ and coefficients
   $\beta$ to approximate the model's actual tangent effect

   $$
   E\mathbf 1\approx E_S\beta.
   $$

   A selected span may preserve counterfactual gate directions while failing this
   all-on target, or vice versa. Consequently the implementation reports both
   cross-half projection capture and coefficients fit on one half then transferred to
   the other. Restricted strong convexity can make forward sparse selection weakly
   submodular, but this is an assumption to test, not something granted by the
   transformer ([Elenberg et al. 2018](https://arxiv.org/abs/1612.00804)).

The executable fixed-grammar price of $K$ retained native MLP gates is
$3456K+1152$ floating values plus support/precision metadata and $K$ bilinear
multiplications per token. Response rank, gate count, byte count, description length,
and causal error remain separate currencies.

### Assumptions that may fail

- $E$ is a first derivative at the all-on model. Its low-rank or sparse structure can
  disappear at finite scaling or after simultaneous edits.
- Fisher/probability probes can miss a rare behavior. Context balancing and document,
  role, target-frequency, and intervention strata are mandatory.
- Ridge leverage is a property of the measured response matrix, not a proof that hard
  retention reconstructs the MLP. Refitting Down changes the response object.
- Native gates are checkpoint coordinates. Although their response is invariant to
  scale and permutation, a different exact polynomial factorization could expose
  simpler atoms.

### Predicted consequence beyond reconstruction

If a stable small support exists, it must predict responses on fresh documents and
unseen downstream probes; small finite scales must agree with the tangent prediction;
and retained packages must improve selective removal, mixture composition, OOD
transport, and executable gate count at matched CE/KL. These are uses of simplicity,
not merely a smaller local MSE.

### Cheapest falsifier

At MLP1 in the rank640 shell, compare $K\in\{32,128,512\}$ ridge, response-energy,
activation/Down-norm, factor-derangement, and hash-random supports on independent
document and probe halves. Reject if ridge does not beat every control on both
projection capture and all-on transfer, if support Jaccard is below 0.5, or if a
document stratum fails. The first finite action is only $\alpha=0.9$ on a selected
package. Full removal is forbidden until predicted versus observed Fisher/KL response
passes.

## Ranked move 2: finite nonlinear identification of gate actions

### Exact object in bilin18

After move 1 proposes a few physical gate packages, let $a\in\mathbb R^m$ be their
shared scale changes and let $F_c(a)$ contain downstream registered scores for context
$c$. Identify the controlled suffix map around the live program:

$$
F_c(a)-F_c(0)
\approx J_ca+\tfrac12 H_c[a,a].
$$

This is a small Volterra/Taylor model of **edits**, not a new model of token
activations. It directly represents singleton effects, pair interactions, and the
failure of edit additivity.

### The mathematics and operational definition

Multivariate Taylor's theorem bounds the quadratic truncation error by a cubic term
when the third derivative is bounded on the action region. That assumption can be
tested by scaling curves rather than asserted. Empirical controllability and
observability Gramians use simulated nonlinear input-output perturbations for model
reduction, but provide an approximation tied to the sampled trajectory and perturbation
ensemble ([Himpe 2019](https://arxiv.org/abs/1902.09836)).

Operationally, freeze $m=8$ packages and fit $J,H$ on balanced Rademacher masks at
scales 0.10 and 0.25. Test unseen masks first at the same scales, then extrapolate to
0.50. Define an empirical $\varepsilon$-simulation only over this registered action
and observation algebra: two reduced states are equivalent if every allowed package
action keeps their future observed distributions within $\varepsilon$. Approximate
causal abstractions explicitly relate low-level interventions to high-level ones, but
finite intervention coverage limits the claim
([Beckers, Eberhardt, and Halpern 2020](https://arxiv.org/abs/1906.11583)).

### Assumptions that may fail

- RMSNorm can make derivatives steep near low-norm trajectories, invalidating a wide
  Taylor tube.
- Eight packages may omit a third package that changes a pair interaction.
- A low average error can hide a high-error document or behavioral stratum.
- Empirical approximate equivalence on registered actions is not global
  bisimulation.

### Predicted consequence beyond reconstruction

A successful action law predicts unseen mixtures, supplies an interaction-aware rule
for composing MLP edits, estimates collateral damage before selective removal, and
states the scale at which the tangent compiler ceases to be valid. This directly
addresses the previously observed fact that joint MLP loss can greatly exceed the sum
of singleton losses.

### Cheapest falsifier

Use eight move-1 packages and a small fixed mask design. Reject the quadratic law if it
does not beat the linear law on untouched masks, if error fails to scale cubically in
the small-action regime, or if any preregistered context stratum violates its bound.
This is cheaper and more informative than immediate hard deletion.

## Ranked move 3: intrinsic polynomial tensor rank under the gauge quotient

### Exact object in bilin18

Before its residual addition, an MLP is a vector-valued quadratic polynomial with
intrinsic symmetric coefficient tensor

$$
T_{oij}=\tfrac12\sum_{n=1}^{4608}d_{on}
(\ell_{ni}r_{nj}+r_{ni}\ell_{nj}).
$$

Unlike a list of checkpoint gates, $T$ is invariant under gate rescaling, swapping
$\ell_n,r_n$, and permutation. Its tensor/arithmetic-circuit rank asks for the fewest
bilinear products in *any* exact factorization, not merely the smallest subset of the
existing 4,608.

### The mathematics and operational definition

Every tensor unfolding rank is a computable lower bound on CP/bilinear rank. The
existing coefficient-space audit finds output-flattening rank 1,152 at sampled MLP
sites, so it already proves that this grammar cannot have exact product rank below
1,152, but it leaves a 1,152--4,608 interval. Robust Kruskal conditions can certify
essential uniqueness of a tensor decomposition
([Bhaskara, Charikar, and Vijayaraghavan 2014](https://arxiv.org/abs/1304.8087)), but
the elementary condition cannot apply to the native rank-4,608 decomposition here:
each of the three factor k-ranks is at most 1,152, so their sum is at most 3,456 while
$2R+2=9,218$. Native gate identity is therefore not certified as an intrinsic atom.

A cheap restricted-dictionary audit is the Gram matrix of native input quadratic
forms,

$$
G_{nm}=\tfrac12[(\ell_n^\top\ell_m)(r_n^\top r_m)
+(\ell_n^\top r_m)(r_n^\top\ell_m)].
$$

Full numerical rank, together with nonzero Down columns, would rule out exact deletion
within the native dictionary. It would not rule out a new lower-rank factorization.
Any approximate refactor must be scored by complete executable cost and consequences.
Prequential coding measures how many labels a representation lets a fixed learner
encode as data accumulate and is useful as a prospective tie-breaker among executable
candidates ([Voita and Titov 2020](https://arxiv.org/abs/2003.12298)); it is not an
admission criterion by itself. The broader online/prequential MDL construction is
described by [Blier and Ollivier 2018](https://arxiv.org/abs/1802.07044).

### Assumptions that may fail

- Numerical unfolding or Gram rank can be ill-conditioned and is not approximate CP
  rank.
- Exact polynomial equality before RMSNorm may be needlessly strict for behavior, while
  a behavior-only factorization may fail OOD.
- Border-rank decompositions can be numerically unstable and poor editable programs.
- Prequential codelength depends on the decoder family, precision convention, data
  order, and registered tasks.

### Predicted consequence beyond reconstruction

An intrinsic refactor should preserve the same polynomial under gauge changes, lower
bilinear multiplication count rather than only stored bytes, transfer across corpora
without refitting its atoms, and give more stable extracted/removable components than
checkpoint-coordinate gates. If native Gram rank is full, move 1 can still find
behaviorally dispensable gates, but they must be described honestly as approximate
causal sparsification rather than exact algebraic redundancy.

### Cheapest falsifier

First compute the native quadratic-form Gram spectrum from the checkpoint and verify it
under scale gauges. If it is well-conditioned full-rank, abandon exact native deletion
as a compression route. Only then test a small alternate factorization at MLP1 and
require it to beat native-support programs on multiplication count, finite
consequences, OOD transport, and prequential codelength under a frozen decoder.

## Routes reconsidered and pruned

| route | decision in this review |
|---|---|
| Tensor and arithmetic-circuit rank | Promoted only as move 3's invariant lower bounds and exact product count. No claim that response rank equals tensor rank. |
| Simultaneous factorization/shared dictionaries | Defer until one-site physical supports or intrinsic atoms survive finite action tests. Aligning raw layer coordinates would repeat the failed local-frame assumption. |
| Polynomial invariant theory/gauge quotient | Used to define the legitimate object $T$ and gauge-invariant responses. Computing a full invariant ring across RMSNorm is not the next executable step. |
| Algebraic complexity | Count products and complete decoder/storage cost. Border rank without conditioning, extraction, and finite replay is pruned. |
| System identification/minimal realization | Redirected from the falsified independent tangent frames to move 2's small controlled nonlinear action system. |
| Hankel/weighted automata | Token-prefix Hankel remains pruned because prior splice objects were OOD and bilin18 is not a finite linear automaton. Spectral weighted-automata recovery does have empirical-Hankel concentration guarantees under its own assumptions ([Balle, Carreras, Luque, and Quattoni 2017](https://www.jmlr.org/beta/papers/v17/14-501.html)); those assumptions do not presently match this interface. |
| MDL/prequential coding | A tie-breaker among runnable programs, not a generator of components and not a substitute for causal/OOD tests. |
| Causal abstraction/bisimulation | Incorporated as a finite action-observation consequence in move 2. Discrete clusters are not presumed. |
| Information bottleneck | Pruned: deterministic continuous mutual information needs an arbitrary noise/quantization choice and can erase rare causal variables. |
| Sparse program synthesis | Restricted to physical gates and a frozen response/all-on objective. Generic native-product sparse regression already underperformed the affine baseline and is not repeated. |
| Approximation certificates | CSS/PCP certificates cover the measured matrix; Taylor remainder tests cover a local action tube. Neither is advertised as a global transformer certificate. Spectral sparsification can preserve a quadratic form with few weighted terms ([Batson, Spielman, and Srivastava 2012](https://arxiv.org/abs/0808.0163)), but only after the correct response Gram is fixed. |

## CPU action executed

The highest-priority move now has a fail-closed mathematical kernel:

- `mlp_global_gate_response.py` computes the exact trajectory-complete contraction,
  context balancing, stable numerical ridge leverage, deterministic supports,
  cross-half response-span capture, and separately fitted all-on transfer error;
- `test_mlp_global_gate_response.py` proves equality with shared-scale autograd, rejects
  a position-local substitute, checks scale-gauge invariance and permutation
  equivariance, catches roundoff-only spectral tails, and checks malformed inputs;
- `MLP1_GLOBAL_GATE_RESPONSE_PREREGISTRATION.md` freezes budgets, controls, stability
  gates, executable prices, finite-action order, and forbidden claims.

Focused CPU result: **4/4 tests pass in 1.90 s**. During validation, the initial ridge
implementation treated an exactly rank-three response's roundoff tail as a real ridge
penalty and failed split stability. The numerical-rank branch was corrected while the
test remained strict. This is an implementation/proof result only; it does not claim
that real MLP1 gate responses are sparse.

## Decision

The post-negative ranking is:

1. trajectory-complete physical-gate response and separate CSS/all-on tests;
2. finite nonlinear identification of the surviving gate-package actions;
3. intrinsic polynomial tensor rank/refactorization under gauge, with prequential MDL
   only as a downstream utility tie-breaker.

Move 1 has the highest expected information gain per GPU minute because one assay can
distinguish stable executable gate support from diffuse response, tests real global
edits rather than coordinate frames, and feeds both moves 2 and 3. If it fails, the
correct next move is the intrinsic tensor/Gram audit—not another local rank sweep.
