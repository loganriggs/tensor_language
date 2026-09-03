# Three-hour mathematical review — 2026-09-03 04:20 UTC

## Circuit target and exact model object

The program target remains a smaller transparent tensor program that predicts fresh and out-of-distribution text,
composes when several replacements are installed, supports selective removals/swaps/edits, and improves a literal
storage/compute/edge/state/program price. Its units must specify what is read, what operation is performed, what is
written, and which downstream computations use the write. They may join pieces across native heads or MLPs and split
one native component. Rank and reconstruction are controls or prices, not circuit evidence.

The next object is attention layer 8's output tensor `Y[b,t,o]`, with residual index `o=1..1152`, at the output of the
head concatenation and output projection. For a recipient activation `y` and a natural donor activation `d`, a rank
`r` subspace with orthonormal columns `Q in R^(1152 x r)` defines the interchange

`I_Q(y,d) = y + ((d-y)Q)Q^T = y + (d-y)P`, where `P=QQ^T`.

The model weights stay fixed. The intervention changes only the selected part of the actual attention8 write, then
runs layers 9 through 17 and the final readout normally. Let `F_x(y)` be that nonlinear suffix for input `x`, and let
`ell(F_x(y))` be per-token cross-entropy. The finite response is

`Delta_P(x,d) = ell(F_x(I_P(y_x,d))) - ell(F_x(y_x))`.

The shared/private proposal uses mutually orthogonal projectors `P_S,P_1,P_2,P_3`, provisionally of fixed rank 4.
Circuit `i` receives `P_S+P_i`, a rank-8 matched capacity. The full experimental union has rank 16. Storing an
orthonormal basis costs `1152*(4+3*4)=18,432` floating values; one rank-8 application costs approximately
`2*1152*8` multiply-adds per patched token. This is an experimental price, not a compression claim.

The local intervention is linear in `y,d,P`, but `F_x` is not polynomial because every later block has RMS
normalization and the logits use a tanh soft cap. Its unnormalized attention contractions are degree five in normalized
inputs and its MLP contractions are degree two. Thus finite response composition cannot be inferred from polynomial
degree or from adding single-intervention loss effects.

Allowed inputs are the frozen 1,000-row FineWeb census and natural within-split donor maps. Primary target masks are
the mutually exclusive cells of `r.2.0.2`, `r.2.1.1`, and `r.2.2.1` relative to the entire historically known
four-circuit attention8 cluster; `r.2.0.1` is reserved as a reuse target. The three training targets overlap pairwise
on 185, 193, and 208 of 864 member positions, with 77 positions in their triple intersection, so raw masks cannot
identify reuse. Outputs to preserve are signed per-token causal responses, member-minus-matched-control effects,
all other attention8 circuits, and ordinary off-target CE.

The important symmetries are:

- `Q` and `QR` define the same projector for every orthogonal `R`; individual basis columns are gauge, while `P` is
  the invariant object;
- without an orthogonality or sequential-fit rule, directions can rotate between the shared and private spans while
  their union and every intervention remain unchanged;
- attention has its own internal matched query/key basis changes and head permutations, but an output-space projector
  after the output projection avoids assuming those internal gauges are semantic.

Projector agreement will therefore use principal angles or normalized projector overlap
`tr(P_a P_b)/r`, not column cosine. The causal approximation norm is optimally scaled response residual plus signed
response cosine on unseen documents/donors, accompanied by unscaled physical amplitudes and collateral CE.

## Closest theorem 1: joint and individual variation

JIVE decomposes several data blocks as `X_i=J_i+A_i+E_i`, with a common low-rank row space and block-specific row
spaces. Its orthogonality of joint and individual row spaces makes those matrices identifiable under the stated
linear model; no pairwise orthogonality among all individual parts is required. The direct source is Lock et al.,
[Joint and Individual Variation Explained](https://pmc.ncbi.nlm.nih.gov/articles/PMC3671601/) (2013).

The tempting mapping is:

- blocks `X_i`: causal response tables for the three circuits;
- joint structure `J_i`: the response produced by `P_S`;
- individual structure `A_i`: the conditional response produced by `P_i`; and
- orthogonal row spaces: geometrically orthogonal activation projectors.

This is not an exact theorem for our problem. JIVE is a linear additive matrix model with a Frobenius objective. Here
the learned variables are activation projectors before a nonlinear suffix, circuit masks overlap, and rung520 found
that adding singleton loss responses predicts a grouped intervention with median relative error 8.51 on tasks and
9.68 on circuits. Orthogonality in activation space does not imply additive or orthogonal causal responses. JIVE
therefore justifies a **parameterization and identifiability guard**—fit shared first, freeze it, fit each private part
in its orthogonal complement—but cannot certify a circuit.

Executable consequence: never optimize an unconstrained shared/private union. Fit a shared projector with a
leave-one-circuit-out maximin causal objective, freeze a training-only Grassmann representative, and then fit each
private projector only to the remaining response. Require exclusive-cell held-out interventions to prove which part
is genuinely shared or private.

## Closest algorithm 2: Grassmann optimization and subspace comparison

Edelman, Arias, and Smith develop optimization directly on Stiefel and Grassmann manifolds, where orthonormal frames
parameterize subspaces and quotienting by right-orthogonal rotations removes basis gauge
([primary paper](https://math.mit.edu/~edelman/publications/geometry_of_algorithms.pdf)). Björck and Golub show that
principal-angle cosines are the singular values of the cross-product of orthonormal bases
([primary paper](https://rainbow.ldeo.columbia.edu/~alexeyk/BjoerckGolub1973.pdf)).

Exact mapping:

- optimization variable: `P=QQ^T` in `Gr(r,1152)`;
- product constraint: `[Q_S,Q_1,Q_2,Q_3]` lies on `St(1152,16)`;
- retraction: replace a step `U` with its polar orthonormal factor `U(U^TU)^(-1/2)`;
- restart stability: singular values of `Q_a^T Q_b`, or `tr(P_aP_b)/r`.

These algorithms give a correct constraint-preserving optimizer and gauge-invariant stability metric. Their cost per
retraction is `O(1152*r^2+r^3)`, negligible beside a transformer forward/backward for `r<=16`. They do not make the
causal loss convex, guarantee a global optimum, or identify semantic variables. Five fixed seeds, optimizer-health
checks, and fully retrained label-permutation controls remain necessary.

## Closest theorem 3: causal representation learning from interventions

Bing et al. establish identifiability of linearly mixed causal variables from sufficiently diverse multi-node
interventions, using changes across intervention environments
([primary PMLR paper](https://proceedings.mlr.press/v236/bing24a.html)). Their theorem requires a latent causal model,
a linear observation mixture, and intervention coverage/diversity assumptions that expose the latent variables.

Our activation `y` is linearly observed at one site, but the donor projection is designed by us rather than an unknown
environmental intervention on independent latent mechanisms; the downstream target is a nonlinear deterministic
suffix; and we do not know that the desired circuits are coordinate-wise latent variables. The theorem does not
identify `P_S` or `P_i`. What transfers is a falsifier: one donor map is insufficient. The same projector must predict
responses for disjoint donor ensembles, both interchange directions, and a different action—mean-centered projection
removal. A direction that works only for its training donor is steering, not a recovered causal variable.

## Multiple mediators and exact finite composition

Vaidyanathan et al. show that a single activation-patching natural indirect effect includes interactions with other
mediators and that these interactions decompose into pairwise and higher-order group terms
([primary preprint](https://arxiv.org/abs/2606.27510)). Rung520 measures the practical size of the problem here: the
joint source-star response is almost an order of magnitude away from the sum of its individual responses.

For the four candidate projectors `N={S,P_1,P_2,P_3}`, define `E(A)` as the actual finite loss response after jointly
installing subset `A`. The exact Möbius interaction is

`m(A)=sum_(B subset A) (-1)^(|A|-|B|) E(B)`.

Then `E(A)=sum_(B subset A) m(B)` is an exact algebraic identity over the measured 16 subset endpoints. This does not
assume CE additivity. It is the right tensor-program composition assay. The same factorial must be repeated with
attention6 native versus intervened, because all three targets also depend strongly on attention6; a sign-changing
background dependence identifies an `a8 x a6` interaction rather than an autonomous attention8 variable.

## Executable consequence and decision

No located theorem exactly solves the nonlinear causal shared/private problem. The literature does, however, turn
three common failure modes into concrete gates:

1. **Power before optimization.** On training documents, run two independent whole-attention8 donor ensembles. Abort
   before fitting unless every exclusive target has material, selective, donor-stable responses and the 32-circuit
   effect pattern reproduces across document halves. This directly tests whether Claude's `0.016` source-star
   self-correlation problem also affects the actual DAS object.
2. **Projectors, not directions.** Optimize and compare Grassmann projectors. Fit the shared span first with each
   target held out in turn, freeze it, then fit private spans in its orthogonal complement. Never interpret columns.
3. **Reuse and splitting by intervention.** The fourth historically known cluster member must accept the frozen shared
   projector; every private projector must add its owner's residual effect without adding the same effect elsewhere.
   All other attention8 circuits are negatives or evidence that the unit is broader than the proposed cluster.
4. **Finite composition.** Run all 16 subsets, both swap directions, mean removal, and the attention6 background
   factorial. Do not predict joint CE by summing marginal effects.

This dominates immediately collecting a larger corpus. Rung520's power failure is on a different, much smaller
MLP10 source-star effect; archived whole-attention8 interventions are 5.07–7.14 times more concentrated on these
targets, so the exact-object power gate can distinguish “DAS is currently measurable” from “raise N first” in a few
hundred forwards. If it fails, corpus expansion becomes mandatory. If it passes, shared/private DAS tests grouping,
splitting, held-out prediction, reuse, stable identification, and manipulation directly. The next action is to freeze
that two-stage rung and implement the power kill-switch before any gradient outcome is allowed.
