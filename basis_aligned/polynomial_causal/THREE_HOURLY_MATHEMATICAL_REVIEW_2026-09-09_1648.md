# Mathematical checkpoint — 2026-09-09 16:48 UTC

## Actual object and unresolved goal

The user's controlling reconstruction handoff requires an explicit reusable
computation, OOD/full-distribution prediction, independent execution, accurate
removal/joint effects and reduced structural description including opaque weights.
Our native join evidence does not meet the full goal. Whole-head equality-kernel
replacement is rejected. Context interchange and literal payload simplifications
retain task semantics but miss quantitative fidelity. Do not redefine success.

The current trained restriction is attn4-rms-seed0:4 attention layers,D128,H4,
key/value width32,V29,T51 coldinputs,400640 parameters, affine-free pertoken RMS,
fixed causal mask/absolute RoPE, product QK scores divided by32², residual lerp1/2,
linear final unembedding, no final output normalization or logit softcap. This is
not the full18-layer model. We edit the output of L2; only one attention layer
remains. Selected native join removals are independent additive writes whose
supports are disjoint pairs of token positions. Their values are computed from
native upstream states, not assumed to be constant semantic token functions.

Let b_i∈{0,1} select removal of write d_i. Define z_s(b)=z_s(0)-sum_i b_i d_i(s).
Each token s is touched by at most one i. With n(z)=RMS(z), fixed positional maps
inside Q/K and F_h=W_head O_h, final logits at q are

 L_q(b)=.5 W_head z_q(b)
       +.5 sum_{h,s<=q} [Q1_h(q,n(z_q))·K1_h(s,n(z_s))]
                         [Q2_h(q,n(z_q))·K2_h(s,n(z_s))]
                         F_h V_h n(z_s) /32².

The contraction graph of each summand has one query state and one source state.
It is degree2 in the exposed normalized query vector anddegree3 in the normalized
source vector (degree5 when they coincide). It is not polynomial in unnormalized
residuals because RMS remains explicit. Reciprocal Q/K and compatible head gauges
preserve the scalar contraction; the intervention variables are native output
writes and therefore do not require choosing a latent coordinate gauge.

## Primary literature and exact mapping

Kuo, Sloan, Wasilkowski and Wozniakowski give decompositions using commuting
projections, including anchoring variables to reference values, with uniqueness
under the anchoring conditions and a minimality property excluding interactions
that no summand requires. Here P_i fixes b_i=0. The cube is closed under these
projections; they commute. Apply the construction coordinatewise to logits.
No smoothness, fitting split or independence of natural text is needed for this
finite anchored case. This is not an orthogonal variance decomposition.
[Author preprint](https://web.maths.unsw.edu.au/~fkuo/pubs/preprint/ksww09-decomp.pdf).

On the Boolean subset lattice the coefficients are precisely
I_U=sum_{A⊆U}(-1)^(|U|-|A|) L(1_A), with every other variable anchored at0.
The inverse sums I_U over subsets of the active edits. This is the relevant
instance of incidence-algebra inversion, not a newly discovered transform.
[Rota's original paper](https://webhomes.maths.ed.ac.uk/~v1ranick/papers/rota1.pdf).

Our model-specific consequence is derived next; neither paper claims transformer
circuits or independently executable semantic abstractions.

## Derived restriction, proof, and literal price

Because supports are disjoint, z_q depends on at most one b_i, and z_s on at most
one b_j. Pointwise RMS can be arbitrarily nonlinear without introducing another
variable. Each attention summand consequently depends on at most two edit bits.
On a Boolean domain any function of two bits is exactly constant+unary+pairwise.
The linear residual depends on at most one bit. Summing proves I_U=0 for|U|>=3.

If q is untouched, no summand depends on a query edit bit. L_q is then additive
in the source edit bits and all pair interactions vanish. At a changed q, a pair
term can exist only when another changed source s is causally visible. With
ordered disjoint pairs as in our test, that means the later changed query pair.
This predicts support of the interactions, not just their aggregate size.

The crucial limitation is grouping. Three E/Y0/Y1 edits at the same token all
enter one RMS/QK/V argument. A summand can depend on all three bits, so third
interaction can be nonzero. Native head or algebraic-term boundaries therefore
do not automatically give independently composable units; shared normalized
state is a real coupling boundary. The theorem supplies a grouping criterion,
not a claim that an entire residual position has one semantic function.

The unrestricted transform costs O(m 2^m N) arithmetic and O(2^m N) storage for
N=T·V outputs, after2^m evaluations. Under the proved degree2 restriction only
1+m+binom(m,2) evaluations are needed to recover the entire Boolean response;
for3 edits, seven responses predict the eighth. Pair coefficients can be stored
only at their allowed changed-query positions; unchanged finalqueries need only
unary coefficients. Uniqueness is relative to the fixed baseline and edit values,
not uniqueness of the model's causal units. Different removal amplitudes require
new function evaluations unless an explicit continuous reader is retained.

This does not beat one direct native forward for one intervention. It replaces an
exponential response table for repeated combinations by a quadratic response
program, conditional on the same token input and fixed edits. All native400640
coefficients and response-generation computation remain charged. No parameter,
state-memory or walltime reduction of the trained model is claimed. Tensor banks
stay below256MiB; the proposed three-edit audit has16 response arms,B4FP64,T51.

## Why nearby methods do not replace this consequence

CLUE supplies exact linear closure for polynomial ODE reductions. Our immediate
object is a finite intervention cube with pertoken normalization and a fixed
feedforward suffix; the Boolean support proof applies directly without inventing
an ODE or polynomial normalizer state. CLUE does not give the semantic quotient
or justify discarding contextual terms here.
[CLUE](https://arxiv.org/abs/2004.11961).

Weighted-tree-automaton/Hankel methods concern minimal multilinear compositional
realizations. Our native RMS transitions do not satisfy that class's multilinear
state assumptions; flattening observed responses into a low-rank table would be
a probe, not the proved normalized-state grouping criterion.
[Weighted tree automata](https://proceedings.mlr.press/v51/rabusseau16.html).

Contraction-width methods address evaluation cost of a supplied network. They
motivate examining the query/source factor graph, but do not by themselves show
semantic identifiability or replace the variable-dependence argument above.
[Markov and Shi](https://arxiv.org/abs/quant-ph/0511069).

Softmax across sources, a downstream normalization after the attention sum, or
another attention layer would generally introduce additional edit variables
into a summand. Overlapping write supports also violate the premise. The theorem
must not be applied unchanged to arbitrary earlier layers or the fullbilin18
architecture. The anchored expansion itself still exists, but low order is then
unproven. Log_softmax/CE/KL likewise introduce nonlinear interactions after logits;
nonadditive loss does not refute raw-logit composition.

## Executable consequence and next action

`source_support_interaction_reference.py` and
`SOURCE_SUPPORT_INTERACTION_CONTROLS.json` implement seven passing controls.
Three disjoint FP64 source edits have thirdinteraction4.16e-16 and forbidden
pair leakage3.89e-16, with all allowed pairs live~.001. Three overlapping edits
have thirdinteraction.00112. An integer, norm-free product-attention fixture
has exactlyzero thirdinteraction for disjoint supports and416 for overlapping
supports. These controls verify the instrument and algebraic restriction, not a
trained semantic program. No trained interaction outcome has been opened.

The selected successor is `SOURCE_SUPPORT_INTERACTION_V1_PREREGISTRATION.md`:
fresh three-chain IID/three8-cycle worlds, native disjoint join removals, withheld
triple prediction, pair-support localization, overlapping producer control and
log-probability nonadditivity control. This can change grouping and composition
claims directly; it is preferable now to another local-field sufficiency sweep.
The native origin-copy result remains useful but not quantitatively sufficient.
