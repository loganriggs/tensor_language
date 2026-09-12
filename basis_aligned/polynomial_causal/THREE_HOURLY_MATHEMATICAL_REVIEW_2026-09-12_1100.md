# Three-hour mathematical review — 12 September 11:00

The full target remains a simpler executable explanation with OOD prediction,
extraction, selective removal and composition/reuse. The source block now has
prospective regional-spelling and unseen-cue/lexicon removal screens; it is not
yet a four-property circuit. The next mathematical question is where the regional
cue acts inside its explicit shared computation, rather than another rank sweep.

## Actual object and contraction

For query position $t$ and source $s$, the tuple $x_s\in\mathbb R^{2304}$ joins
current attention17 input and first-attention input. Query $q_t\in\mathbb R^{1152}$.
The four source readers give one quadratic parent and two child readings:

$$
z_s=(a^Tx_s)(b^Tx_s),\qquad u_{sj}=c_j^Tx_s,\quad j\in\{1,2\}.
$$

Private quadratic query writers $\beta_{hsj}(q_t)\in\mathbb R^{1152}$ are the
exact eliminated coefficients for these two source features in the frozen
16-function dictionary. Head $h$ ranges over9; rotary position is compiled into
their constants. Native head gate $g_{hts}$ includes all four QK RMS denominators
and128-squared scaling, divided by discovery reference scale $g_{0h}$ already
absorbed in the writer. The candidate write is

$$
b_t=\sum_{s\le t}\sum_{h=1}^{9}\sum_{j=1}^{2}
g_{hts}\,z_s\,u_{sj}\,\beta_{hsj}(q_t).
$$

Without gates, it has source degree3 and query degree2. The downstream native
MLP17 adds quadratic interactions between this write and the retained residual;
RMS denominators and final tanh prevent a global polynomial-logit identity.
Keep those operations explicit in every intervention. Source0/previous-only
coverage and all-source coverage are different programs; the latter passed the
behavioral screen. All current/base inputs, gates, private query coefficients,
background and downstream weights remain required dependencies.

The graph has a two-consumer parent, not an assumption that one native head is
one circuit. Native source readers can be rescaled with inverse writer changes;
$a,b$ may exchange roles. The child pair has invertible basis freedom with inverse
writer action. Swap the child pair as a bundle; individual-child attribution can
otherwise depend on the chosen basis. The source parent is still a candidate
semantic variable, not automatically a regional-style label.

Runtime extraction reads four source vectors per token, while the other14source
functions enter the constant coefficients defining $\beta$, not additional
runtime source scalar reads. The current prototype still computes private writers
through all16 coefficient contractions. Literal price must include those folded
query/output constants and native gates; four source vectors alone is not the
full program cost. No entire unembedding tensor is stored.

## Primary literature mappings

[Geiger et al., Causal Abstractions of Neural Networks (2021)](https://arxiv.org/abs/2106.02997)
uses alignments between neural representations and high-level causal variables,
then interchange interventions to test their causal correspondence. Our mapping
is operational: the extracted parent, child tuple, writer tuple and gate tuple
are four variables, with paired-context values as donors. An exact replay of
these ports verifies implementation but does not establish a semantic abstraction.
We still need a high-level prediction about cue transfer and counterfactual
behavior. A finite screen neither proves universal correspondence nor learns
an alignment for us. No interchange-intervention training or data fitting is used.

[Kuo, Sloan, Wasilkowski and Wozniakowski, On Decompositions of Multivariate
Functions (2010)](https://web.maths.unsw.edu.au/~fkuo/pubs/preprint/ksww09-decomp.pdf)
provides decomposition formulas encompassing anchored decompositions. Here each
coordinate is a whole port array. Holding a coordinate at its recipient value is
a commuting idempotent projection, so inclusion-exclusion isolates anchored
interactions exactly. This is not variance-based ANOVA: the four native arrays
are statistically dependent, and no product data distribution or orthogonality
is assumed. The result is unique for the chosen ports and anchor, not invariant
to arbitrary regrouping or a proof of semantic minimality.

These matches address the current intervention object more directly than CP,
Tucker, tensor-train or hierarchical-rank theorems: the source graph is already
frozen, and the unresolved question concerns its causal variables. Earlier exact
source projection, secant-block and rank-ill-posedness work remain relevant
representation tools; none supplies the missing causal identification theorem.

## Exact executable consequence

Let $F(S)$ be the write, or the final logit contrast after replacing the candidate
write, with ports in subset $S$ taken from the donor and all other ports plus
native background retained from the recipient. There are four ports, hence16
vertices. The anchored interaction for subset $S$ is

$$
I(S)=\sum_{T\subseteq S}(-1)^{|S|-|T|}F(T),\qquad
F(S)=\sum_{T\subseteq S}I(T).
$$

At the multilinear write level, each $I(S)$ is the contraction with donor-minus-
recipient factors in $S$ and recipient factors elsewhere. At the nonlinear logit
level, evaluate the16vertices and apply the same finite transform; it remains
exact on this intervention grid, with no polynomial approximation of RMS/tanh.
Computing all vertices costs16candidate contractions/suffix evaluations, and the
transform costs $O(4\,2^4)$ output-array additions. No giant tensor is needed.

The [executed control](FACTORIAL_SOURCE_PORTS_V1_CONTROL.json) verifies the direct
product-difference expression, all-vertex reconstruction and a nonlinear
normalized/tanh example within2.9e-16. The
[extracted-port control](COMMON_QUADRATIC_PORTS_V1_CONTROL.json) agrees with the
original selected two-child executor within2.75e-16. This is a concrete tool for
identification and composition testing, not merely a literature analogy.

Decision: use the existing paired cue rows to distinguish a parent-mediated
regional signal from query-writer-, child- or gate-mediated behavior. A parent
swap that does not carry the cue would narrow the parent's role to a reusable
operation; it would not erase the whole block's held-out selective contribution.
Do not force that role to migrate to whichever factor scores highest without a
new held-out test. Record interactions rather than assume single-port effects add.
Next math review14:00; hourly review11:36. Full goal remains active.
