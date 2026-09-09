# Three-hour mathematical tensor-network review — 2026-09-09 02:26 UTC

## Exact current object: tokens, directed attention cells, and a nonlinear suffix

Let `B=64`, padded length `T=9`, residual width `D=1152`, head width `P=128`, heads
`H={L8H1,L9H1,L9H4,L11H3}`, and vocabulary size `V=50304`. A row begins with discrete token IDs
`i[b,t] in {0,...,V-1}`. The tied embedding lookup gives `x[0,b,t]=E[i[b,t]]`. At a selected head,
normalized/rotary query and key vectors and the output-projected effective value are

\[
q^1_{bth},k^1_{bsh},q^2_{bth},k^2_{bsh}\in\mathbb R^P,
\qquad u_{bsho}\in\mathbb R^D.
\]

For destination role `d` and source role `s`, the exact head write is

\[
C_{bhdsto}=1[t\in d]\sum_{r\in s,\,r\le t}
\left(\frac{q^1_{bth}\cdot k^1_{brh}}{P}\right)
\left(\frac{q^2_{bth}\cdot k^2_{brh}}{P}\right)u_{bhro}.
\]

The role inventory is `changed`, `unchanged_prefix`, and `matched_suffix` on both axes. Hence each
head has nine cells. The contraction graph has two QK inner-product branches sharing destination
and source indices, their scalar product, an effective-value branch, source summation, the head slice
of `W^O`, and residual addition. As a function of the exposed factors `(q1,k1,q2,k2,u)`, one term is
multilinear of degree five. It is not degree five in token IDs or residual states: embedding lookup is
discrete, RMSNorm introduces an inverse square root, rotary maps are position dependent, later MLPs
are bilinear, and later attention repeats the multiplicative routing.

Parameters are tied over rows and positions. Physical residual coordinates are fixed by the
checkpoint and later readers. Private head coordinates admit invertible gauge changes only when the
corresponding Q/K/V/output factors are transformed together; raw private columns are therefore not
semantic units. Role cells are operationally fixed by token equality and semantic positions, so they
do not inherit that private gauge. MLP hidden factors retain permutation and reciprocal-scaling
symmetries. The allowed v23 inputs are the sealed 64 token pairs, not arbitrary English; structural
OOD remains untested.

For a selected set of cells `S`, let `F_b(S)` be the final signed `is`/`was` margin after installing
the summed cell delta into an otherwise native base run. `F` includes the complete nonlinear suffix.
The outputs to preserve are the 30 capable A1/A2 row effects plus low P/C response; approximation is
measured by signed projection, cosine, direction fraction, relative residual, and control RMS ratio.
The v4 price is exactly 84 forwards and 5,376 sequence evaluations: two native, six captures, and
four heads times nine singleton sufficiencies, nine singleton resets, and one full replay. It stores
no fitted parameters. A standalone extracted program would additionally price embeddings/early
state production, four restricted attention contractions, downstream readers, RMSNorm, unembedding,
and softcap; those missing native pieces cannot be priced away yet.

## Exact cooperative-game mapping: what reciprocal arms do and do not identify

For one head's nine cells, define the centered set function

\[
v(S)=F(S)-F(\varnothing),\qquad S\subseteq N,\quad |N|=9.
\]

The unique Möbius or Harsanyi decomposition is

\[
\delta(T)=\sum_{S\subseteq T}(-1)^{|T|-|S|}v(S),
\qquad v(S)=\sum_{T\subseteq S}\delta(T).
\]

This is the exact object behind cooperative-game interaction dividends, introduced in Harsanyi's
cooperative-game formulation ([Harsanyi, 1963](https://doi.org/10.2307/2525487)); Shapley's value is
one symmetric allocation of these coalition effects ([Shapley, 1953](https://doi.org/10.1515/9781400881970-018)).
The mapping is literal: players are source/destination cells and the game value is a rowwise output
margin or a fixed linear projection of its vector. No theorem makes a dividend a natural-language
feature; the meaning comes only from the frozen role intervention and controls.

V4 measures two exact marginal quantities for every cell `e`:

\[
S_e=v(\{e\})=\delta(\{e\}),
\]

\[
N_e=v(N)-v(N\setminus\{e\})
=\sum_{T\ni e}\delta(T).
\]

Therefore

\[
N_e-S_e=\sum_{T\ni e,\,|T|\ge2}\delta(T)
\]

is an exact signed total of all higher-order interactions involving `e`. This is the mathematical
reason reciprocal sufficiency/reset is useful: with `2|N|+1=19` arms per head, it identifies the
singleton effect and that cell's total interaction burden at the grand coalition. It does **not**
recover each interaction or the Shapley allocation. Exact unrestricted Möbius recovery requires all
`2^9=512` coalition values per head; no algorithm can recover an arbitrary set function from fewer
queries without structural assumptions, because it has 512 independent degrees of freedom.

The registered signed projection is linear in the intervention-effect vector, so the interaction
bracket is directly computable from the saved v4 reports. For the reciprocal cells, reset-minus-
sufficiency projection gaps are approximately:

| head | cell | higher-order projected burden |
|---|---|---:|
| L8H1 | matched suffix <- changed | 0.0896 |
| L8H1 | matched suffix <- matched suffix | 0.0854 |
| L9H1 | matched suffix <- changed | 0.0506 |
| L9H1 | matched suffix <- matched suffix | 0.0506 |
| L9H4 | matched suffix <- changed | 0.0674 |
| L9H4 | matched suffix <- matched suffix | 0.0674 |
| L11H3 | matched suffix <- matched suffix | 0.00744 |

Thus L11H3 is nearly singleton-additive at this role resolution, while the earlier heads have modest
positive synergy. This makes a 2,048-forward four-head Shapley enumeration poor value now: it would
allocate interactions but would not supply the missing token-state generator or downstream reader.

## Relation to exact tensor contraction and causal abstraction

Once a downstream reader map `R_j` is identified, the weight-side directed edge is the contraction

\[
E_{jhds}=R_j\,C_{hds},
\]

or its bilinear/finite-dose generalization when the reader depends on live context. This is the
correct point to fold a causal subspace into weights and then consider PCA, SAE, Tucker, or a
hierarchical factorization: the input and output axes have operational meaning. Before `R_j` is
identified, decomposing `C` describes possible writes but not who uses them.

Interchange interventions provide an operational test for whether a proposed high-level variable is
implemented by a low-level state; causal abstraction work formalizes this through aligned
interventions rather than decoder accuracy alone ([Geiger et al., 2021](https://arxiv.org/abs/2106.02997)).
Our source/destination swaps instantiate only the writer side of that mapping. The assumptions for a
complete abstraction are violated until the same variable predicts downstream interchange across
new parses and composes with identified readers.

If the final object becomes a pure tensor network, exact contraction cost is governed by contraction
order/treewidth; Markov and Shi connect tensor-network contraction complexity to graph width
([Markov and Shi, 2008](https://doi.org/10.1137/050644756)). The whole current circuit is not such a
fixed multilinear network because RMSNorm and softcap are data-dependent nonlinear gates. They may
be retained as explicit program nodes, but a treewidth theorem cannot justify deleting or replacing
them. Weighted-automata/Hankel minimal realization likewise does not apply without a finite-rank
linear state update; the transformer's normalized multiplicative routing violates that premise.

## Executable consequence and route decision

The immediate zero-forward consequence is to publish `N_e-S_e` for every cell, not another Shapley
game. It will identify which reciprocal writer edges are already close to additive and which require
a grouped interaction test. The opposing predictions are:

- if the role cells are stable compositional units, interaction burdens remain small and a sum of
  reader-contracted singleton predictions matches the full finite-dose intervention;
- if role labels hide essential coalitions, reset/sufficiency gaps are large or change sign across
  halves, and singleton edge compilation must be rejected or expanded prospectively.

Empirically, the dominant missing information is still `R_j`, so the 28-forward reciprocal A12--M17
reader atlas dominates an exhaustive 512-coalition game. In parallel, a structure-diverse v25 bank
dominates another noun-swap confirmation because the allowed-input assumption is currently the
largest identification gap. After a reader validates, the first composed test should contract only
the seven reciprocal v4 cells into that reader, predict their signed finite-dose effects, and compare
those predictions with transfer/reset on held-out rows. Failure kills the claimed directed tensor
program; success licenses weight decomposition of the causally restricted edge tensor.

