# Mathematical review: an interaction that a reusable cue encoder must explain

Due13:49 UTC; restored current state13:50 and recorded at13:54. Original bilinear
handoff/pilot and its appended discovery criterion govern. The target remains
OOD prediction, extraction, selective manipulation, composition/reuse and fewer
independently specified computations. No new circuit is promoted by this review.

The current useful object is a fixed trained correlative interface:26 attention
heads grouped into14 blocks (layers3 through16), one scalar per block. It transfers
both/neither on new word combinations with raw recovery .969/.899. Exact weight
folding works, but first-value-only and local-value-only transfers fail. Their
joint downstream effects also fail addition: full-vocabulary interaction .313/.286.
The proposed completed-inner-correlative test was already run as v493/v495/v497;
the unmatched neutral phrase was shorter. Correct that prior-art error before
building. The next experiment fixes length and changes only either to only.

For token positions t,j, model widthD=1152, head widthd=128, layer l and selected
head h, the actual folded computation is

    s_l(t) = sum_h sum_{j<=t} p_lh(t,j)
             [r_local_lh^T x_l(j) + r_first_lh^T x_0(j)],
    delta_x_l(t) = w_l [s_l(donor) - s_l(live recipient)].

x_l and x_0 are native RMS-normalized attention inputs; r vectors and w have
width1152. p is the product of two dot products of independently head-normalized
Q/K projections after native rounded RoPE, each divided by128. There is no
softmax. Local/first readers include actual signed value-mix coefficients. The
writer is c_proj folded with the per-block head direction. Thus p and value
content are contracted over source position; layers remain connected through
the full native residual computation. First values share x_0 across consumers,
but local inputs, routing and independently editable histories remain distinct.

The local contraction is bilinear in independently supplied p and value
projections. Without normalizers, its product-QK/value form would have degree5
in free local input vectors. In the actual model, RMS square roots and output
tanh prohibit treating the complete token-to-logit map as a fixed low-degree
polynomial. The18-layer model has9heads/layer, MLPwidth4608 and50304 logits.
The MLP on its normalized input is a quadratic three-index tensor
T[a,b,c]=sum_m Down[a,m] Left[m,b] Right[m,c], plus output bias; tying the two
input vectors exposes only its symmetric input-index part. Decomposing that
tensor alone need not determine downstream causal units.

Allowed inputs here are64 authored matched outer-cue pairs and their registered
swaps. Native background and suffix are explicit dependencies. Outputs are task
margin and full prediction-position logits; compiler norms are maxabs and
relativeL2. All545902902 native parameters remain charged, plus the existing
76032 reader/writer coefficients; no structural saving. The matched test costs
24 body forwards/384 sequences, no updates. Rank-one sign changes negate both
s_l and w_l, preserving their product; use write-weighted errors or absolute
scalar contrasts for sign-invariant conclusions. General changes of basis are
not automatically legal through the native products or normalizers.

Three candidate mathematical routes have exact but restricted mappings:

* [CLUE](https://arxiv.org/abs/2004.11961) finds the minimum linear lumping of a
  polynomial ODE while retaining prescribed linear observables. Mapping: choose
  observable rows L and seek closure under the coefficient matrices of the
  polynomial Jacobian. For a discrete map F, the corresponding differentiable
  condition is L DF(x) ker(L)=0 on the whole domain; on connected affine fibers
  it makes LF constant on each fiber. Layer-dependent normalized transitions
  and donor interventions violate the polynomial autonomous ODE assumptions.
  A naive queue closure over N dense n-by-n coefficient matrices has at most n
  accepted rows and O(N n^3) arithmetic with maintained elimination (our bound,
  excluding polynomial expansion). The minimal invariant row space is fixed
  by the observables; its basis is not unique. We lack a tractable coefficient
  representation of the full native transition, so no global minimum claim.

* [Rabusseau et al.](https://proceedings.mlr.press/v89/rabusseau19a.html) connect
  linear second-order recurrent networks to weighted automata and recover them
  from sufficiently informative Hankel blocks. Mapping: tokens select linear
  state transitions and an observable suffix reads the resulting state. Exact
  finite Hankel rank, complete prefix/suffix bases and linear state updates
  permit minimal realization up to a state-basis transformation. For a finite
  p-by-q block, dense SVD costs O(pq min(p,q)), before shifted-block contractions.
  Our contextual multilayer normalized producers are not known to be such a
  finite linear realization. Four context corners are not a complete Hankel
  basis; a rank observation here cannot certify a minimal reusable parser.

* [Geiger et al.](https://www.jmlr.org/papers/v26/23-0058.html) supply causal
  abstraction semantics: low-level interventions must correspond to operations
  in a specified higher-level model. Mapping: the14 scalar patches are native
  operations; a proposed cue state must predict their effects across contexts.
  This is a consistency framework, not automatic identification or unique
  recovery. Exhaustive finite checking costs contexts times interventions times
  native evaluation cost. We can afford this restricted panel, while a universal
  abstraction remains unestablished. Passing answer-preserving swaps alone does
  not identify a shared state or prove independence of its contextual producer.

The tensor/neighboring-literature search also checked identifiability, polynomial
relations and graph contraction. [Robust Kruskal results](https://arxiv.org/abs/1304.8087)
need suitable factor-column independence for a specified CP tensor; we have no
such condition for these tied, overcomplete and normalized causal programs.
[Graph-width contraction](https://arxiv.org/abs/quant-ph/0511069) bounds evaluation
cost in graph width, without discovering semantic operations. [Vanishing
component analysis](https://proceedings.mlr.press/v28/livni13.html) proposes
relations on observed points; sample vanishing alone supplies no intervention
closure. Tensor trains/HT change contraction parameterization, and arithmetic
bilinear complexity prices local multiplication; neither bypasses these missing
closure and causal assumptions. These are not new decomposition sweeps.

An executable consequence follows without fitting. At fixed reporter/frame/layer,
write s(c,k) for the native scalar with outer cue c in {both,neither} and context
k in {only,either}. An additive reusable cue encoder proposes

    s(c,k)=a(c)+b(k).

Let D=s(neither,either)-s(both,either)-s(neither,only)+s(both,only).
Every additive table has D=0. Conversely on exactly this2-by-2 domain, D=0
is sufficient for such an additive representation (its offset allocation is
nonunique). In corner order(both,only),(neither,only),(both,either),(neither,either),
v=(1,-1,-1,1) is orthogonal to the three-dimensional additive subspace. Projection
leaves residual(D/4)v, so minimum squared error is D^2/4 and minimum L2 error is
|D|/2. For any maximum corner error epsilon, triangle inequality requires
epsilon>=|D|/4. With numerical scalar errors e_i, replace |D| by
max(0,|D_observed|-sum_i e_i) for a bridge-aware discrepancy bound. Those e_i
bound the observed folded-versus-native scalar disagreement, not arbitrary
hardware/platform uncertainty. This is our elementary finite-domain derivation.

For vector writes w_l s_l, the same result holds in Euclidean norm with
|D| replaced by ||w_l D||. Stacking separate block writes gives a direct-sum
diagnostic; it must not be confused with final-logit error after nonlinear
downstream recomputation. A nonzero result rejects additive separation of the
FIXED coordinates on these inputs, not all nonlinear representations or a
different circuit. A zero result alone says nothing about intervention closure.

The immediate tool is therefore the frozen matched four-corner native capture,
its full-head reachability ceiling and same-answer scalar context swaps. It can
distinguish a simple additive cue encoder from a context-dependent computation
at low cost. A positive context interaction plus poor neutral transfer would
close this additive explanation, not justify a rank/gain rescue. A selective
matched deficit would motivate identifying which context-sensitive operations
change transfer, with lexical/syntactic confounding stated. This has higher
information value than another local tensor-rank or exact recurrence fixture.
