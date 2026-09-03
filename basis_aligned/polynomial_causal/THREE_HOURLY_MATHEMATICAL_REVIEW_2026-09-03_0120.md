# Three-hour mathematical review — 2026-09-03 01:20 UTC

## The exact object we are trying to simplify

The target is not a low-rank weight matrix. It is the input/output computation of the 18-block bilin18 model,
together with named interventions that let us extract and selectively change circuits. A sequence has at most 256
positions and each residual-stream vector has dimension 1,152. Each attention layer has nine heads of width 128.

For one attention head, after the model's RMS normalizations and rotary position map, the score from key position
`k` to query position `q` is

`p[b,h,q,k] = <Q[b,q,h], K[b,k,h]> <Q2[b,q,h], K2[b,k,h]> / 128^2`

when `k <= q`, and zero otherwise. Its write is

`A[b,q,o] = sum_(h,k,d) O[o,h,d] p[b,h,q,k] V[b,k,h,d]`.

Thus the unnormalized local attention contraction is degree five in its normalized inputs: four factors make the
two dot products and one is the value. At MLP11,

`M[b,t,o] = sum_r Down[o,r] Left[r,:]z[b,t] Right[r,:]z[b,t] + bias[o]`,

with residual index `o=1..1152` and bilinear hidden index `r=1..4608`. This local map is degree two. The full model
is not a polynomial in token embeddings because RMS normalization divides by a state-dependent square root and the
final logits use a tanh soft cap. Consequently a local tensor identity does not imply equality after the nonlinear
suffix; finite model execution is the relevant test.

Rung515 starts with six fixed nonempty MLP10 branch changes, four fixed implementations of the equality score, and
the exact next-layer changes they induce. Attention11 is expanded by Boolean-cube inversion into all 31 nonempty
interactions among `Q,K,Q2,K2,V`; MLP11 is expanded into its Left-only, Right-only, and Left-by-Right changes. A node
is therefore

`u = (MLP10 branch subset, score implementation, site, exact term)`.

There are `6 * 4 * (31 + 3) = 816` nodes. Candidate comparisons keep the branch subset and site fixed and compare
the three registered implementation pairs, giving

`6 * 3 * (31^2 + 3^2) = 17,460`

possible cross-implementation pairs. This deliberately allows, for example, a `Q-by-value` interaction under one
implementation to represent the same downstream variable as a different exact interaction under another.

For document window `w`, the observed finite causal signature of a node is

`O_w(u) = [four copy-task CE changes, 32 circuit member-minus-control CE changes]`.

The discovery relation is approximate proportionality, `O_w(u) approximately beta O_w(v)`, with one scale fitted
only on the first document half and bounded by `0.25 <= |beta| <= 4`. Both discovery halves must pass the frozen
task and circuit cosine/residual bars. The 30 other circuit coordinates and documents 752:1000 are held out. A
surviving relation is then tested by physically replacing each term with the scaled donor term at the same site in
both directions and recomputing the suffix. This last operation, not the response cosine, is the operational circuit
test.

The native factors have permutation and rescaling gauges; the two MLP input slots can also be exchanged, and each
normalized Q/K pair permits matched basis changes that preserve its dot product. Rung515 does not claim that its
31+3 coordinates are a canonical weight basis. It asks whether the downstream computation itself equates any of
their finite interventions. The experiment adds no deployed parameters and saves none. Its maximum registered price
is 108,748 full-model forwards, zero backwards, plus local exact-corner contractions; the B-false discovery route
costs 52,452 forwards.

## Closest exact mathematics: a finite observational quotient

The best mathematical match is finite-state partition refinement, not tensor rank. Paige and Tarjan give efficient
algorithms for a coarsest partition stable under a finite relation
([Paige and Tarjan 1987](https://doi.org/10.1137/0216062)). In the stochastic setting, Givan, Dean, and Greig define
state equivalence by equal immediate observations and equal transition probabilities into every quotient block;
iterative splitting yields the bisimulation partition
([Givan, Dean, and Greig 2003](https://cs.brown.edu/people/tdean/publications/archive/GivanetalAIJ-03.pdf)).

The exact mapping would be:

- finite states: our 816 intervention nodes;
- observable labels: the complete downstream task/circuit response under every allowed context;
- actions: every legal removal, substitution, and later composition;
- transitions: the resulting internal state after each such action; and
- quotient blocks: internal interventions with identical observations whose action successors remain in identical
  blocks.

With that complete finite table, the coarsest stable partition would be canonical for the chosen observations and
actions. This would directly implement the desired rule: merge pieces across native modules when every downstream
use treats them as the same variable, and split a native module when some use distinguishes its pieces.

The theorem does **not** apply globally to the current assay. We observe sampled documents, 62 circuit families and
a few task cells rather than every continuation; we test only a registered subset of node pairs; and substitutions
are evaluated only after a pair passes discovery rather than forming a closed transition table. Our thresholds also
define approximate rather than exact equality. The factored-MDP paper warns that symbolic stability tests can be
computationally hard even when the explicit finite partition procedure is simple. Therefore “Rung515 found the
minimal Theseus state machine” would be false. The defensible claim is narrower: it searches for a quotient relative
to the registered contexts, observations, actions, and tolerances.

Quantitative bisimulation metrics are a possible extension: Ferns, Panangaden, and Precup construct state distances
whose zero set is bisimulation and relate distance to value differences
([Ferns, Panangaden, and Precup 2004](https://www.cs.mcgill.ca/~prakash/Pubs/Ferns_MetricsForMDPs.pdf)). Their bound
assumes a finite MDP, Markov transitions, rewards, and discounted future value. Our deterministic neural suffix with
interventions is not that MDP, so the published value bound cannot be imported. What does transfer is the design
principle that approximate grouping needs an operational distance over both present observations and future actions;
cosine of write vectors alone is insufficient.

## Predictive states and why the recent rank-one task screen is not a realization theorem

Computational mechanics groups histories when they induce the same full conditional distribution over futures; the
resulting causal-state representation is minimal among equally predictive representations
([Shalizi and Crutchfield 2001](https://csc.ucdavis.edu/~cmg/compmech/pubs/cmppss.htm)). Here an analogous “future”
would be all downstream continuations and interventions following an internal term. Our registered task/circuit
signatures are finite tests of that condition, not the complete conditional future distribution. They can falsify an
equivalence, but passing them alone cannot prove global predictive-state identity.

Weighted-automaton realization theory gives a sharper warning about rank. For a function on strings, a finite-rank
prefix-by-suffix Hankel matrix corresponds to a finite linear realization
([Carlyle and Paz 1971](https://doi.org/10.1016/S0022-0000(71)80005-3)); spectral methods recover such realizations
from suitable Hankel blocks
([Balle et al. 2014](https://borjaballe.github.io/papers/preprint-bclq13.pdf)). The recent task-space matrix is only
four score implementations by document-and-task coordinates. It is not indexed by a prefix/suffix-closed experiment
set, and bilin18's normalized nonlinear transition is not a weighted finite automaton. Its singular rank is therefore
neither the number of model states nor the number of circuits.

There is also a concrete correction to that screen. The two document halves use different columns, so their right
singular vectors live in different document-coordinate systems and should not have been compared elementwise. The
comparable objects are the four-dimensional left singular vectors, which describe the implementation loadings. A
direct CPU recomputation gives cosine `0.9997287` between those loadings, while the leading uncentered energy is
`0.9642/0.9715`. This is a useful screen that the four implementations share a large task-effect component. It is not
an identified circuit: after subtracting the shared column mean, leading energy is only `0.5874/0.4644`, and no
Hankel, extraction, substitution, or selective-manipulation theorem follows.

## Tensor-decomposition alternatives and their limit here

CP, Tucker, tensor-train, and hierarchical-Tucker algorithms can simplify local coefficient tensors or their exact
contraction order. For the MLP tensor `T[o,i,j] = sum_r Down[o,r] Left[r,i] Right[r,j]`, however, the native rank is
4,608 while each factor mode is at most 1,152. Kruskal's standard sufficient uniqueness inequality cannot hold at
that rank. More importantly, a canonical or low-error tensor factorization would still not say which downstream
circuit uses a factor or whether two factors can be exchanged. These methods remain useful later for pricing a
causally identified program, but they do not dominate the current downstream-observation route.

## Executable consequence: a circuit witness certificate

Rung515 is the correct first empirical restriction of the finite quotient. Its result should be followed by a
certificate that converts failures into circuit information rather than merely reporting zero pairs:

1. For every registered pair, retain its two-half task and circuit response signatures and its fitted scale.
2. Build an incompatibility graph: connect two nodes when a registered observation proves they cannot be in the same
   approximate proportionality block.
3. For each incompatible pair, record the task cell or circuit coordinate contributing most to the failed residual.
4. On the 32 discovery circuits, greedily find a small set of observation coordinates that witnesses as many forced
   splits as possible; freeze that set and measure its split coverage on the 30 held-out circuits and fresh documents.
5. Within each fixed branch/site, compute a maximum clique or certified lower bound in the incompatibility graph.
   Its size lower-bounds the number of distinct operational variables required by this registered observation family.

This does not turn rank into interpretation. It answers which known circuits force the internal terms apart, whether
the same small circuit set repeatedly defines the boundary, and how many distinct variables are unavoidable under
the current causal tests. The discovery/held-out separation prevents choosing observers that merely memorize these
documents. If Rung515 finds a positive pair, bidirectional substitution remains the higher-priority test. If it finds
zero pairs, this witness certificate is the highest-information CPU extraction before leaving the exhausted
MLP10-consumer descent for a task-defined state-transition experiment or another documented program gap.

## Decision

Continue the managed Rung515 run unchanged. It tests cross-boundary grouping, held-out prediction, reuse, stable
identification, and selective substitution without optimizing rank. Partition-refinement theory explains exactly
what a complete solution would require and exposes the missing closure conditions. The immediate mathematical
addition is the observer-witness/lower-bound certificate above; it is cheap because Rung515 already computes the
finite response table. Hankel rank, local tensor rank, and the uncentered task-space singular value remain screens or
prices, not circuit discoveries.
