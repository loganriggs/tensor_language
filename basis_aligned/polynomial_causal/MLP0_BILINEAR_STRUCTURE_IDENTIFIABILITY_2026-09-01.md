# MLP0 bilinear structure: what the full embedding can and cannot identify

## Question

At position zero, all token inputs to MLP0 can be folded exactly through the embedding, normalization, remix, and
self-only attention. Given the entire finite input population and all MLP0 weights, when can a block, hierarchy,
or DAG be recovered from the bilinear layer itself?

## Function-level object

Ignoring bias, MLP0 is

`F(x) = sum_u d_u (l_u^T x)(r_u^T x)`.

For an output direction `q`, the scalar contraction is the symmetric quadratic form

`q^T F(x) = x^T A(q) x`,

where

`A(q) = sum_u (q^T d_u) sym(l_u r_u^T)`.

The exact folded vocabulary supplies a positive input metric `C = E_token[xx^T]`. The gauge-invariant functional
family is therefore

`B(q) = C^(1/2) A(q) C^(1/2)`.

This family is invariant to hidden-unit permutation, reciprocal Left/Right scaling, Left/Right swap, and any
alternative CP factorization that represents the same function. Raw neuron labels and raw support patterns are not.

## Blocks are an algebraic claim

A common block decomposition exists exactly when there is a nontrivial orthogonal projector `P` such that

`[P, B(q)] = 0`

for every output direction in the claimed family. Equivalently, the commutant of the contraction algebra contains
an idempotent other than zero and identity. With a generic reference contraction diagonalized, the remaining
commutator energy becomes a graph Laplacian; disconnected components are exact blocks and low stable modes are
approximate blocks.

This is what rungs340/346 tested. The instrument recovered planted blocks exactly and survived common gauge
changes. Real full1152D MLP0 had real/null Fiedler ratio `1.294`, split overlap `.172`, and greater optimized
off-block energy than its matched null. Therefore the *unconditioned* contraction algebra is irreducible under the
tested criterion. More embedding rows cannot repair this: all 50,304 inputs were already included exactly.

## A hierarchy is a nested reducing flag

A hierarchy is stronger than one partition. It requires a nested sequence

`0 < P_1 < P_2 < ... < I`

whose projectors reduce the relevant contraction families at successive levels. A valid recovery test should:

1. recover a planted nested flag after gauge scrambling;
2. select each split using one set of output contractions and reproduce it on disjoint contractions;
3. beat spectrum- and entry-matched coordinate nulls at every level; and
4. improve a literal executable price or predict held-out interventions.

Because the full unconditioned algebra is already irreducible, recursive clustering of its eigenvectors cannot
create an identified hierarchy. A hierarchy can remain only after the contraction family is restricted by an
external task variable.

## Why a DAG is not oriented by one bilinear layer

The scalar product `(l^T x)(r^T x)` is commutative. Swapping `l` and `r` leaves the function unchanged, and the
quadratic matrix contains only `sym(l r^T)`. Moreover, the 4,608 hidden products are evaluated in parallel; there
is no native hidden-unit-to-hidden-unit transition inside this single layer. Consequently, an arrow between two
hidden features is not a function-level observable of MLP0 alone.

A directed claim needs an asymmetric external operation, for example:

- an intervention followed by a measured downstream response;
- temporal or layer order in a multi-layer composition;
- a behavior-conditioned innovation rule with parent variables fixed before child variables; or
- an explicitly asymmetric input/output pairing rather than a symmetric quadratic contraction.

Without one of these, a recovered DAG orientation is a prior imposed by the fitting method, not structure
identified in the weights.

## Legitimate reopening: behavior-conditioned contraction algebras

Let `z` be a named variable defined independently of the candidate decomposition—for example a registered
retrieval state, intervention condition, or syntactic state with a legal executable interface. For each state,
form its held-out input metric `C_z` and a preregistered set of output/response directions `Q_z`:

`B_z(q) = C_z^(1/2) A(q) C_z^(1/2),  q in Q_z`.

Then ask whether the state-conditioned families share a stable reducing flag, or whether a small finite state
chooses among a fixed bank of such flags. This is different from the failed token/context kmeans routers: the state
is named by behavior, not discovered from the same geometry whose partition is being judged.

Advancement requires all of the following:

- planted recovery and a live shuffled-state null;
- state definitions and output directions frozen without evaluation labels;
- split-stable projectors/flags and held-out contextual transfer;
- intervention evidence for any orientation claim;
- a literal bill for state detection, projectors, experts, exceptions, and compute; and
- superiority to an equal-price single shared subspace.

Until such a state is named, the correct result is not “MLP0 has no structure.” It is narrower: its useful
low-rank input action is one shared spectral object, while generic token partitions, live-input kmeans partitions,
and task-free block/tree/DAG decompositions are unsupported.
