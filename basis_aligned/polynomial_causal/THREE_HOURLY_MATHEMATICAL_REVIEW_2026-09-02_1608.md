# Three-hour mathematical review — 2026-09-02 16:08 UTC

## Goal and present boundary

The target is a smaller executable tensor program whose units say what information is read, what operation is
performed, what is written, and which later computations use the write. The units must predict held-out and shifted
inputs, compose when installed together, and support selective removal, swapping, or editing. Literal storage and
compute price matter after a circuit unit is identified; rank, reconstruction, quantization, or CE alone do not
identify one.

Rung494 closes the scalar equality-composition route at its current grain. A fitted one-dimensional nonlinear
readout was worse than ordinary addition for every half-strength condition. It helped for all pooled 1.5-times
conditions, but only 62.8–76.0% of those inputs were inside the fitted range and one fixed document half failed.
The next uncertainty is therefore whether attention1 contains computations shared across native heads, or distinct
computations within one head, when equivalence is defined by downstream use rather than weight or activation
similarity.

## Exact attention1 object

For at most `n=256` positions, the residual width is `d=1152`. Attention1 has `H=9` heads of width `r=128`. For head
`h`, query position `i`, and causal source position `j <= i`, write the two normalized, rotary-encoded query/key
scores as

`A[h,i,j] = <q1[h,i], k1[h,j]> / 128`

and

`B[h,i,j] = <q2[h,i], k2[h,j]> / 128`.

The value vector is `v[h,j] in R^128`. It mixes attention1's own value projection with the retained earlier value
according to the model's learned scalar. If `O_h : R^128 -> R^1152` is head `h`'s slice of the output matrix, the
exact write is

`a[i] = sum_h sum_(j<=i) A[h,i,j] B[h,i,j] O_h v[h,j]`.

Thus the contraction indices are head `h`, query position `i`, source position `j`, and head coordinate `u`. If the
already-normalized `q1,k1,q2,k2,v` are regarded as independent inputs, each source-edge term is set-multilinear of
degree five. As a function of the residual stream it is not a polynomial: each Q/K vector is divided by its
input-dependent RMS norm. The actual map is smooth and rational away from zero norms.

The projections are tied across positions. Q/K coordinates have simultaneous basis-change gauges that preserve
their inner products; value coordinates have a matching value/output gauge; native head labels can also be
permuted with their parameter blocks. Internal MLP product coordinates have their own permutation and reciprocal
scaling gauges. Consequently, a raw sparse or low-rank factor is not a semantic unit unless an observable reader or
interchange test fixes its meaning.

The allowed inputs are real states produced by normal text and by registered MLP0 `T/C/I/S` branch removals. The
outputs to preserve initially are attention1's 1,152-dimensional write, its exact contribution through MLP1, and a
62-number vector of downstream circuit effects. Discovery uses documents0:500 split into two fixed halves;
documents500:1000 remain unopened unless a unique candidate is selected. The numerical norm is relative squared
write error for exact closure and cosine plus best-scale residual for response signatures. The causal price is model
forwards/backwards and physical interventions; a later executable replacement must also state stored values,
operations, and intervention edges.

## An exact below-head decomposition

For one branch removal `b`, capture the real factors `(A_N,B_N,V_N)` and the factors `(A_b,B_b,V_b)` produced when
`b` is absent. For every head, run the eight combinations that choose the normal or absent version of each of the
three factors. Möbius subtraction of those eight writes produces seven nonempty finite terms:

`A, B, V, A×B, A×V, B×V, A×B×V`.

Here `A×B` means the part of the finite change that remains after subtracting the individual A and B changes; it is
not an outer product. Across 9 heads this gives 63 exact write pieces `theta[h,S]`, with

`sum_(h,S nonempty) theta[h,S] = attention1(normal) - attention1(b absent)`.

The head label only records where an exact term came from. Candidate circuit units may group terms from different
heads or split the seven terms within one head.

MLP1 is a quadratic map `m(z)=Down((Left z) elementwise-multiplied-by (Right z))+bias`. Let `P(u,v)` be its symmetric
polarization:

`P(u,v)=Down((Left u)*(Right v) + (Left v)*(Right u))`.

Freeze the direct residual part to its branch-absent value `d_b`. If `a_N` and `a_b` are the normal and absent
attention1 writes, then the attention-only change through MLP1 is exactly

`m(d_b+a_N)-m(d_b+a_b) = P(a_N-a_b, d_b+(a_N+a_b)/2)`.

Because `P` is linear in its first argument, each of the 63 attention pieces has an exact MLP1 response

`rho[h,S] = P(theta[h,S], d_b+(a_N+a_b)/2)`,

and the 63 responses sum exactly to the complete attention-only MLP1 change. This avoids the pairwise explosion from
expanding MLP1 independently in all head pairs while still including every self-head and cross-head interaction
through the shared midpoint.

## What existing mathematics does and does not solve

Kruskal's CP uniqueness theorem identifies rank-one terms of a three-way tensor, up to permutation and scaling, when
the three factor matrices satisfy a k-rank inequality. The exact object here has five tied modes, head-sized blocks,
Q/K normalization, and likely collinear/shared factors; the required condition is not established and a CP term need
not have a unique downstream role. The theorem is a useful diagnostic on a restricted projected three-way tensor,
not a solution to the model. See J. B. Kruskal,
[“Three-way arrays: rank and uniqueness of trilinear decompositions,” 1977](https://doi.org/10.1016/0024-3795(77)90069-6).

Block-term decomposition allows a tensor to be a sum of multilinear-rank blocks and gives essential-uniqueness
conditions. One attention head is more naturally a block than a rank-one term, so this is closer than CP. But the
conditions require suitable independence among block subspaces, while the scientific hypothesis is precisely that
different heads may share one score half or downstream variable. It also identifies an input tensor factorization,
not equivalence under the model's later readers. See L. De Lathauwer,
[“Decompositions of a Higher-Order Tensor in Block Terms—Part II,” 2008](https://doi.org/10.1137/070690729).

The fundamental theorem and canonical forms for matrix-product states say when two one-dimensional tensor-network
descriptions represent the same state and characterize the remaining virtual-index gauge. Bilin18 is a finite
feed-forward network with content-dependent all-prefix attention edges, residual additions, and RMS normalization;
it is not a translation-invariant MPS family evaluated on every chain length. These theorems clarify that gauge
equivalence is weaker than semantic equivalence, but they do not supply the wanted decomposition. See Cirac,
Perez-Garcia, Schuch, and Verstraete,
[“Matrix Product States and Projected Entangled Pair States,” 2021](https://arxiv.org/abs/2011.12127), and Acuaviva
et al., [“The minimal canonical form of a tensor network,” 2022](https://arxiv.org/abs/2209.14358).

The Hankel-rank theorem for weighted finite automata gives a minimal linear realization, unique up to similarity, for
a finite-rank string function. Our network has continuous hidden states and nonlinear attention/MLP updates, so the
theorem does not apply globally. It does motivate a restricted operational quotient: two pieces are equivalent if
all chosen downstream observations give the same response. The resulting response row space is canonical even
though a basis inside it is not. This is the correct role for the 62 existing circuit measurements, not a claim that
they form a complete behavioral basis.

Polynomial identity testing could certify that two candidate attention formulas are algebraically equal after
treating normalized Q/K vectors as independent formal variables. Raz and Shpilka give polynomial-time white-box
tests for pure set-multilinear circuits. RMS normalization and reused residual variables violate that pure formal
object, and identity testing discovers no candidate decomposition; it can nevertheless become a later exactness
check for a proposed formula. See Raz and Shpilka,
[“Deterministic Polynomial Identity Testing in Non-Commutative Models,” 2005](https://doi.org/10.1007/s00037-005-0188-8).

Finally, unrestricted exact tensor-rank minimization is NP-complete. That rules out expecting a general efficient
algorithm for the globally smallest raw tensor decomposition and reinforces why another rank sweep is not the next
move. See J. Håstad,
[“Tensor Rank is NP-Complete,” 1990](https://doi.org/10.1016/0196-6774(90)90014-6).

## Executable consequence: downstream-use quotient before factor fitting

Rung495 should compute the 63 exact `QK1 × QK2 × OV` pieces for each of the four MLP0 branches and map each piece
through the exact MLP1 polarization above. It should then use the existing 62 circuit masks as downstream probes.
For circuit `c`, form the difference between its member-token CE average and its matched-control CE average and
differentiate that scalar with respect to MLP1's write. The response coordinate for piece `theta` is the inner
product of this gradient with `rho(theta)`.

This produces a 62-number downstream-use signature for every below-head piece. It is a first-order screen, not a
physical circuit result. A candidate cross-head merge must be a mutual nearest neighbour, have high response cosine
and low best-scale error in both document halves, remain after subtracting shared token difficulty, and be more
similar than position-permuted and circuit-label-permuted controls. Raw write similarity is recorded but not
required. Conversely, one head is split if two of its finite terms have stable, materially different response
signatures.

The opposing predictions are concrete:

- shared computation: pieces from different heads have the same held-out downstream-use signature even when their
  raw writes differ, and a later physical interchange preserves the target circuits;
- native heads are already distinct: cross-head signature matches do not survive the half split or controls;
- the 62 probes are insufficient: many pieces collapse under them but their physical held-out effects differ.

Only the first outcome licenses a held-out physical swap. The screen cannot itself establish extraction or selective
manipulation. If no candidate survives, the next different object is to split the selected score side from `A` or
`B` into query versus key factors and test shared halves directly; it is not to tune a rank.

## Decision

No reviewed theorem exactly recovers the desired circuits under the real normalization, parameter ties, and
downstream definition of equivalence. The strongest exact consequence is nevertheless useful: finite Möbius
decomposition supplies 63 closure-checked pieces below the head boundary, exact quadratic polarization carries them
through MLP1 without approximation, and the 62 circuit readers define the operational quotient the user proposed.
This dominates another SAE/Tucker or rank experiment because it can merge across heads, split within a head, predict
which later circuits use each piece, and nominate a selective physical interchange. Proceed with rung495 on that
object.
