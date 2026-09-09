# Can the document summary be reused after records move?

The first-layer query state has the exact source partition
`S(document, hop) + L(query entity, hop)`. The fresh extraction reproduces full
outputs, separate removals and their interaction to at most 2.99e-13. Its four
128-coordinate document summaries are reused across 24 query entities. This is
ordinary shared evaluation, with all 387,968 native constants retained. It does
not establish the structural simplification requested in the bilinear handoff.

We next ask a stronger reuse question: can those summaries update themselves
when the 24 binding records are reordered, without consulting the document?
Two restricted answers are now available. Neither is a theorem that the entire
model lacks a simpler explanation.

## Explicit object and domain

Let Xk and Xv be 24-by-24 permutation matrices: rows are record positions and
columns are key or value identities. Choose C with 23 orthonormal zero-sum
columns. Every matrix in the affine hull of permutation matrices is uniquely
`11^T/24 + C Z C^T`. Row and column constraints leave 23² coordinates; the
2-by-2 exchange matrices span the corresponding zero-sum linear space.
Independent key/value permutations therefore have affine dimension 1058.

Stack the four native hop summaries into s in R^512. First-layer token-only
projections and additive source contributions give `s = o + A x`, where
`x = (vec(C^T Xk C), vec(C^T Xv C))` and A has shape 512-by-1058.
The implementation computes A directly from the original RMS, RoPE, Q/K/V/O
weights. This identity uses the affine hull of **all bijections**, broader than
the IID 24-cycle and OOD three-8-cycle families used in behavioral tests.

A record permutation pi induces an orthogonal position action Rpi on the
23-dimensional contrast space and an action Ppi on x. The new summary is
`o + B x`, B = A Ppi. Affine summary transport `s' = T s + b` on this domain
requires B = T A, with b = o - T o.

## Norm-bounded transport obstruction

For any right operator Q and spectral norm bound ||T||₂ <= K,

    ||B - T A||F >= (||B Q||F - K ||A Q||F) / ||Q||₂.

This follows from submultiplicativity and the reverse triangle inequality.
We use Q = I - V^T V, with V the first 384 computed right singular vectors,
K = 10^6, and the conservative upper bound
`||Q||₂ <= 1 + ||V V^T||infinity + 1e-10`.
The inequality does not require treating any small singular value as zero.
The fixed 384-row choice is motivated by four hops times four heads times
24 possible token value vectors; it is not selected for a desired verdict.

[The receipt](../SUMMARY_RECORD_TRANSPORT_BOUND_V1.json) gives relative
Frobenius lower bounds .110689 for swapping the first two records and .117490
for shifting all records by one. Direct token correspondence is 3.11e-15 on
16 opened worlds and both reorderings. Six controls include a near-null map
that admits an exact transport only above the norm cap.

These are FP64 evaluations of an analytic inequality, not interval certificates.
The cap depends on the chosen summary coordinates. There is no conclusion about
unbounded transports, nonlinear decoding, arbitrary gauges, cohort KL, or the
full model. A finite set can in principle have an injective low-dimensional
encoding even when affine-domain transport fails.

## Minimal linear closure under every record permutation

There is also a direct representation argument. Write the coefficient rows of
A in Vposition tensor W, with dimensions 23 and 46; W combines parity and
entity contrast. Let U be the span in W of every position slice of every row.
Then the smallest permutation-invariant row space containing A is Vposition
tensor U.

Proof: for each i < 23, the transposition exchanging record i with record 23
acts on contrast coordinates as I - r_i r_i^T, where
`r_i = C^T(e_i - e_23)`. These 23 roots form a basis, have squared norm 2,
and have pairwise inner product 1. Products of I minus these transpositions
therefore span every rank-one map r_i r_j^T, hence all End(Vposition).
Applying arbitrary position matrix units to each observable isolates its
slices and places them at every position. This proves both inclusions and
minimality. The invariant subspace is unique; a coordinate basis is not.

The required slice matrix is only 11776-by-46. The fixed full-column test in
[SUMMARY_PERMUTATION_CLOSURE_V1.json](../SUMMARY_PERMUTATION_CLOSURE_V1.json)
finds singular extrema 3.012879 and .048296. More decisively, a selected
46-by-46 minor has determinant 1808312459 modulo prime 2147483647, after exact
conversion of each stored dyadic coefficient into the finite field. All
denominators are powers of two and invertible modulo this prime. A nonzero
residue proves the rational determinant is nonzero. The receipt includes
every minor coefficient in exact hexadecimal float notation and its indices.
Three determinant controls pass; the audit takes .059 seconds on two CPU
threads. It certifies the stored FP64 operator, not unrounded real arithmetic.

Thus this operator's linear permutation closure has dimension 23*46 = 1058:
the full centered incidence space. Extending this summary to support arbitrary
linear record transport cannot discard any of those incidence coordinates.
This is a decision about an intervention interface, not a rank-compression
proposal. Materializing A adds 541,696 numerical values; no native coefficients
are removed, and we do not adopt the expanded representation.

The next relevant distinction is whether record dependence reaches the native
output or can disappear downstream. A function-only abstraction would predict
identical outputs after answer-preserving record permutations. That requires
a full-output test; the summary obstruction alone does not establish it.
