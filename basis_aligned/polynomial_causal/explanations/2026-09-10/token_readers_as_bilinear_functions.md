# Do different tokens read the same bilinear computation?

**10 September, 15:33 UTC.** The weight-based screen found no qualifying shared
pair among 518 sampled unembedding rows. The typical token's closest partner
leaves 97% error as a scaled substitute for its bilinear function. The old
cluster means also remain poor substitutes after folding.

This rejects a specific simple proposal: discovering reuse through nearly
proportional **whole token readout functions**. Different token functions could
still combine shared smaller computations. This test does not rule those out.

We checked the MLP17 dossier and previous reader experiments, derived an exact
comparison of input functions, ran the weight contractions through bqrunner,
then inspected the nearest pairs and cluster errors on CPU. The GPU computation
took .713 seconds with zero model forwards. Both instrument predictions passed;
the registered sharing prediction failed.

## The computation

The unembedding row U_t scores token t. Folding it into the last bilinear MLP
gives c_t=U_t D, coefficients over its 4,608 product features. Equivalently,

    Q_t = sym(L^T diag(c_t) R), where sym(A)=(A+A^T)/2.

Q_t is a symmetric 1,152-by-1,152 matrix describing which pairs of input
directions interact to affect token t. For raw MLP input x, this MLP's
contribution before final normalization is

    x^T Q_t x / (mean(x²)+epsilon) + U_t bias.

The residual skip, bias, final shared RMS normalization and tanh softcap remain
separate parts of the computation. Similar Q matrices would not establish
identical final logits or an independently extracted circuit.

We avoid constructing every Q by comparing the native input forms
H_i=sym(l_i r_i^T). Their Gram matrix records pairwise inner products:

    K_ij = [(l_i·l_j)(r_i·r_j)+(l_i·r_j)(r_i·l_j)]/2.
    <Q_t,Q_s> = c_t K c_s^T.

This exactly computes norms and angles after folding. It is unchanged by
exchanging bilinear halves or compensating their scales. The metric uses native
Euclidean input coordinates, not arbitrary non-orthogonal coordinate changes.

We also compare Q_t°=Q_t-trace(Q_t)I/1152. This removes the part responding only
to squared input length, so that a generic norm response alone cannot count as
shared token computation.

Each sampled token chooses the other token with greatest absolute angle
similarity of these trace-free forms. Signed scales are allowed because two
consumers could use one computation at different strengths or signs. The best
scaled-substitution error is sqrt(1-cosine²). A candidate needed error <=10% for
both full and trace-free forms, while raw unembedding rows differed by >50%
under their own best scale. The registered target was 16 qualifying readers;
there were zero.

These are errors in quadratic coefficients, not CE or fractions of behavior.
Even small coefficient error need not imply small relative response error on
inputs where the true response cancels.

## Results

| Comparison | Median substitution error |
|---|---:|
| Closest trace-free partner, full quadratic function | 97.10% |
| Closest trace-free partner, trace-free function | 97.22% |
| Old cluster mean, raw unembedding | 95.53% |
| Old cluster mean, full quadratic function | 95.83% |
| Old cluster mean, trace-free function | 95.92% |

Thus the coarse cluster failure is visible in this coefficient metric too.
This supports, but does not establish a causal explanation for, the earlier
94–98% live-effect prediction errors. Nearest partners share an old cluster
78.0% of the time, versus a 55.8% random-partner expectation accounting for
uneven sampled cluster sizes. Some neighborhood structure survives folding;
that is much weaker than interchangeable computations.

Only IDs 196/50274 passed the function-error condition before the raw-reader
control. Their trained unembedding rows are literally identical. ID 50274 is a
padded model output beyond GPT-2's 50,257 tokenizer entries, so this is not new
linguistic sharing. There is one padded ID in the frozen 518-row sample.
Excluding selected pairs involving it leaves zero close function pairs, without
choosing new partners or changing the original verdict.

The GPU Gram agreed with explicitly constructed full-width FP64 forms within
4.13e-8 relative error, and 3.29e-8 for trace-free forms. Tiny independent algebra
and gauge checks were below 2.54e-16. Peak allocated CUDA memory was about 407 MiB.
All native weights remain required; the temporary Gram holds 21,233,664 numbers.
No structural saving or causal circuit is claimed.

The next useful object is a **shared subterm with explicit inputs and consumers**.
Two tokens may use different sums of such operations even when their complete
functions are dissimilar. Those proposed subterms still need held-out causal
tests, extraction, selective removal and composition evidence.

Evidence: [preregistration](../../UNEMBEDDING_QUADRATIC_READER_GEOMETRY_V1_PREREGISTRATION.md),
[weight result](../../UNEMBEDDING_QUADRATIC_READER_GEOMETRY_V1_RESULT.json),
[CPU accounting](../../UNEMBEDDING_QUADRATIC_READER_GEOMETRY_AUDIT_V1_RESULT.json),
[duplicate audit](../../UNEMBEDDING_QUADRATIC_READER_DUPLICATE_AUDIT_V1_RESULT.json),
and [MLP17 dossier](../MLP17_CURRENT_UNDERSTANDING.md).
