# How the final reader uses a join write

The contextual matcher's paired gate cannot be freely transferred between
matched middle-label cases. Its signed values change substantially, and the
native reader is sensitive to those changes. Rather than fit a sign correction,
we now ask which algebraic operations the final reader performs on the write.

The scope is the existing four-layer attention model, with a selected L2 write
to binding positions and one final attention layer remaining. The final query
is not changed by this intervention. A curve for this fixed query is an exact
partial response program; it is not an independently reduced whole model.

## Exact token-derived response

Let d_s be the selected join write at source position s, and let z_s be that
source's post-L2 state with the write removed. Scaling the native write gives
`x_s(alpha) = z_s + alpha d_s`. All other source states and the final query
stay fixed. For each final attention head, define linear functions

    a(x) = q1 · K1(x) /32,
    b(x) = q2 · K2(x) /32,
    v(x) = V(x),

with actual absolute-position RoPE inside K1/K2 and the fixed normalized native
query inside q1/q2. These key/value projections act on the unnormalized source;
its three RMS factors will be supplied together. The folded output map and
final .5 interpolation factor are included when decoding the following vectors.

Write a(x)=a0+alpha a1, b(x)=b0+alpha b1, v(x)=v0+alpha v1. Multiplication gives

    C0 = a0 b0 v0
    C1 = a1 b0 v0 + a0 b1 v0 + a0 b0 v1
    C2 = a1 b1 v0 + a1 b0 v1 + a0 b1 v1
    C3 = a1 b1 v1.

After summing heads and decoding, each Ck is a29-dimensional source contribution.
The exact final-query logits are a fixed background plus

    sum_s g_s(alpha)^3 * (C0_s + alpha C1_s + alpha² C2_s + alpha³ C3_s),

where g_s(alpha) is the original RMS gain at that source. There is no degree
truncation in this identity. Separate source normalizers prevent merging their
coefficient vectors before normalization.

For stable scalar evaluation, let t=mean(z*d)/mean(d²), with t=0 when d=0.
Then the squared RMS denominator is

    mean(d²)*(alpha+t)² + mean((z-t*d)²) + epsilon.

This completed-square form avoids subtracting large terms in an expanded
quadratic. The identity is over real arithmetic; the implementation must still
pass numerical correspondence. Epsilon follows the actual affine-free RMSNorm.

The query-removal effect at alpha1 versus alpha0 splits exactly into
`(g1³-g0³)C0`, `g1³ C1`, `g1³ C2`, and `g1³ C3`, summed over sources. The first
term accounts for normalization changing the unmodified-source contribution.
These are degree attributions, not separately editable physical circuit states
or independent variance fractions.

## The operation hypotheses and their tests

Earlier native port restoration found predominantly value use for the forward
write and joint key-pair use for the backward write. That motivates two fixed
tests: retain C0, the actual normalizer and C1 forward; retain C0, the normalizer
and C2 backward. It does not establish either truncation in advance.

[JOIN_WRITE_READ_DEGREE_V1](../JOIN_WRITE_READ_DEGREE_V1_PREREGISTRATION.md)
uses all opened32 worlds, six orders and four query hops. The full curve must
match physical source edits and native score scaling at alpha=-1,0,.5,1,2.
Each fixed truncation must predict the complete centered query effect relative
to alpha0 within1%, separately for every population/orientation/hop/scale group.
No degree or scale range may be selected after seeing the result.

The [primitive](../join_write_read_degree_reference.py) passes12 CPU controls:
five physical source-edit scales, exact removal partition, live curve, nonnegative
norm representation, planted linear/quadratic cases, a live omitted-degree
negative, and rejection of writes that change the final query. The managed
trained-model audit is now complete: exact response and partition pass, both
fixed degree hypotheses fail. The derivation is invalid for a changed query or
an additional contextual layer after the edited states without further work.

## What is and is not saved

Compilation explicitly runs the token-derived native prefix and readers. It
retains all387968 export constants. For two changed sources and29 outputs, a
curve stores232 coefficient values, six norm values and29 background logits:
267 FP64 values,2136 bytes, plus source indices and the RMS epsilon. The weights
and compilation cost remain charged. Evaluating several scales can reuse these
values without rerunning the prefix or projecting changed source states.

The background is computed by the explicit native executor, not an unexplained
external activation cache. Nevertheless, this is a query/context-specific
response representation. No whole-model storage reduction, novel semantic
operation or OOD causal sufficiency follows from compiling it exactly. The
degree hypotheses, if supported, would identify a simpler operation in the
specified reader; their broader extraction and structural cost remain separate.

## Trained result and constraint

The [managed receipt](../JOIN_WRITE_READ_DEGREE_V1_RESULT.json) covers768 requests
from32 opened worlds in4.384 seconds. Five-scale native/source-edit correspondence
is1.28e-13 absolute and1.34e-15 relative RMS; removal partition closes1.33e-14.
At the native scale, hop3 forward-C1 relative errors are .06999 IID/.06772 OOD;
backward-C2 errors are .65392/.53191. Both registered hypotheses are rejected.
No degree, scale range, head or task subset is selected to repair the null.

For backward hop3 removal, the linear term's projection onto the full removal
is .498 IID/.422 OOD; the quadratic term's is .488/.575. Both contribute
substantially. These correlated terms are neither variance fractions nor
independently editable causal variables. Forward removal has a large linear
term but the omitted terms still violate the1% prediction bar. Port necessity
and polynomial degree therefore describe different properties of the reader.

Curve compilation excluding native context takes .252 seconds; saved cohort
curve tensors total1,643,520 bytes. All387968 opaque export coefficients remain.
The exact response compiler is useful intervention machinery, not the requested
structural discovery.

[Metadata correction](../QUERY_HOP_METADATA_CORRECTION_V1.json) supplies576
answer/expected-answer fixes for the degree rows: their hop-fork loop retained
hop3 labels. These labels were not consumed by the registered exactness, KL,
centered-effect or degree/partition scores. Saved logits, curves and results
remain unchanged. Apply the overlay before future answer-based analysis.

The next completed screen groups the existing direct forward endpoint writer
with all final readers; see [matcher dossier](join_matcher_factors.md). It finds
sign compensation at hop3 but fails both grouped-vector portability and agreement
with physical removal. Neither exact response compilation nor conditional
path attribution has yet supplied a structurally simpler sufficient circuit.


## Two joins writing into the same binding

For a->b->c->d with b->c serialized after its two neighbors, the existing native
L2H1 forward join and L2H2 backward join write into the same two positions.
Queries a and F^-1(a) use different two-edge tails there. This tests composition
of context-dependent computations, without assuming invariant gate strengths.

The [overlap screen](../OVERLAPPING_JOIN_NORMALIZER_V1_RESULT.json) uses512
opened requests from32 worlds, two overlapping orders, two origins and four
hops. Native execution and the complete single-write response curves agree to
1.42e-13/1.03e-13; source writes are identical across the eight query forks.
The CPU run takes5.224 seconds. Native hop3 accuracy is .969–1 and each
intended join removal has a large effect, but the writes are not independently
selective: backward removal changes the other, forward query's gold probability
by mean absolute .17580 IID/.35091 OOD. Forward removal's OOD cross-change .05944
also exceeds the .05 bar. No subset of eligible groups is adopted.

A fixed composition hypothesis retained both complete univariate cubic
numerators and the exact joint RMS denominator, omitting only mixed numerator
terms around native state. For source state x and writes u,v it uses

    g(x+(a-1)u+(b-1)v)^3 * [P(x+(a-1)u)+P(x+(b-1)v)-P(x)].

The axes a=1 or b=1 are exact. The joint hypothesis fails every registered
panel: at both-cut, interaction-relative errors are4.784/4.693 forward and
2.322/2.178 backward IID/OOD. Shared normalization alone is insufficient;
mixed key/value numerator products are required. This is a different overlap
from the earlier raw/endpoint field test, which also found that freezing RMS
did not restore independent field semantics. Neither null is rescued here.

## The backward writer affects both readers through keys and values

The [native port factorial](../OVERLAPPING_JOIN_PORTS_V1_RESULT.json) then uses
all eight K1/K2/V subsets, with K1+K2 fixed as the address hypothesis. Native
projection-hook correspondence1.28e-13, saved full-cut replay1.42e-13 and query
reuse0 validate the instrument on the same512 requests; CPU2.622 seconds.
All opaque export coefficients and routing are still retained.

For the backward query, key-pair effects recover1.054 IID/1.043 OOD of the full
cut's signed gold-probability effect; V-only is .045/.035 in absolute ratio.
For the forward query, the key-pair cut overshoots the full cut: ratios1.351/
1.176, and the IID value effect is .169 of the full effect. The registered
shared-address nomination therefore fails; no backward-query-only success
replaces it. Value-only cuts improve gold probability on both origins, so
keys and values have partially opposing effects under these interventions.
Probability effects are nonlinear; their factorial interaction is retained.

Key-pair-only full-vector errors are .187/.187 on the forward query and
.092/.111 on the backward query. The sufficient-port hypothesis fails too.
The next [key gain protocol](../OVERLAPPING_JOIN_KEY_GAIN_V1_PREREGISTRATION.md)
separates direction K(x-v) from RMS gain g(x-v), before interpreting a changed
key as address information. It is a confound diagnostic, not a replacement for
the failed composition or full-source sufficiency tests. No structural
simplification has yet been established by these response decompositions.
