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
negative, and rejection of writes that change the final query. The trained-model
audit remains unexecuted. The derivation is invalid for a changed query or an
additional contextual layer after the edited states without further work.

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
