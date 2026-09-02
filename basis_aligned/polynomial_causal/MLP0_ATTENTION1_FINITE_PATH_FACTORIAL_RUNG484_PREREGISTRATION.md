# Rung 484 preregistration — exact finite MLP0-to-attention1 path factorial

Written after rung483's strong null and before any rung484 model outcome.

## Why this experiment

Rung483 measured the complete attention1 and MLP1 output changes caused by the exact MLP0 token-only branch `T` and
token-by-context branch `I`. Their complete-removal responses were different, especially at attention1, but the local
derivative at the native state did not predict full removal. A smaller derivative step can diagnose rung483's `S`
check, but it cannot repair that scientific failure.

Rung484 therefore uses only complete finite states. It splits attention1 into the three exact multiplicative parts
that create its write: its first Q/K score, its second Q/K score, and the value/output part. It asks which of these
paths carries `T` and `I` to later behavior. It does not select a rank, native attention head, sparse support, or
compressed replacement.

## Exact MLP0 branches

Reuse rung401's identity

`MLP0 write = fixed remainder + T + C + I + S + bias`.

Only `T` and `I` are manipulated here because rung483's registered split concerned those branches. For branch `b`,
the native block0 state is compared with the exact state produced by subtracting all of `b` while retaining every
other branch and the numerical remainder.

## Exact attention1 factorization

For attention1 head `h`, query `q`, and causal source `k`, define

`A_h(q,k) = dot(rotary_norm(q1_h(q)), rotary_norm(k1_h(k))) / 128`,

`B_h(q,k) = dot(rotary_norm(q2_h(q)), rotary_norm(k2_h(k))) / 128`,

and

`V_h(k) = (1-lambda) c_v(state)_h(k) + lambda first_value_h(k)`.

The complete attention1 write is

`c_proj(sum over h,k<=q of A_h(q,k) B_h(q,k) V_h(k))`.

`A` and `B` are the two scalar attention-score sides. `V` is the information carried from each source, including the
fixed attention0 first-value contribution. This is an exact computation, not a learned decomposition. The scalar
score matrices are unchanged by reciprocal basis changes inside Q and K, and the carried values are compared only
after the native output projection.

For each MLP0 branch, compute `(A,B,V)` at the native attention1 state and at the branch-removed state. Rebuild all
eight hybrids obtained by choosing native or removed values independently for `A`, `B`, and `V`. The all-native and
all-removed corners must reproduce the corresponding native and branch-removed attention1 writes.

## Physical downstream intervention

Keep the chosen MLP0 branch physically absent from the residual stream. At attention1, replace its naturally
recomputed write by each of the eight exact hybrid writes, then let MLP1 and all later blocks recompute normally.

The all-removed attention corner is ordinary physical branch removal. The all-native corner restores attention1's
entire native write while leaving the direct residual contribution of the MLP0 branch absent. Their difference is the
complete finite attention1 route, separated from MLP1's direct access to the missing branch.

For each token position store only contracted CE sums, sums of squares, cross-products, and the seven nonempty
factorial effects. No logits, token IDs, or hidden states are written to the result.

## Factorial effects and physical subset selection

For an attention-component subset `S`, let `y_b(S)` be the per-token CE change when components in `S` use their native
values and the other components use branch-removed values, while branch `b` remains absent from the residual stream.

The seven nonempty Möbius effects are the three singleton restorations, the three pair interactions, and the residual
three-way interaction. For example,

`interaction(A,B) = y({A,B}) - y({A}) - y({B}) + y({})`.

These seven terms must sum exactly to `y({A,B,V})-y({})`. They describe this explicit native-versus-removed causal
coordinate; they are not claimed to be coordinate-free semantic features.

On equality-positive positions in documents0:250, select separately for `T` and `I` the smallest **nonempty proper**
physical subset of `{A,B,V}` whose per-token CE effect predicts the complete attention1-route effect. The full
`{A,B,V}` corner is excluded because it predicts itself trivially. Among eligible subsets, minimize component count,
then scale-adjusted relative error, then the integer subset bit mask. A subset is eligible only if cosine is at
least`.90` and best scalar-adjusted relative error is at most`.35`. If no proper subset qualifies, selection is null
and B fails. Apply the frozen subset without refitting on documents250:500.

## Data and task-conditioned views

Use the same hash-bound 1,000-document row authority as rungs477b,481,483. Discovery is documents0:500 split at250.
Documents500:1000 remain closed unless the registered discovery license holds, then split at750 for validation.

Measure per-token effects on:

- all scored positions;
- positions with at least one exact equality-successor edge; and
- a fixed position-shift control for that equality mask.

The CPU feasibility audit found `25,344 / 24,861 / 25,306 / 25,541` equality-positive positions in the four fixed
250-document quarters, so no view is support-limited. Sixteen fixed nonzero cyclic position shifts supply empirical
same-norm pairing controls. No 62-circuit average is used.

## Frozen predictions

### A — exact and lawful instrument

- All rung401, rung481, rung483, row, source, and model hashes match; rung483 remains a registered strong null with
  validation closed.
- Native replay is exact and the all-native/all-removed reconstructed attention1 writes have relative squared error at
  most`1e-8` in float32 and at most`1e-5` after deployed BF16 conversion.
- Every hybrid runs with exact dispatch and factor-call counts; all finite branch and attention-route effects are live.
- The seven Möbius effects reconstruct the complete attention-route CE vector to relative squared error at most`1e-8`.
- Validation documents, FINAL, and SEALED remain unopened before the discovery decision.

### B — a physical attention1 path predicts each complete attention-route effect

For both `T` and `I`, the physical component subset selected on documents0:250 must predict the all-components
attention-restoration effect on documents250:500 with cosine at least`.80` and best scalar-adjusted relative error at
most`.50`, on all positions and equality-positive positions. Its equality-positive cosine must exceed the 95th
percentile of the position-shift controls by at least`.15`.

### C — the path decomposition is stable

For each branch, the seven signed Möbius effects form a path profile after division by their total absolute size.
The same branch's profile cosine between the two discovery halves must be at least`.85`, and the absolute share of
each term may change by at most`.20`. The selected physical subset must be identical in both halves when each half is
analyzed descriptively.

### D-shared — `T` and `I` use the same attention1 path

In both discovery halves, their path-profile cosine is at least`.90`, their independently selected physical subsets
are identical, and one scalar fitted from `T` to `I` on half0 predicts the half1 complete attention-route vector with
relative error at most`.35` on equality-positive positions.

### D-split — `T` and `I` use different attention1 paths

In both discovery halves, their path-profile cosine is at most`.60` and their selected physical subsets differ. For
each branch's selected mask, its cosine with that branch's complete route must exceed the cosine produced by applying
the same mask to the other branch by at least`.20`, in both halves. This tests branch-specific routing rather than
merely obtaining two different labels from the subset search.

The shared and split outcomes are mutually exclusive. Intermediate values identify no relation.

### E — the selected paths are task-informative rather than generic damage

For both branches, the selected subset's signed effect must have the same sign on equality-positive positions in both
halves, its equality-positive mean absolute effect must be at least1.25 times the position-shift-control median, and
its equality-positive prediction cosine must exceed its all-position cosine by at least`.05`. This is intentionally a
task-specific requirement; failure means attention1 path anatomy exists but has not explained the equality behavior.

### F — held-out documents

Open documents500:1000 only if A/B/C/E hold and exactly one of D-shared/D-split holds. Require A/B/C/E again, the same
D outcome, and the same selected physical subsets. No component, subset, scale, or threshold may change after
validation opens.

## Null and routing

The scientific strong null fires when A, B, C, or E fails or neither D outcome holds. A failure does not invalidate
the exact MLP0 branch formula or rung483's descriptive finite split. It says the registered attention1 factorization
does not yet yield a stable, task-specific operational path.

- If A alone fails, repair only the factor replay or accounting instrument.
- If B fails, retain the full finite attention1 route and move to MLP1's direct bilinear factorization rather than
  tuning a subset or rank.
- If C fails, preserve per-example finite paths and test a context-conditioned mixture; do not average them harder.
- If E fails, use a behavior other than equality/copy or learn the reader from per-token downstream computation; do
  not relabel generic damage as a circuit.
- A validated D-split licenses a below-head comparison of the selected score/value paths across attention1 heads,
  using scalar attention patterns and output-projected carried vectors rather than head identity.
- A validated D-shared licenses exact cross-branch interchange of the shared path.

No outcome in this rung is compression or adoption. It identifies the next finite computational boundary to test.

## Price

Discovery runs one in-process native baseline and, for each of `T` and `I`, the eight attention-component corners with
the MLP0 branch absent. The conditional validation repeats the same fixed computation. Stored parameter savings and
added parameters are both zero.
