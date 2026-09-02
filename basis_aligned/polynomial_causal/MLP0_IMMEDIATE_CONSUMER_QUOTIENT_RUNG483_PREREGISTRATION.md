# Rung 483 preregistration — MLP0 branches grouped or split by their immediate consumers

Written after rung481's valid strong null and before any rung483 model outcome.

## Question

Rung481 showed that the exact MLP0 token-only branch `T` and token-by-context branch `I` have somewhat similar
downstream effects, but averages over the 62 existing circuit labels were too noisy and nonselective to decide
whether the two branches supply one variable or different variables. Rung483 changes the measurement rather than
tuning a rank, sparse support, or threshold.

It asks how the actual next attention and MLP computations read each exact MLP0 branch:

1. Does attention1 treat `T` and `I` as the same direction?
2. Does MLP1 treat them as the same direction when attention1's write is held fixed?
3. Does the complete path, with attention1 recomputed, treat them as the same direction?
4. Do complete two-branch removals show that any pair must be analyzed jointly?

## Exact branch intervention

Use rung401's exact decomposition

`MLP0 write = fixed remainder + T + C + I + S + bias`.

For branch `b` and scalar `alpha`, replace the native MLP0 write by

`native MLP0 write - alpha * b`.

The four branches retain their rung401 meanings: token-only (`T`), context-only (`C`), token-by-context (`I`), and
the example-specific normalization-scale correction (`S`). The fixed remainder and bias are never altered.

## Three consumer outputs

For every document and every token position, capture three 1,152-dimensional outputs:

- `attention1`: attention block1's recomputed write;
- `MLP1 direct`: MLP1's write after restoring attention1's native write, so only MLP1's direct response to the
  changed incoming residual stream remains; and
- `MLP1 total`: MLP1's write with attention1 recomputed normally.

This separates the attention route from MLP1's direct read and their complete composition. It does not assume that
an attention head or a whole MLP is the semantic basis.

## Tangent and complete-removal computations

For consumer output `F_c` and branch `b`, compute the exact forward-mode automatic derivative

`r_c,b = d/dalpha F_c(native - alpha*b) at alpha=0`.

This is the consumer's local response to that complete MLP0 branch direction. Check it against the symmetric finite
difference

`[F_c(native + epsilon*b) - F_c(native - epsilon*b)] / (2 epsilon)`

with `epsilon=1/8`. The signs are equivalent to the `native-alpha*b` convention after applying the same convention
to both calculations.

Also compute the physical removal response

`q_c,b = F_c(native-b) - F_c(native)`

and every pair response `q_c,bd`. The nonlinear pair term is

`q_c,bd - q_c,b - q_c,d`.

Thus a first-order result cannot be called a circuit unless it predicts the corresponding complete intervention.

## Response Gram matrices

Within each document half and consumer, contract over every document, position, and all1,152 output coordinates:

`G_c[b,d] = sum r_c,b * r_c,d`.

Compute the same matrix from `q`. Its normalized entries are branch-response cosines. These matrices use complete
consumer outputs and are unchanged by an orthogonal rotation of output coordinates. No response rank is selected.

For a proposed relation `I = alpha*T`, fit

`alpha = G_half0[T,I] / G_half0[T,T]`

on documents0:250 and evaluate

`sqrt(||I-alpha*T||^2 / ||I||^2)`

from the independently accumulated Gram matrix on documents250:500. Fit and report separate scalars for each
consumer and for tangent versus complete-removal responses.

As a matched control, cyclically shift `I`'s response by16 fixed nonzero offsets along the256 token positions before
contracting it with `T`. This preserves every response vector and its norm but destroys the same-position pairing.

## Data and validation

Discovery uses the already frozen 1,000-document rung477b/rung481 row authority, documents0:500, split at250. All
positions enter the response Gram; the 62 circuit labels do not. Documents500:1000 remain unopened unless the
discovery result has a valid instrument, the tangent predicts the complete intervention, and exactly one of the three
relations below holds. Conditional validation repeats the unchanged calculation with a split at750.

No FINAL or SEALED outcomes are opened. No CE result is optimized or used to choose a relation.

## Frozen predictions

### A — the derivative and prefix instrument are valid

- All rung401, rung481, row, source, and model hashes match, and rung481 still has its valid scientific strong null.
- A separately dispatched native full-model forward and the shortened block0-to-MLP1 computation agree at
  attention1 and MLP1 to relative squared error at most`1e-12`.
- The exact MLP0 branch reconstruction is at most`1e-8` analytically and`1e-5` at deployed BF16.
- For every branch, consumer, and discovery half, automatic tangent versus symmetric-finite-difference cosine is at
  least`.98` and its best scalar-adjusted relative error is at most`.20`.
- Every registered branch and pair executes with exact call counts and produces a nonzero deployed change.
- Validation documents, FINAL, and SEALED remain unopened before the discovery decision.

### B — the local reader predicts complete branch removal

For both `T` and `I`, every consumer and discovery half must have tangent-versus-complete-removal cosine at least
`.75` and best scalar-adjusted relative error at most`.60`. This does not require unit slope: it requires the local
reader to predict which complete output changes occur.

### C-shared — one downstream variable

For all three consumers and both halves:

- `cos(r_T,r_I) >= .90` and `cos(q_T,q_I) >= .80`;
- the half0-fitted scalar predicts half1 with relative error at most`.35` for both tangent and full removal; and
- the actual tangent and removal cosines exceed their respective position-shuffle95th percentiles by at least`.15`.

### C-split — two downstream variables

For all three consumers and both halves,

`abs(cos(r_T,r_I)) <= .65` and `abs(cos(q_T,q_I)) <= .65`.

This means every measured immediate consumer distinguishes their output directions.

### C-consumer-specific — grouped for one reader and split for another

In both halves, at least one consumer has tangent cosine at least`.85` and full-removal cosine at least`.75`, while a
different consumer has absolute tangent and full-removal cosines at most`.55`. The identities of the shared and split
consumers must be the same in both halves.

The three C outcomes are mutually exclusive by construction. Intermediate values identify no relation.

### D — immediate-consumer interactions determine joint follow-up

Report every pair's nonlinear full-removal term at every consumer. A pair is stable and material when, in both
halves, its norm is at least`.20` times the smaller singleton norm, the two half ratios differ by no more than2x, and
the cosine between its mean 1,152-dimensional output vectors in the two halves is at least`.50`. Predict at least one
stable material pair involving `I`. A passing pair is analyzed jointly later; a failed pair cannot be promoted by a
rank or sparsity fit.

### E — the relation validates on new documents

Open documents500:1000 only if A and B hold and exactly one C outcome holds. Require A and B again, require the same C
outcome with unchanged thresholds, and require every discovery-selected D pair to remain material with the same
mean-vector sign relative to its two singleton responses. No relation may change after validation opens.

## Null and routing

If A fails, preserve the receipt and repair only the derivative or prefix instrument. If B fails, immediate
Jacobians are not an adequate description of complete branch removal.

The scientific strong null fires when A or B fails or none of the three C outcomes holds. It does not invalidate the
exact `T/C/I/S` formulas. It says that these three immediate output maps do not supply a stable operational grouping.
Do not respond with rank reduction, quantization, PCA, or threshold tuning. Change the observed object to
task-conditioned reader functionals or exact finite interchange on a registered behavior.

If one C outcome validates, use its consumer relation to define the next branch decomposition. Any D-positive pair
must be included jointly. A validated relation is still identification, not a compressed replacement or semantic
name; selective behavioral manipulation is required next.

## Price and stored data

The run uses one native full-model identity forward per batch, one shortened native prefix, four automatic
directional derivatives, eight finite-difference prefixes, four singleton-removal prefixes, and six pair-removal
prefixes. The conditional validation repeats the same work. It stores only Gram matrices, contracted derivative
checks, pair norms/means, hashes, and call audits—no tokens, logits, or raw hidden states. Deployed parameters saved
and added are both zero.
