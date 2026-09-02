# Rung 478 preregistration — sparse mixed product gates for a shared downstream response

Registered after the valid rung477b repair and before fitting any mixed product gate. This rung uses only the corrected
32-discovery-circuit response tensor. The30 odd-root circuits and documents500:1000 remain reserved for exact causal
validation.

## Question

Rung477b confirms that individual native product coordinates have no stable cross-MLP behavioral identity: only3/4/4
of13,824 coordinates survive the source/half stability screen and every cross-MLP graph is empty. A shared computation
could still be distributed across different sets of products in each MLP. Test that possibility by defining the common
object in downstream circuit-response space, then finding a sparse weighted implementation in each MLP.

This is not a rank reduction. A candidate must predict the same named circuit-response direction across sources and
document halves, beat alignment-destroyed controls, and later pass exact held-out-family interventions.

## Frozen fit view and common target

From rung477b, form the member-minus-matched-control response matrix `X_m[v]` of shape32 circuits ×4,608 products for
each MLP and each view `v=(document half, matcher source)`. Center every product column across the32 circuits.

Use only `v_fit=(documents0:250, native matcher)` to fit. For MLP pair `(a,b)`, let `u_a=X_a 1` and `u_b=X_b 1` be
the complete-parent first-order profiles. Define one common target direction

`p_ab = normalize(normalize(u_a) + normalize(u_b))`.

If the sum is numerically zero, that pair is invalid. The target is fixed by the parents' common downstream effect;
it is not a freely optimized response direction.

## Two sparse implementations

Fit separate gates for each endpoint MLP using matching pursuit on its fit-view columns:

1. **nonnegative gate:** at each step add the unused product with largest positive residual correlation, then refit
   nonnegative least squares on the active set;
2. **signed gate:** add the unused product with largest absolute residual correlation, then refit ordinary least
   squares on the active set.

Start empty and stop at the first active set whose response has cosine at least`.95` with `p_ab` and optimally scaled
relative L2 error at most`.20`. Stop unsuccessfully if no coefficient improves the residual or after32 terms, the
dimension of the observed circuit-response space. There is no chosen feature count, rank sweep, or top-K fallback.
Normalize each fitted coefficient vector to maximum absolute coefficient1 for later exact intervention; this changes
magnitude but not the fitted response direction.

The nonnegative gate is a weighted partial removal of real equality-induced product contributions. The signed gate is
an algebraic mixed direction and may require adding some contributions; it is reported separately and cannot inherit
the interpretation of a removal-only group.

## Frozen controls and scoring

Apply each fixed gate to all four source×half response matrices. For each pair/arm, report cross-MLP response cosine,
target cosine, and optimally scaled error in every view. Select the candidate with largest minimum cosine over the
three non-fit views; ties prefer nonnegative, then pairs8+9,8+12,9+12.

For16 frozen seeds, independently permute the32 circuit coordinates of the second MLP in the fit view, rebuild the
common target and both endpoint gates, then evaluate those gates on the unpermuted non-fit views. This preserves
within-MLP response geometry but destroys cross-MLP circuit alignment.

## Frozen predictions

### A — lawful corrected input and deterministic solver

- preregistration, rung477b result/bundle/source hashes match;
- rung477b A--E are true and the original native-coordinate null remains true;
- the bundle has exactly two halves, two sources, two mask types, three MLPs,4,608 products, and32 discovery tags;
- all outputs are finite, repeated fitting is exact, and no validation-family/SEALED outcome opens.

### B — a sparse implementation reaches the fit target

For both endpoint MLPs of at least one pair/arm, the solver stops with2–32 active terms, fit cosine at least`.95`, and
optimally scaled relative error at most`.20`.

### C — the shared response survives source and document shifts

The selected candidate has cross-MLP cosine at least`.80` in all three non-fit views, and each endpoint gate has
target cosine at least`.70` in all three non-fit views.

### D — mixing adds information beyond parents and search multiplicity

The candidate's minimum non-fit cross-MLP cosine exceeds the corresponding complete-parent minimum by at least`.15`
and exceeds the95th percentile of the16 alignment-destroyed control scores by at least`.15`.

### E — the response is task-selective and not one-family driven

In every view and endpoint, the gated member-response norm is at least1.5 times its matched-control norm. After
omitting each of the six discovery top-level families from scoring, at least five omissions retain cross-MLP cosine
at least`.70`.

## Strong null and routing

The strong null fires if A fails, no pair/arm reaches the fit criterion at both endpoints, or every candidate has
minimum non-fit cosine at most`.50` or fails to exceed the permuted95th percentile. A+B+C+D+E licenses exact weighted
product interventions on the reserved odd-root circuit families and documents500:1000; a positive signed-only result
must remain labeled a signed edit. A with strong null closes sparse weighted native-product mixtures and routes to a
gauge-aware block-bilinear factorization whose factors are scored in this response metric—not another rank sweep.

## Price

CPU discovery only, zero model forwards, zero deployed parameters saved or added. Report active indices,
coefficients, solver steps, controls, runtime, and literal gate metadata. Do not store validation responses, raw tokens,
logits, or hidden states.
