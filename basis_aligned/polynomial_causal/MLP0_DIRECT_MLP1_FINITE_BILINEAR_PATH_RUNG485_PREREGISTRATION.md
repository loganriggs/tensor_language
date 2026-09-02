# Rung 485 preregistration — exact finite MLP0-to-direct-MLP1 bilinear path

Written after rung484's registered strong null and before any rung485 model outcome.

## Question

Rung484 exactly split attention1 into its two score sides and carried-value side. The path profiles were extremely
stable and `T` versus `I` were oppositely directed, but no proper attention subset predicted the complete token-only
route. The one eligible `I` subset did not become task-specific on equality positions and failed the second half's
selection bar. The attention path is therefore useful anatomy, not an identified equality circuit.

Rung483 independently found that complete `T` and `I` removals have only `.25/.26` response cosine at MLP1 when
attention1 is held native. Rung485 follows rung484's frozen failure route and asks whether MLP1's exact two-sided
bilinear multiplication provides a simpler finite path for that direct response. It also tests whether the token-only
route has repeatable token-conditioned effects, which would license grouping tokens by what downstream computation
does with them rather than by raw embedding or MLP0 product coordinates.

## Exact direct MLP1 states

Reuse rung401's exact MLP0 branches `T` and `I`. For branch `b`, compute:

- the native input to MLP1; and
- the input with all of `b` removed from MLP0 while attention1's write is restored to its native value.

Thus every arm keeps the branch absent from the direct residual stream and prevents attention1 recomputation from
entering the measurement. This is the finite version of rung483's `MLP1 direct` consumer.

## Exact bilinear factorization

For normalized MLP1 input `z`, the deployed write is

`MLP1(z) = Down1[(Left1 z) elementwise-multiplied-by (Right1 z)] + bias1`.

Let `(L_N,R_N)` be the two 4,608-dimensional activation vectors at the native input and `(L_b,R_b)` those at the
branch-absent direct input. Rebuild all four writes

`W_b(i,j) = Down1[L_i elementwise-multiplied-by R_j] + bias1`,

where each of `i,j` chooses the native or branch-absent state. The native/native and absent/absent corners must
reproduce the corresponding deployed MLP1 writes. The mixed corners are exact finite cross-state computations.

Reciprocal rescaling or joint permutation of native product coordinates leaves every rebuilt write unchanged. A
global exchange of Left and Right only swaps the two labels, so shared-versus-split conclusions and the existence of
a proper one-side path do not depend on which side is called Left.

## Physical downstream arms

Keep branch `b` absent at MLP0, restore attention1's native write, inject each of the four rebuilt MLP1 writes, and
let layers2--17 and the output recompute normally.

For subset `S` of `{L,R}`, let `y_b(S)` be minus per-token CE with native activations restored on the sides in `S`.
The two singleton effects and their interaction are

`L = y({L}) - y({})`,

`R = y({R}) - y({})`,

`LR = y({L,R}) - y({L}) - y({R}) + y({})`.

They must sum exactly to the complete direct-MLP1 restoration effect. These are effects of this named native/absent
causal coordinate, not claims that the trained product axes are semantic units.

## Physical path selection

On documents0:250, separately for `T` and `I`, test the two nonempty proper subsets `{L}` and `{R}`. A side is
eligible when its per-token CE effect predicts the complete direct-MLP1 restoration effect with cosine at least`.90`
and best scalar-adjusted relative error at most`.35`. If both qualify, minimize error and then prefer `L`. The full
two-side corner is excluded because it predicts itself trivially. If neither qualifies, selection is null and B
fails. Apply the selected side unchanged on documents250:500.

## Token-conditioned view

The data are the same hash-bound 1,000 documents used by rungs477b,481,483,484. Discovery is documents0:500 split at
250. Exactly 698 token IDs occur at least eight times in each discovery half, covering `39,723/39,981` positions.
This set is determined from input IDs before model outcomes and is frozen.

For each branch's complete direct-MLP1 restoration effect:

1. estimate one mean signed effect for each of the 698 tokens on documents0:250;
2. apply those means without refitting to every occurrence in documents250:500; and
3. compare prediction prediction error with one global mean fitted on documents0:250.

Also compare the 698-dimensional token-mean profiles between halves. This is a diagnostic lookup, not a proposed
compressed circuit. A pass licenses a separately validated grouping of tokens with similar downstream effects; it
does not license storing 698 labels as the explanation.

Of those 698 frozen IDs, 656 occur at least four times in each validation quarter. If validation opens, use exactly
those 656 tokens and the original document0:250 means; do not refit on validation.

## Data controls

Report every result on all positions and on exact equality-positive positions, but equality selectivity is not a
gate: rung484 already showed that the attention path is generic at that behavior. Use the same 16 cyclic
position-shift controls for same-position prediction, and report token-frequency-weighted as well as unweighted
token-profile results so rare tokens cannot dominate a cosine.

Documents500:1000 remain unopened unless the discovery license below holds. FINAL and SEALED remain closed.

## Frozen predictions

### A — exact and lawful instrument

- Rung401/481/483/484, row, model, and source hashes match; rung484 remains a valid strong null with validation closed.
- Native replay and restored-attention1 state replay are exact.
- Native/native and absent/absent MLP1 factor reconstructions have relative squared error at most`1e-8` in float32
  and at most`1e-5` after deployed BF16 conversion.
- All four arms run with exact attention, MLP, MLP0-removal, attention1-restoration, and MLP1-injection call counts.
- The three Möbius effects reconstruct the complete direct-MLP1 CE vector to relative squared error at most`1e-8`.
- Both branch effects, both sides, and the complete route are nonzero; validation/FINAL/SEALED remain unopened.

### B — one physical MLP1 side predicts each complete direct route

For both `T` and `I`, the side selected on documents0:250 must predict the complete direct-MLP1 restoration effect on
documents250:500 with cosine at least`.80` and best scalar-adjusted relative error at most`.50` on all positions. The
same-position cosine must exceed the 95th percentile of the 16 position-shift controls by at least`.15`.

### C — the bilinear path profile is stable

For each branch, normalize the three signed mean effects `(L,R,LR)` by their total absolute size. Its profile cosine
between discovery halves must be at least`.85`, every absolute share may change by at most`.20`, and the side chosen
descriptively within each half must equal the frozen half0 selection.

### D-shared — `T` and `I` use the same direct MLP1 path

Their `(L,R,LR)` profile cosine is at least`.90` in both halves, their selected sides are identical up to a single
global Left/Right exchange, and one scalar fitted from T to I complete-route effects on half0 predicts half1 with
relative error at most`.35`.

### D-split — `T` and `I` use different direct MLP1 paths

Their profile cosine is at most`.60` in both halves and their selected sides differ after allowing one global
Left/Right exchange. Each selected side's own-route cosine must exceed the cosine from applying that side to the other
branch by at least`.20` in both halves.

The two D outcomes are mutually exclusive. Intermediate values identify no relation.

### E — token identity predicts the token-only downstream effect

For `T`, the 698-dimensional mean-effect profile has weighted and unweighted cross-half cosine at least`.70`. The
half0 token means reduce half1 per-position RMSE by at least10% relative to the half0 global-mean predictor, and their
Pearson correlation with half1 per-position effects is at least`.30`. Report the identical measurements for `I`
without requiring them to pass. Also report T-minus-I margins; this tests the expected token-only/contextual
asymmetry without making I failure automatic.

### F — held-out documents

Open documents500:1000 only if A/B/C/E hold and exactly one D outcome holds. Require A/B/C and the same D relation
again with unchanged selected sides. For the 656 frozen validation-supported tokens, the document0:250 T-token means
must reduce validation per-position RMSE by at least5% versus the same frozen global mean, with positive correlation
in both validation quarters. No path, token set, mean, scale, or threshold may change after validation opens.

## Null and routing

The scientific strong null fires when A, B, C, or E fails or neither D outcome holds.

- If A alone fails, repair only the replay/factor/accounting instrument.
- If B fails, neither immediate consumer has a proper finite architectural subpath; preserve their full finite
  responses and move to a coupled token-by-context response tensor across both consumers rather than another subset,
  rank, or sparse fit.
- If C fails, retain per-example direct responses and test an explicitly context-conditioned route.
- If E fails, the token-only formula does not imply a stable token-conditioned downstream role; condition the reader
  on live state rather than forcing token groups.
- A validated D-split licenses separate below-product/state analysis for T and I.
- A validated D-shared licenses exact cross-branch finite interchange through the shared side.
- An E pass licenses clustering the 698 token effects with a frozen complexity rule, followed by held-out physical
  within-group versus between-group interchange. Clustering itself cannot be the claim.

No outcome here is compression or adoption. The experiment identifies or falsifies a finite direct-consumer path and
tests whether the original “tokens with the same downstream effect” proposal has measurable support.

## Price and stored data

Discovery runs one native forward and eight branch/arm forwards per four-document batch: `1,125` full-model forwards.
Conditional validation repeats the same fixed price. Store only contracted CE statistics, factor/replay errors, token
counts and means for the frozen frequent set, hashes, and call audits. Store no logits, hidden states, or raw token
sequences. Deployed parameters saved and added are both zero.
