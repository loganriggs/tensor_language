# Rung 482 preregistration — exact held-out removal of a downstream-defined attention0 direction

Conditional registration written while rung480 is still running and before any rung480 or rung482 outcome is opened.
Run this only if rung480 passes A--E, its strong null is false, and the named winning mode is itself listed in
`passing_b_modes`. A valid rung480 scientific null routes to rung481 instead. A failure confined to the stored
rung424/425 numerical comparison routes first to the already-queued in-process repair.

## Question

Rung480 is a discovery screen. It asks whether downstream circuit effects choose a stable one-dimensional subspace
inside one of the two continuous attention-score factors or the output carried by attention0. Even a complete screen
does not show that this direction is an independently manipulable circuit.

Rung482 asks whether removing that frozen direction:

1. has exact held-out effects predicted by its first-order downstream response;
2. affects prospectively selected target circuits more than predicted non-target circuits;
3. beats one-dimensional directions selected only by activation variance or at random; and
4. remains distinct from removing the rest of the same varying coordinate space.

This is a held-out prediction and selective-removal test. It does not change the fitted ranks, save parameters, or
claim compression.

## Exact attention computation and removal

Rebuild the frozen rung480/rung424 continuous attention0 block in the same process. Its three augmented coordinate
groups have dimensions `7 x 7 x 33`: one fixed affine coordinate plus varying dimensions `6 x 6 x 32`. For one
source-to-query edge, the fitted output is

`F(a,b,c) = sum_i,j,k a[i] b[j] c[k] K[i,j,k]`.

Let `z` denote the varying part of rung480's winning coordinate group, and let `P` be its frozen one-dimensional
orthogonal projector. The constant coordinate is never changed. Define

- `F_full`: evaluate the complete varying coordinate `z`;
- `F_without_P`: replace `z` by `(I-P)z`;
- `F_only_P`: replace `z` by `Pz`; and
- `F_constant`: replace `z` by zero.

At the attention-write level,

`F_full - F_constant = (F_only_P - F_constant) + (F_without_P - F_constant)`

must hold to relative squared error at most`1e-8`. This is exact because the block is linear in each coordinate group
when the other two are fixed.

The fitted block is only part of native attention0. Preserve the native-minus-fitted remainder exactly. The physical
removal output is

`native_attention0_write - (F_full - F_without_P)`.

Thus the experiment removes only the frozen fitted-block contribution associated with `P`; every other attention0
contribution remains. All later attention and MLP modules recompute from the changed residual stream.

## Prediction before exact outcomes

Use the30 reserved odd-root circuit tags with top-level roots `{1,3,5,7,11,13,23}` and documents500:1000. Documents
500:750 are the prediction/selection half; documents750:1000 are the held-out exact-effect half. Use the existing
`member` and matched `slice_control` masks for every circuit.

Before running any exact removal arm, compute the native first-order prediction for each circuit and half. For all
target positions selected by one circuit mask in a document batch, differentiate their **mean CE** with respect to
all winning-mode coordinates `z` in that batch. Contract this complete gradient with the change produced by removing
`P`. This includes every path from an earlier attention0 write to a later selected target, matching the global exact
intervention rather than keeping only the same-position derivative. The predicted CE change is

`predicted_delta = - gradient_z(CE)^T P z = trace(P A)`,

where `A = -sym(z outer gradient_z(CE))`. Average separately on member and matched-control positions. Their
difference is the predicted circuit-selective effect.

Using only documents500:750, freeze:

- **predicted targets:** the ten circuits with largest absolute member-minus-control prediction;
- **predicted non-targets:** the ten circuits with smallest absolute prediction; and
- **nontrivial-sign set:** circuits whose absolute prediction is at least the median absolute prediction.

Ties are broken by the frozen lexicographic circuit-tag order. No exact removal CE may be calculated before these
sets are fixed in memory and written to the execution audit.

## Exact effects and sign convention

For intervention arm `r`, circuit `c`, mask `m`, and half `h`, define

`effect_r[h,c,m] = mean CE_r[h,c,m] - mean CE_native[h,c,m]`.

Positive means removal damages prediction. The circuit-selective exact effect is member minus matched control.
Report native and intervened CE separately as well as their difference; never infer an exact pair effect by summing
single effects.

Run these fixed arms:

1. no removal, returning the native attention0 write directly while separately auditing the
   fitted-block-plus-native-remainder identity;
2. remove the downstream-response projector `P`;
3. remove its varying-space complement `I-P`;
4. remove the activation-only one-dimensional projector reconstructed from rung480's frozen fit rows; and
5. remove16 random one-dimensional projectors in the same winning mode, with seeds `2026090282..2026090297`.

Every control has the same one-dimensional mode rank as `P`, except the deliberately larger complement arm used to
test whether the proposed split has distinct downstream effects.

## Frozen predictions

### A — lawful exact intervention

- rung480 passes A--E, has `strong_null=false`, and its winner belongs to `passing_b_modes`;
- all rung480, rung424/425, circuit, row, model, and source hashes match;
- the rebuilt fitted block, projector, and winning mode reproduce rung480's in-process values;
- the exact `P + (I-P)` write identity above has relative squared error at most`1e-8`;
- the fitted block plus native remainder reconstructs the native attention0 write in float to relative squared error
  at most`1e-12`; the deployed no-removal arm returns that native write directly and reproduces native logits exactly;
  all arms have exact live call/patch counts, and every projector is symmetric, idempotent, rank one,
  and confined to the varying coordinates;
- each removal knob changes the attention0 write, the document750 boundary and circuit supports are exact, and no
  FINAL or SEALED outcome is opened; and
- the target/non-target/sign sets are frozen from first-order predictions before exact removal effects are computed.

### B — first-order response predicts exact held-out effects

On documents750:1000, for the response projector:

- centered predicted versus exact30-circuit selective-effect cosine is at least`.70`;
- after fitting one scalar from predicted to exact effects on documents500:750 and freezing it, held-out relative
  error is at most`.50`;
- at least75% of the nontrivial-sign circuits have the predicted sign; and
- the held-out prediction cosine exceeds the activation-only direction and the95th percentile of the16 random
  directions by at least`.15`.

The controls use their own first-order predictions and exact removals; they do not reuse `P`'s prediction.

### C — prospective target/non-target selectivity

On documents750:1000, using the target and non-target sets selected only on documents500:750:

- the target median absolute exact circuit-selective effect is at least twice the non-target median;
- the target exact-effect norm is at least twice the non-target norm;
- at least eight of ten target circuits have the predicted sign; and
- the member exact-effect norm on the targets is at least`1.5` times their matched-control norm.

### D — exact effects transfer across document halves

The response-projector exact30-circuit selective profile has cross-half cosine at least`.70`; at least seven of the
ten predicted targets remain in the exact top15 by absolute effect on each half; and the target/non-target median
ratio is at least`1.5` on documents500:750 as well as satisfying C on the held-out half.

### E — the physical split has distinct downstream uses

Removing `P` and removing `I-P` must each have nonzero member and member-minus-control effects in both halves. Their
centered30-circuit selective profiles must have absolute cosine at most`.70` in each half, and at least eight circuits
must have opposite-signed effects from the two removals in both halves. Report the same minimum-effect floors
`{0,1e-4,1e-3,1e-2}` descriptively so sign changes near zero are visible without changing the frozen gate.

## Null and routing

Any A failure is an instrument failure. Preserve the receipt and repair only the named measurement; no circuit result
is claimable.

With A valid, the scientific strong null fires if B or C fails. In that case rung480 remains a stable downstream-
response screen, but its one-dimensional projector is not an identified manipulable circuit. Do not tune the rank,
number of selected circuits, target count, thresholds, or random seeds. Return to the independent exact MLP0
branch-by-circuit route rather than searching nearby attention projectors.

A+B+C+D+E identifies one downstream-defined part inside attention0 at the tested circuit-family scope. It does not
yet adopt a smaller program, because no values are removed from storage and the fitted attention block itself is
still an analysis object. The next adoption step must reuse this exact intervention handle in a legal executable
replacement and test composition with at least one downstream consumer.

## Price

Diagnostic GPU work on500 held-out documents: one first-order-response collection before outcomes, then native,
no-removal, response-projector, complement, activation-only, and16 random exact-removal arms. Store only aggregated
CE sums/counts, response profiles, target/non-target identities, projectors, controls, hashes, and execution audits.
Store no raw rows, tokens, logits, attention maps, or hidden states. Deployed parameters saved and added are both zero.
