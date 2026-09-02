# Rung499 preregistration — prospective copy-task calibration of the finite-action assay

**Registered:** 2026-09-02 18:06 UTC

**Parents:** rung498's exact broad-equality strong null and the frozen CPU task-mask diagnosis.

**Claim level:** prospective calibration on previously uncomputed action outcomes. A pass validates the assay on the
known directed copy-score relation; it does not discover a new circuit, merge whole heads, or compress the model.

## Correction and data boundary

Rung498's action implementation was exact, but its `all_positive` observation meant “the current token has any
earlier equal-token edge.” Rung459's known positive was measured on a copy task: if `p` is the nearest earlier
position containing the current token at `q`, then `token[p+1]` must equal the target `token[q+1]`. A post-result CPU
diagnosis on the already-open documents0:500 showed that this semantic mismatch fully accounts for B's failure. That
diagnosis is not prospective evidence and cannot change rung498's verdict.

Rung499 opens only action outcomes on documents500:1000, which rung498 did not compute. The two fixed quarters are
500:750 and750:1000. All model actions, frozen rung459 RMS scales, thresholds inherited below, and donor/recipient
identities are unchanged.

## Exact task and circuit observations

The primary mask contains positions64:255 satisfying the nearest-predecessor copy condition above. Near/far means
nearest-match distance at most16 versus greater than16. One/multiple means the current token has exactly one versus
more than one predecessor. Off-target means every scored position not satisfying the copy condition.

The30 validation circuit tags were already frozen by top-level family before rung498. Before any model outcome, keep
only tags with at least10 copy-positive member positions and10 copy-positive within-slice control positions in each
quarter. The resulting fixed nine tags are:

`r.1.0, r.1.1, r.1.2, r.1.3, r.11.1.1, r.11.1.2, r.11.3.1, r.3.0, r.5.0.1`.

This support filter uses tokens and existing Boolean circuit masks only. It cannot inspect losses, activations, or
intervention outcomes.

## Actions and computed effects

Use rung498's exact five recipient states—native, absent, donor-score, donor-payload, whole-donor—under both
early-present and early-absent backgrounds for the fixed L5H5 and L7H3 donors. Recompute the complete model suffix.

For background `g`, define the native and hybrid restored per-token effects from loss vectors as

`d_native(g) = CE(late absent, g) - CE(late native, g)`,

`d_hybrid(g) = CE(late absent, g) - CE(hybrid, g)`.

For tasks, compare per-document means. For circuits, subtract the within-slice control mean from the member mean and
stack the signed values across the nine tags. Report cosine and the remaining RMS error after one positive scale.

## Predictions

### A — exact, supported, and untouched instrument

All frozen hashes match; rung498 validation NLL is absent; the two copy-positive quarters each contain at least3,000
positions; every selected circuit member/control cell has at least10 positions; native/replay logits match exactly;
factor reconstruction relative-squared error is at most`1e-10`; every intended edit is nonzero; and call counts equal
125 native plus2,375 analytical forwards. No document0:500 outcome is loaded by the rung499 scorer.

### B — prospective recovery of the known copy-score relation

For `L5H5 score -> L8H4`, in both quarters and both early backgrounds, copy-positive recovery lies in `[.75,1.30]`,
per-document effect cosine is at least`.75`, positive-scale residual is at most`.70`, and absolute off-target CE
change relative to the native recipient is at most`.01 nat. These are rung498's original B bars without change.

### C — task and circuit patterns reject both typed controls

In every quarter/background cell, the L5 score hybrid must beat both the L7 score hybrid and L5 payload hybrid:

- on per-document copy effects, by at least`.10` cosine or`.15` lower scaled residual; and
- on the nine-circuit fingerprint, by at least`.10` cosine or`.15` lower scaled residual.

The old rung498 requirement that the positive also be`.20` closer to one in *total recovery* is reported but does not
gate C. This change is frozen before any document500:1000 action outcome. One scalar total is not identifying: a
control can restore the same total loss while changing the wrong documents or circuits. C instead requires the
signed task and circuit patterns that correspond to computational reuse. The margins are rung498's already frozen
validation margins; they are not fit to rung499 data.

### D — stability under removing the earlier service

For the L5 score hybrid in both quarters and backgrounds, circuit cosine is at least`.60`, scaled circuit residual is
at most`.80`, and the fitted scale is positive. Within each quarter, the hybrid circuit fingerprints under
early-present and early-absent have cosine at least`.65`, and their fitted scales differ by no more than50% relative
to the larger magnitude. These are rung498's frozen held-out closure bars.

### E — calibration scope

E is true only if A/B/C/D are all true. Then the finite-action assay is calibrated for the directed statement
“L5H5's continuous copy score can supply L8H4's score computation while payload and wrong-score donors cannot.” It
licenses a separately registered directed search among the four known equality scores. It does not establish whole-
head equivalence, bidirectionality, OOD generalization beyond FineWeb, selective multi-edge composition, or adoption.

## Nulls and routing

- A false repairs only the instrument.
- A true/B false means the copy-score relation is not prospectively stable on this census; abandon this calibration.
- A/B true/C false means the task/circuit observation set cannot reject typed controls; change observations before
  any search.
- A/B/C true/D false records background-conditional directed reuse and routes to the explicit donor×recipient finite
  interaction, not a quotient.
- A--D true licenses the four-score directed search with all action semantics frozen.

No outcome licenses rank, sparsity, SAE, quantization, or compression tuning.

## Literal price

The run has125 four-document batches. Each batch uses one direct native forward, one analytical replay, and18
donor/background/non-native forwards: exactly2,500 complete forwards. Stored output is per-token CE only. Deployed
parameters saved and added are both zero.
