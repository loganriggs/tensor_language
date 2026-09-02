# Rung498 preregistration — calibrate a finite causal-action quotient on the shared equality matcher

**Registered:** 2026-09-02 17:48 UTC

**Parents:** rung459's held natural-text `L5H5 score -> L8H4 output` transplant, rung460's guarded code transfer,
rung475's62-circuit row/mask authority, and rung497's archive-coverage result.

**Claim level:** prospective calibration of an action-defined grouping method. A pass recovers one already known
directed shared computation and rejects fixed controls; it does not discover a new circuit or compress the model.

## Why calibrate before searching

Rungs495b and496 found no stable cross-head grouping from one downstream derivative fingerprint, even after splitting
attention1 below heads and below complete scores. Rung497 showed that existing finite interventions are too
heterogeneous to construct a closed action table after the fact.

The next method should first recover a positive we already trust. Layer5 head5's equality score can drive layer8
head4's own output on held-out natural text, while layer7 head3's score is a fixed between-candidate control and the
layer5 versus layer8 output vectors are almost orthogonal. This gives the assay one positive and two qualitatively
different negatives before it searches for unknown groups.

## Fixed candidates and directions

The recipient is always `L8H4`. The two earlier donor heads are fixed:

- `L5H5 -> L8H4`: known-positive score donor from rung459;
- `L7H3 -> L8H4`: frozen negative donor used by rung459's interchange test.

For each donor, keep the recipient's native equality score and output/value contribution separately. Construct:

- `late_native`: L8H4's native equality term;
- `late_absent`: remove L8H4's equality term;
- `score_donor`: donor score multiplied by L8H4's own output/value contribution;
- `payload_donor`: L8H4's score multiplied by the donor's output/value contribution; and
- `whole_donor`: donor score multiplied by donor output/value.

The donor score and payload use rung459's frozen fit-row RMS scales without refitting. In particular, the score scales
are `.5371214944` for L5H5 and `.5780356128` for L7H3 into L8H4. The payload scales are `3.1112227972` and
`5.1961131283`. No scale is selected on rung498 outcomes.

## Finite action table

Every late state above is evaluated under two earlier-head backgrounds:

1. `early_present`: retain the donor head's own equality term at its original layer;
2. `early_absent`: remove the donor head's own equality term before the late intervention.

This table contains the required actions:

- remove: `late_native -> late_absent`;
- restore: `late_absent -> late_native`;
- substitute: `late_absent -> score_donor/payload_donor/whole_donor`;
- compose: remove both the early and late equality terms; and
- refinement under a second action: ask whether the donor-to-recipient relation survives changing
  `early_present -> early_absent`.

Every arm edits the exact equality term inside its attention computation and then runs the complete remaining model.
No gradient, local linearization, rank reduction, or activation reconstruction is used.

## Common rows, task masks, and downstream circuits

Use the frozen1000-document census rows and62 circuit masks from rung475/rung481. Equality-positive positions are
computed directly from those same token rows. Split them as follows:

- discovery circuits: the frozen32 tags, documents0:250 for the first half and250:500 for confirmation;
- conditional validation: the frozen30 tags, documents500:750 and750:1000.

No candidate is selected. Validation opens only after the discovery instrument, known-positive, control-separation,
and background-refinement conditions pass.

Retain per-document/per-position CE for every arm. Dedicated task reports use all equality-positive positions, near
versus far previous matches, one versus multiple previous matches, and positions without an equality edge after
position64. Circuit fingerprints use each tag's equality-positive member positions minus its equality-positive
within-slice nonmember positions. The dedicated masks prevent the known matcher from disappearing merely because
the general circuit battery does not name induction; the circuit coordinates test collateral and reuse.

## Computed response vectors

For background `g`, let `B_g` be the final loss vector with L8H4's equality term absent, `N_g` the vector with its
native term, and `H_g` a hybrid. Define the native and hybrid restored effects

`d_native(g) = B_g - N_g`,

`d_hybrid(g) = B_g - H_g`.

For each circuit, average these effects separately on member and matched within-slice control positions and subtract
the two averages. Stack the signed values across circuits. Compare `d_hybrid` with `d_native` by cosine and by the
remaining RMS error after one positive scale. Also report the equality-positive CE recovery

`recovery = sum(d_hybrid on equality positives) / sum(d_native on equality positives)`.

One hundred percent means the hybrid restores the same total equality-positive effect as L8H4's native term; it does
not by itself imply equal effects on individual circuits.

## Predictions

### A — exact, lawful, and live instrument

All source/result/row/mask hashes match; the parent verdicts are unchanged; native versus analytical replay has max
absolute logit error0; equality-factor reconstruction relative squared error is at most `1e-10`; every arm changes
the intended equality term with nonzero RMS; every scored task/circuit cell has support; all call counts match; and
the conditional validation rows/tags remain unopened unless B/C/D hold.

### B — recover the known positive on dedicated equality behavior

For `L5H5 score -> L8H4`, in both discovery halves and both early backgrounds:

- equality-positive recovery lies in `[.75,1.30]`;
- the per-document equality-positive effect cosine is at least `.75`;
- the scaled residual is at most `.70`; and
- off-target absolute CE change relative to `late_native` is at most `.01 nat`.

The broad interval allows the already observed beneficial overshoot while rejecting an inert or sign-reversed assay.

### C — reject the fixed score and payload controls

In both discovery halves and both early backgrounds, the L5H5 score hybrid must exceed both the L7H3 score hybrid and
the L5H5 payload hybrid by at least `.15` in per-document cosine or by at least `.20` in scaled residual, and by at
least `.20` in closeness of recovery to1. The control cannot be chosen separately by cell; the same L5 score relation
must win throughout.

### D — the directed relation is closed under removing the earlier service

For the L5 score hybrid, both `early_present` and `early_absent` circuit fingerprints must match their corresponding
native L8H4 effects with cosine at least `.70` and scaled residual at most `.75` in both discovery halves. The fitted
positive scale may change by at most50% between backgrounds, and the hybrid's two circuit fingerprints must agree
with each other at cosine at least`.75`. This is the first partition-refinement action: removing the earlier redundant
service must not destroy the proposed directed grouping.

### E — prospective circuit and document validation

Only if A/B/C/D hold, open the30 frozen validation circuit tags and documents500:1000. In both fixed quarters and both
backgrounds require equality-positive recovery in `[.65,1.40]`, task-effect cosine at least`.65`, circuit cosine at
least`.60`, scaled circuit residual at most`.80`, positive scale sign, and the same control-separation alternatives
with reduced `.10/.15/.15` margins. The early-background circuit fingerprints must agree at cosine at least`.65`.

### F — interpretation

F is true only if A/B/C/D/E are true. Then the action-conditioned method is calibrated for the directed relation
“L5H5's score can supply L8H4's matching computation,” while correctly keeping its output and the L7H3 donor
distinct. This licenses a separately registered search for new directed simulations or mutual equivalences using the
same action semantics. It is not a new semantic discovery, bidirectional head equivalence, or compression result.

## Nulls and routing

- A failure repairs only the instrument.
- A true/B false means the new census-row assay cannot recover the known task effect; abandon it before quotient
  search.
- A/B true/C false means the metric cannot distinguish known positive and negative relations; abandon or change the
  observation set, not the threshold.
- A/B/C true/D false means the matcher transplant is background-specific rather than closed under the action
  alphabet; record directed conditional reuse, not a quotient state.
- A/B/C/D true/E false means the calibration is discovery-specific.
- A–E true licenses a new-candidate search with the frozen action semantics.

No outcome licenses rank, sparsity, SAE, or compression tuning.

## Literal price

Discovery contains125 four-document batches. Per donor and batch there are two early backgrounds times five late
states, with native/replay baselines shared where exact caching permits. The implementation must print and verify its
exact optimized call formula before enqueue; the unoptimized ceiling is `2 donors × 2 backgrounds × 5 states + 2 =
22` full-model forwards per batch, or2,750 discovery forwards. Conditional validation has the same ceiling. All
stored outputs are per-token CE only; no raw tokens, logits, or hidden states enter the bundle. The experiment saves
and adds zero deployed parameters.
