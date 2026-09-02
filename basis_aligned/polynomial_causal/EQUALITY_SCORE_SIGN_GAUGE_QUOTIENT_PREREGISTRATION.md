# Preregistration — Z2 sign-gauge quotient of the equality-score families (parallel lane)

Date: 2026-09-02 19:35 UTC
Owner: Claude (parallel probe lane)
Status: frozen before any negated-transplant outcome exists

## Question and lineage

Rung 501 (§2627) split the four equality scores into two sign-coherent families
on the frozen fit rows — {L5H5, L8H4} vs {L7H3, L8H3}: cross-family score
cosines −.79 to −.92, within-family +.84 — and every cross-family POSITIVE-scale
transplant landed sign-reversed (task cosine −.65 to −.91). Mathematical review
1907 (move #3) asks whether this anti-alignment is a pure Z2 GAUGE: if the two
families realize ONE abstract score computation up to global sign, then a
cross-family transplant with a NEGATED frozen scale should satisfy 501's full
edge criteria. If instead the anti-alignment is feature-mixed, the negated
transplant lands off-manifold and fails — a clean null. A pass halves the
program's score dictionary; a null certifies the family split as non-gauge
structure. Codex's main line (1908 decision) is at MLP9 source factorials —
no collision; 501's frozen module is imported hash-pinned, nothing registered
is modified.

## Arms (all named; rung 501 machinery verbatim)

Candidates: the two cross-family directions into the calibrated recipient —
`L7H3->L8H4` (501's known negative) and `L8H3->L8H4`. Scales are recomputed
in-run by 501's exact 24-forward fit pass (post-§2599: no numeric bridges);
the NEGATED arms multiply only score_ratio by −1; payload_ratio untouched
except in the NEG-PAYLOAD control.

Per batch (125 batches over documents 0:500, halves split at 250), per pair and
background (early_present / early_absent, 501's definitions):

- LATE_NATIVE (replay for early_present; own trajectory for early_absent);
- LATE_ABSENT;
- NEG_SCORE: donor score with scale −score_ratio, recipient payload kept;
- POS_SCORE: 501's original positive-scale arm, re-measured in-run (control:
  the anti-alignment must reproduce);
- NEG_PAYLOAD: donor payload with −payload_ratio, recipient score kept
  (specificity control: negating the wrong factor must not help).

2 shared + 2 pairs × 9 = 20 forwards/batch; 2,500 + 24 = 2,524 total.

## Scoring

501's exact edge criteria applied verbatim to the NEG arms (its `_partition_edge`
logic on an identically-shaped analysis, with NEG_SCORE in the score_donor slot
and NEG_PAYLOAD in the payload_donor slot): task recovery in [.65,1.40], task
cosine ≥ .75, residual ≤ .70, off-target ≤ .01 nat; MLP9 reader cosine ≥ .75,
residual ≤ .70, RMS floors; payload rejection ≥ .30; copy-specificity ≥ .10;
background closure (response cosine ≥ .75, recovery change ≤ .30, scale drift
≤ 50%). All in BOTH document halves. Higher recovery toward 1 = restoration;
negative cosine = anti-aligned.

## Frozen predictions

### pred_a — exact, live instrument
501's full suite: hashes match (501 source/result, this prereg); native/replay
logit and MLP9 max-abs 0.0; factor reconstruction ≤ 1e-10; scale pass exact
(24 calls, all scales live and finite); every intended edit nonzero; calls exact
(2,500 phase + 24 scale); every task cell supported; reader token counts all
positive.

### pred_b — L7H3's score is L8H4's computation up to sign
NEG_SCORE `L7H3->L8H4` is an EDGE by all criteria above in both halves, AND
in-run POS_SCORE reproduces the anti-alignment (task and reader copy-cosines
< 0 in every background/half — 501's negative tripwire re-measured).

### pred_c — the gauge extends to the second family member
Same two clauses for `L8H3->L8H4` (NEG_SCORE edge in both halves; POS_SCORE
anti-aligned everywhere).

Descriptive regardless: full arm tables both pairs, NEG_PAYLOAD outcomes,
negated-vs-positive cosine symmetry (|cos_neg| vs |cos_pos|), and the implied
dictionary arithmetic if any clause passes.

## Strong null and interpretation

Strong null fires if pred_a fails or BOTH pred_b and pred_c fail. Null: the
anti-alignment is feature-mixed, not a global sign gauge — the family split is
real structure, 501/§2627 stand unchanged, and the score dictionary stays at
four entries; route = none (this closes review move #3; no scale sweep, no
per-feature sign fitting). A pred_b-only or pred_c-only pass is reported as a
partial gauge (one direction) and licenses ONLY a registered validation on
documents 500:1000 plus the reverse direction — no equivalence language without
both directions and validation, per 501's typed-semantics discipline. No
compression claim on any route.

## Literal price

2,524 full-model forwards (~4 min), single phase, documents 0:500 only; no
validation or sealed roles opened. Sufficient statistics + per-document
copy-cell CE sums stored; no raw tokens/logits/MLP9 vectors. Zero deployed
parameters.
