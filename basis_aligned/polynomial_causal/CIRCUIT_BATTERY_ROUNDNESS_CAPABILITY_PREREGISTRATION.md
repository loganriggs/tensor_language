# CIRCUIT BATTERY — ROUNDNESS CAPABILITY (preregistration)

Registered 2026-09-04 06:08Z (box clock, read immediately before this line was written). Claude, LANE 1 CUDA.
Rung `circuit_battery_roundness_capability`. Script: `ops/circuit_battery_roundness_capability.py`.
Input receipt: `circuit_battery_v2_bank21_results.json` (§2840, sha 7c1db82061bb6cac010fa7d2141cfc8e2faa4330759bdfe5aed299bab9a94d50).
IMMUTABLE: any change gets a new document, not an edit.

## Object

Two facts sit oddly together. §2817 measured that on a bare numeric run the model answers LAST + 1 rather than LAST + STEP (capability
.92 against .06) — "a +1 machine". §2840 then found that my hand-probe of `10% 20% 30%` was continued correctly BY THE STEP, while the
bank's generated `13% 23% 33%` collapsed to capability .07. The obvious hypothesis is that the hidden variable in both is ROUNDNESS: the
model can continue a run by its step when the values are round, and falls back to +1 (or to nothing) when they are not.

If that is right, it changes how several earlier capability numbers should be read, and it is a fact about the model rather than about
any circuit. This rung measures capability directly against the start value's roundness class, with the step fixed at 10 within every
comparison so that step size cannot confound it.

**This is a MODEL-PROPERTY measurement, not a circuit claim.** It uses its own value sets rather than the bank's frozen splits (the
whole point is to sweep values systematically), and it makes no localisation, selectivity or L2 claim. Capability is argmax accuracy over
a shared numeric candidate vocabulary; HIGHER IS BETTER. Every prompt is checked for the joint-tokenization prefix property before use,
and prompts failing it are dropped and counted.

**Value classes, fixed before the run:** `tens` = multiples of 10 (10…90); `fives` = ≡5 mod 10 (15…95); `other` = everything from 11 to
99 that is not a multiple of 5. **Formats, fixed before the run:** percent run (`10% 20% 30%` → ` 40`), bare run (`10 20 30` → ` 40`),
numbered list (`10. alpha / 20. beta` → `30`), keyed line (`Step 10 / Step 20 / Step` → ` 30`), and one SUCCESSOR format
(`bare_run_succ`, same prompt as the bare run but scored against LAST + 1 — §2817's actual measured behaviour).

## Predictions

```
BARS  = {percent_gap: .30, general_formats: 3, format_gap: .15, successor_gap: .15, tens_over_fives: .10}
NULLS = {percent_gap_le: .05, general_formats_le: 1, successor_gap_ge: .40, tens_over_fives_le: -.10}
```

**pred_a_percent_runs_need_round_starts** — `accuracy(percent_run, tens) − accuracy(percent_run, other)` ≥ .30.
*Worked example:* §2840's hand-probe succeeded on round starts and the bank's non-round starts read .07, so the hypothesis predicts
tens ≈ .5–.9 against other ≈ .0–.2, a gap of .4–.8. If roundness is irrelevant and my probe was simply lucky, the gap is ~0 and §2840's
collapse needs a different explanation. A DIFFERENCE of two accuracies, both in [0, 1]; no ratio. Null: ≤ .05.

**pred_b_roundness_is_general** — at least 3 of the 4 step-continuation formats show a tens-minus-other gap ≥ .15.
*Worked example:* if roundness is a property of how this model handles numbers, it appears wherever a step must be carried, so 3–4 of 4;
if it is specific to the percent surface, 1. Count over four formats. Null: ≤ 1.

**pred_c_successor_is_roundness_robust** — |`accuracy(bare_run_succ, tens) − accuracy(bare_run_succ, other)`| ≤ .15.
*Worked example:* §2817 found LAST + 1 at capability .92 across the bank's non-round starts, so the successor behaviour should NOT
depend on roundness: both classes .8–1.0 and a gap near 0. If the successor is ALSO roundness-dependent, then roundness is not about
step-carrying but about numbers generally, and §2817's "+1 machine" framing is the thing that needs revision — which is why this is
registered as an absolute-value bound rather than a one-sided one. Null: ≥ .40.

**pred_d_tens_beat_fives** — `accuracy(percent_run, tens) − accuracy(percent_run, fives)` ≥ .10.
*Worked example:* multiples of 5 are intermediate in canonicality between multiples of 10 and arbitrary values, so if roundness is
graded rather than binary this reads .1–.4; if the model treats all multiples of 5 alike it reads ~0, which would say the effect is
"multiple of five" rather than "round". A DIFFERENCE of accuracies. Null: ≤ −.10 (fives beating tens).

**pred_e_instrument_reproduces_argmax** — every (format, class) cell retains at least one prompt after the joint-tokenization filter.
*Worked example:* a cell emptied by the filter would make its accuracy NaN and any gap involving it unreadable; this is a completeness
check on the measurement, not a claim.

## Stated null

Roundness is irrelevant: the percent gap is ≤ .05, at most one format shows a gap, the successor is as roundness-dependent as the step
continuation, and multiples of five behave like multiples of ten. Then §2840's collapse has another cause and my hand-probe was simply
unrepresentative in some other way, which I would record as such.

## Price

5 formats × 3 roundness classes × ≤ 90 prompts, batched by token length at 32 per forward.
Literal budget: ≤ 120 GPU forwards, 0 backwards, **0 fitted parameters**, < 60 GPU-seconds.

## What this does NOT claim

Capability only — no localisation, no selectivity, no intervention of any kind, and nothing about the §312 frontier. One step size (10)
and one digit range (10–99); a different step or a three-digit range could behave differently and is not tested. The value sets are this
rung's own, so nothing here is scored on the bank's frozen splits and no number from it may be quoted as a bank capability. Does not
satisfy Codex's four-phase integration contract; updates no circuit record.
