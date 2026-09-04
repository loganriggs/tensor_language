# CIRCUIT BATTERY PROTOCOL v3 — CALIBRATED SELECTIVITY (preregistration / protocol amendment)

Registered 2026-09-04 07:06Z (box clock, read immediately before this line was written). Claude, LANE 1 CUDA.
Rung `circuit_battery_calibrated_selectivity`. Script: `ops/circuit_battery_calibrated_selectivity.py`.
Input receipts: `circuit_battery_writer_arm_saturation_results.json` (§2851, sha 764035134854e0196e3ff0aed1200e14b7d26ad10beaaf93ad3fd52d9a3a46ab)
and `circuit_battery_v2_bank21_results.json` (§2840, sha 7c1db82061bb6cac010fa7d2141cfc8e2faa4330759bdfe5aed299bab9a94d50).
IMMUTABLE, and a **PROTOCOL AMENDMENT**: the v2 document said amending a bar re-runs every behaviour, and this changes the
selectivity stage for every behaviour present and future. v2 is untouched; this supersedes only its selectivity stage.

## What is being amended and why

§2851 established that the battery's selectivity stage cannot measure what it claims:

- the writer arm removes a median **1.207×** the native target margin and **2.552×** the native control margin, so both sides of
  the ratio are pinned past the ceiling and six of eight behaviours read exactly 1.00;
- the copy control's own native margin is a median **.39** — **.09** on the paren list, **.20** on the numbered list and
  **−.72** on month, where the model does not natively give the copy answer at all — so the C term has been measured against a
  near-zero or wrong-signed baseline in every selectivity ratio this campaign has computed.

Two changes, both fixed before this run:

**CALIBRATED ARM.** On FIT rows, walk a fixed ladder — the write removed from all reader edges plus the direct path, then the
first half of the reader edges, then a quarter, then an eighth — and take the LARGEST whose target saturation is ≤ **.80**, i.e.
the strongest arm that still leaves a fifth of the native margin standing. Score selectivity on SELECT with that arm. If no rung
of the ladder qualifies, the weakest is used and the behaviour is reported with its saturation.

**VERIFIED CONTROL.** A control family enters `max |d_control|` only if its OWN native margin on SELECT is ≥ **.50** and
positive. A behaviour whose copy control fails is scored on the answer-preserving family alone and flagged, rather than silently
carrying a term with a wrong-signed baseline.

Sign convention: damage d_m = m_NATIVE − m_arm, POSITIVE = the arm HURTS that family's own answer; saturation =
d_m / max(m_NATIVE, .5); ratio = max over USABLE controls of |d_control| / max(d_A1, .5), LOWER IS MORE SELECTIVE. **No CE and
no §312 L2 — the frontier's L2 is CE ADDED ABOVE THE REAL MODEL where LOWER IS BETTER (§2135, norm-2304 at 2.6735); nothing here
installs.**

## Predictions

```
BARS  = {sat: .80, selective: .25, n_selective: 1, usable_controls: 4, material: .15, repro: .15, floor: .5}
NULLS = {sat_ge: 1.00, n_selective_le: 0, usable_le: 1, material_le: .05}
CEILING = .80 ; CONTROL_MIN_MARGIN = .50
```

**pred_a_calibration_lands_below_the_ceiling** — median over behaviours of the chosen arm's SELECT saturation ≤ .80.
*Worked example:* §2851 measured the full arm at 1.207 and the half arm at .797, so the ladder should mostly select "half" and
land .5–.8. If even the eighth arm saturates, the ladder is too coarse and nothing else here is readable. Null: ≥ 1.00.

**pred_b_some_behaviour_is_now_selective** — at least one behaviour has a calibrated ratio ≤ .25.
*Worked example:* §2851's half-arm probe gave .69 (numeric run) and **.20** (verbatim repeat), so one behaviour plausibly clears
.25 once the arm is calibrated. **This is the prediction that would overturn "ZERO behaviours are writer-selective" as a fact
about the model rather than about the instrument.** If it fails, the campaign's negative survives calibration and becomes much
better supported than it was — which is the outcome that costs me the correction I have been building toward. Null: zero
behaviours.

**pred_c_enough_copy_controls_are_usable** — at least 4 behaviours have a copy control with native margin ≥ .50.
*Worked example:* §2851 measured 1.66, 1.51, 2.51, .39, .20, .09, −.72, n/a across eight, so 3–4 clear .50. If fewer than 2 do,
the copy control is not a usable instrument on this bank at all and the C family needs redesigning, not just gating. Null: ≤ 1.

**pred_d_the_correction_is_material** — median over behaviours of |calibrated ratio − §2840's published ratio| ≥ .15.
*Worked example:* if calibration changes nothing, the amendment is bureaucracy and §2840's numbers stand as they are; §2851's
half-arm moved ratios by up to .38, so .15–.4 is expected. A small value would say the saturation, though real, did not distort
the verdict. Null: ≤ .05.

**pred_e_the_full_arm_still_reproduces_the_battery** — max over behaviours of |this rung's FULL-arm ratio − §2840's published
ratio| ≤ .15. *Worked example:* the uncalibrated arm re-measured here must still match §2840, or this script is not measuring
the same thing and the comparison in pred_d is meaningless. §2851 reproduced six of eight exactly and diverged only on month,
whose control has a negative native margin — so this may fail on that one cell, and if it does I will report which cell rather
than the aggregate. Instrument check.

## Stated null

Calibration changes nothing material (≤ .05), no behaviour becomes selective, and the copy controls remain unusable. Then
"ZERO behaviours are writer-selective" survives its own instrument critique and is a much stronger result than when it was
first reported — and §2851's correction reduces to "the earlier arm was blunt but not misleading".

## Price

≤ 8 behaviours × [4 ladder arms on FIT A1 + 3 families × (native + calibrated arm) + 3 families × full arm] × 24 rows,
batched by token length. Literal budget: ≤ 4,000 GPU forwards, 0 backwards, **0 fitted parameters**, < 3 GPU-minutes.
Uses `ops/fastload.py`.

## What this does NOT claim

The ladder is four fixed strengths, not a search for the exact ceiling; the arm chosen is the strongest qualifying one, so a
behaviour may sit well below .80. Calibration is on FIT and scoring on SELECT, so the arm choice is a selection and only the
score is held out. attn8-writer behaviours only. Nothing installs; no L2. Does not satisfy Codex's four-phase integration
contract; updates no circuit record.
