# Circuit battery — protocol v4 preregistration: selectivity re-scored with the REPAIRED copy control

Registered 2026-09-04T08:0xZ, before the run. Immutable; the rung's frozen-hash check refuses to execute if this file changes.
Amends v3 (§2852) in exactly ONE place — the copy control's construction — and re-runs the calibrated selectivity stage.

## Why, and what this rung can and cannot show

§2851 measured the battery's own copy control and found it broken: median native margin .39, and **−.72 on month**, i.e. the model
does not natively give the copy answer there at all, so any damage measured on it is damage to a behaviour the model does not have.
§2852's v3 gated unusable controls out and scored five of eight behaviours on the answer-preserving family alone; the campaign's
negative (ZERO behaviours writer-selective) survived, median ratio change .000. §2857 repaired the control — a third identical label,
usable controls 3 of 8 → 7 of 7 live, median margin gain 2.63×, month −.72 → +1.89, the SAME design winning on all seven.

The arithmetic decides how to read the outcome, so it is registered here up front. The score is

    ratio = max over USABLE controls of |d_control| / max(d_A1, .5),   d_m = m_NATIVE − m_arm,  POSITIVE = the arm HURTS.

`max` means **adding a second usable control can only raise the ratio** — the repair makes selectivity HARDER to demonstrate, not
easier. The one way it can FALL is if the old, wrong-signed control was contributing a large spurious |d_C| that the proper control
replaces with a smaller one. Both directions are live, and pred_a is registered in the direction that would OVERTURN the campaign's
standing negative, so the null is the negative surviving.

No CE, no §312 L2, nothing installs. (Convention for the frontier, stated per standing rule though unused here: frontier L2 is CE
ADDED ABOVE THE REAL MODEL, LOWER IS BETTER; §2135; the frontier is norm-2304 at 2.6735, §2125 stands.)

## Construction

The repaired control is **derived deterministically from the bank's frozen C rows** by `circuit_battery_copy_control_redesign.variants`
— the exact function §2857 measured — and **the bank is not mutated**. Every `FROZEN_ROW_HASHES` entry and every earlier receipt stays
reproducible; the alternative (editing the C family in `circuit_battery_tasks.py`) would have silently rewritten the rows behind
§2808–§2857. Arm calibration, ladder, ceiling (.80), control usability bar (native margin ≥ .50), splits, and PER_CELL=24 are all
carried over from §2852 unchanged, so the only moving part is the control.

Behaviours: the 7 that §2857 scored (`verbatim_repeat.copy` is the copy behaviour itself and has no separate copy control).
Writer under test: attn8, the roundness/last-item writer (§2842/§2843/§2844). Arms: FULL / half / quarter / eighth reader ladder.

## Predictions, each with the worked-example line

- **pred_a — some behaviour becomes selective.** ≥ 1 behaviour with `ratio_new` ≤ .25.
  *Worked example:* if the old control was inflating the score, a behaviour at §2852's 1.39 with |d_C|=1.2, |d_P|=0.4, d_A1=0.9 and a
  repaired |d_C3|=0.2 reads max(0.4,0.2)/0.9 = **.44**; if the writer is genuinely non-selective every control keeps tracking A1 and
  the ratio stays ≈ 1, so ≈ **1.0–1.4**. Bar .25 is §2852's, unchanged.
- **pred_b — the repair moves the ratio.** median over live behaviours of |ratio_new − ratio_§2852| ≥ .15.
  *Worked example:* a control whose native margin moved .09 → 1.48 (paren list) should not leave the ratio where it was; if the
  repaired control lands at the same damage as P, the move is ≈ **.00** and the repair is cosmetic.
- **pred_c — the new control binds.** the repaired control is the argmax of |d| among usable controls for ≥ 3 behaviours.
  *Worked example:* if it binds nowhere, the ratio is set entirely by P and pred_b's movement (if any) came from usability changes,
  not from the control being informative — which is why this clause is separate from pred_b.
- **pred_d — instrument check: the target arm reproduces.** max |A1 native margin − §2852's| ≤ **.015** (the registered CUDA-atomics
  tolerance) AND every behaviour selects the SAME ladder rung as §2852.
  *Worked example:* same frozen rows, same ladder, same PER_CELL ⇒ expected drift ≈ **.003**. Anything at .1+ means something other
  than the control moved and the comparison to §2852 is void.
- **pred_e — the derived rows are valid.** of every derived control row: single-token answer, joint tokenization intact, answer ≠ its
  own group's A1 successor answer, and the answer token literally present in the prompt. 0 violations required.
  *Worked example:* a design that accidentally made the copy answer equal the successor answer would score as "selective" for a
  spurious reason; §2857 measured 0 violations on the same construction, so ≠0 here means the derivation drifted.

## Nulls (any one met = the rung reports a negative on that clause)

- `a_null_still_none_selective`: 0 behaviours ≤ .25 — **the standing negative survives a stricter test, on two working controls
  for all seven behaviours instead of one control for five of eight.** This is the expected outcome and it is worth the price.
- `b_null_no_movement`: median |Δratio| ≤ .05 — the repair is measurable in the control's own margin but invisible in the score.
- `c_null_new_control_never_binds`: binds on 0 behaviours.

## Price

≤ 900 GPU forwards, 0 backwards, 0 fitted parameters, ≤ 40 GPU-seconds. Receipt:
`circuit_battery_repaired_selectivity_results.json`, read with `price` in the same command the ledger section is written from (§2853).

---

## WITHDRAWN UNRUN, 2026-09-04T08:0xZ — superseded by V5

This document was registered and its rung was smoke-tested but **never landed**; no ledger section cites it and no non-smoke receipt
exists. The smoke run exposed a protocol defect upstream of everything V4 proposed: the answer-preserving control is byte-identical to
the target condition (A1/A2/P share one `base_text` by construction; the family lives in the DONOR), so the ratio V4 set out to
re-score was `|d_P|/d_A1 = 1.000` by construction for most behaviours. Re-scoring that ratio with a better COPY control would have
polished a number whose other term was degenerate. V4 is kept on disk unedited as the honest record of what was registered before the
defect was known. See `CIRCUIT_BATTERY_PROTOCOL_V5_PREREGISTRATION.md`.
