# Circuit battery — donor-control sweep preregistration

Registered 2026-09-04T08:2xZ, before the run, AFTER `circuit_battery_preserving_control_repair` (protocol v5) landed at 08:06.
Immutable; the rung's frozen-hash check refuses to execute if this file changes.

## Provenance, stated plainly

v5 established the defect on the seven attn8 successor behaviours and its results are known to me as I register this:
11 of 21 censused tasks have a byte-identical answer-preserving control; `max |d_P_base − d_A1| = 0.0` exactly on those;
the donor-side control separates from the target by a median of only **.072**; and **§2852's negative survived** — zero behaviours
selective, nothing crossing the bar. So pred_c and pred_d below are registered at bars v5's numbers suggest they will MISS. They are
kept at those bars deliberately: they are the bars the campaign has used throughout, moving them to fit a known result would be exactly
the predicate-fitting this lane keeps catching, and the nulls are what the rung is really for.

What is genuinely unmeasured here: every behaviour outside the attn8 seven, every writer other than attn8, and the FULL arm that
§2840 actually published. v5 used §2852's calibrated sub-arms on one writer.

## The defect and the repair

The bank stores a group's families as transformations of ONE situation with the transformation in the **donor**, so A1/A2/P share
`base_text`. Scoring rungs call `pack(b, "base")` for every family, making the answer-preserving control the target's own prompts. This
is visible in the landed §2840 receipt with no GPU at all: `control_d_m["P"]` is BITWISE identical to `split_d_m["FULL"]` for **9 of
the 16** tasks carrying both fields, and every such task with a positive value reads `selectivity_ratio` exactly **1.000**.

This rung scores all 21 behaviours, each at **its own** identified writer, on the battery's **FULL** arm, comparing the degenerate
control (`P_base`) against the donor-side one (`P_donor`: same answer, different causal variable). §2840 took a SIGNED max over
controls; the comparison ratio reproduces that, the corrected ratio uses |.|, and both are reported.

## Predictions, each with its worked-example line

- **pred_a — the structural census predicts the landed receipt.** For every task carrying both fields, `frac_base_in_a1 == 1.00` iff
  `control_d_m["P"] == split_d_m["FULL"]` bitwise; agreement must be **1.00**. *Worked example:* if the identity has the structural
  cause claimed, a no-GPU census of the row generator predicts a GPU receipt written weeks earlier, task for task — agreement
  **1.00**; if the coincidence has some other cause, agreement lands near chance, ≈ **.5**. This is the rung's physical control and it
  cross-checks a fresh census against bytes already on disk.
- **pred_b — the degenerate control reproduces the target.** On tasks with `frac_base_in_a1` = 1.00, max |d_P_base − d_A1| ≤ **.015**.
  *Worked example:* identical prompts, identical arm ⇒ **.000** (v5 measured exactly 0.0); ≥ .1 would mean the census is wrong.
- **pred_c — the donor control separates from the target.** ≥ 6 behaviours with |d_P_donor − d_A1| ≥ **.15**. *Worked example:* if the
  writer is specific to the causal variable, a same-answer/different-item prompt is damaged noticeably less, giving a few tenths; if
  the writer is required for the surface form regardless of the variable, this reads ≈ **.00**. v5 measured **.072** median on seven
  behaviours, so the expected outcome is the null.
- **pred_d — some behaviour is selective once corrected.** ≥ 1 capable behaviour with corrected ratio ≤ **.25**. *Worked example:*
  corrected ratio = max(|d_P_donor|, |d_C_base|)/d_A1; a specific writer reads ≈ **.2**, a form-required writer ≈ **1.0**.
- **pred_e — the correction is not a uniform shift.** IQR of the corrected ratios across capable behaviours ≥ **.20**. *Worked
  example:* if the repair moves every behaviour by the same amount, the corrected ratios are as tightly clustered as the degenerate
  ones (IQR ≈ **.00**) and the correction carries no per-behaviour information; genuine differences give a spread of a few tenths.

## Nulls

- `a_null_census_does_not_predict`: agreement < 1.00 — the structural account of the defect is incomplete.
- `c_null_donor_control_tracks_target`: ≤ 2 behaviours separating — **the writer is required for the surface form, not for the causal
  variable**, and the campaign's negative is not an artifact of the broken control but a fact about the model. This is the expected
  outcome and it is the most valuable one available: it converts "zero behaviours selective" from a number produced by construction
  into a measured claim.
- `d_null_still_none_selective`: nothing crosses .25.
- `e_null_uniform_shift`: IQR ≤ .05.

## Price

≤ 1,600 GPU forwards, 0 backwards, 0 fitted parameters, ≤ 70 GPU-seconds. Receipt:
`circuit_battery_donor_control_sweep_results.json`, read with `price` in the same command the ledger section is written from (§2853).
