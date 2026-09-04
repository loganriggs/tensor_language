# CIRCUIT BATTERY — WRITER ARM SATURATION (preregistration)

Registered 2026-09-04 07:02Z (box clock, read immediately before this line was written). Claude, LANE 1 CUDA.
Rung `circuit_battery_writer_arm_saturation`. Script: `ops/circuit_battery_writer_arm_saturation.py`.
Input receipts: `circuit_battery_node_vs_edge_selectivity_results.json` (§2850, sha 575ef8071ba51b3097eb76fd84ea574c8abd040233b637249b8037ec1b4f262e)
and `circuit_battery_v2_bank21_results.json` (§2840, sha 7c1db82061bb6cac010fa7d2141cfc8e2faa4330759bdfe5aed299bab9a94d50).
IMMUTABLE: any change gets a new document, not an edit.

## The question, and it is about the campaign's most-repeated result

§2850 showed that §2849's selectivity ratios were arithmetic rather than evidence: whole-component ablation removes **1.538×**
the native margin on the target family and **1.538×** on the controls, so every ratio pinned at exactly 1.00 and said nothing
about selectivity.

The battery's own selectivity stage uses a **different** arm — the writer's final-position write removed from every reader edge
plus the direct path — and its central negative result, **"ZERO behaviours are writer-selective"**, has been reported three
times: §2817 (repaired bank, 8 capable), §2840 (21-behaviour bank, 9 capable), and §2819's reader-side version. **Nothing has
checked whether that arm is saturated too.** If it is, the most-repeated finding of this campaign is the same artifact at a
different granularity, and I would rather find that myself than have it found for me.

§2850 also surfaced a second hazard: the copy control's native margin on the numbered list is **.18**, small enough that
`max(|d_P|, |d_C|)` may be dominated by a quantity measured against a near-zero baseline regardless of saturation.

Arms, on §2840's capable attn8-writer behaviours and the SELECT split the battery's selectivity stage actually scores:
`FULL` (the battery's arm — the write removed from all 19 reader edges plus the direct path) and `HALF` (a genuine partial arm:
the write removed from only the first half of the reader edges). Sign convention: damage d_m = m_NATIVE − m_arm, POSITIVE = the
arm HURTS; saturation = d_m / max(m_NATIVE, .5); ratio = max(|d_P|, |d_C|) / max(d_A1, .5), LOWER IS MORE SELECTIVE. **No CE and
no §312 L2 — the frontier's L2 is CE ADDED ABOVE THE REAL MODEL where LOWER IS BETTER (§2135, norm-2304 at 2.6735); nothing here
installs.**

## Predictions

```
BARS  = {sat: .60, ctrl_sat: .60, repro: .15, half_agree: .15, c_margin: .50, c_margin_frac: .50, floor: .5}
NULLS = {sat_ge: .90, ctrl_sat_ge: .90, repro_ge: .40, c_margin_frac_le: .20}
```

**pred_a_the_writer_arm_is_not_saturated** — median over behaviours of (FULL arm's A1 damage / native A1 margin) ≤ .60.
*Worked example:* §2850 measured the node arm at 1.538 and the edge arms at .053; the battery's writer arm sits between those in
size, and if it lands .2–.6 its ratios are informative and the three writer-selectivity results stand. If it lands ≥ .90 the
result is the §2849 artifact again and **three sections must be re-read**. Registered in the direction that PRESERVES the
campaign's finding, so that failure is the outcome that costs me something. Null: ≥ .90.

**pred_b_controls_are_not_saturated** — median over behaviours of max(P saturation, C saturation) ≤ .60. *Worked example:* the
ratio is uninformative if EITHER side is pinned; §2850's node arm had both at 1.538. Both this and pred_a must hold for the
ratios to mean anything.

**pred_c_ratios_replicate_the_battery** — max over behaviours of |this rung's FULL ratio − §2840's published selectivity ratio|
≤ .15. *Worked example:* the same arm on the same split and rows, so agreement should be within sampling noise. A larger gap
means this rung is not measuring the battery's arm and neither pred_a nor pred_b transfers to §2817/§2840. Instrument check.

**pred_d_half_strength_agrees** — max over behaviours of |FULL ratio − HALF ratio| ≤ .15. *Worked example:* if the selectivity
verdict is a property of the circuit it should not depend on how much of the write is removed; if the two ratios diverge, the
verdict is a function of arm strength and "zero behaviours are writer-selective" is fragile even without saturation. (My first
draft of this arm used `ablate=False`, which is `CB.run`'s default and reproduces FULL exactly — a no-op predicate. Caught
before this document existed, so no bar was ever set against the broken version.)

**pred_e_the_copy_control_has_a_usable_margin** — at least half the behaviours have a copy-control native margin ≥ .50.
*Worked example:* §2850 measured .18 on the numbered list. If most behaviours are like that, the C term in every selectivity
ratio this campaign has computed is measured against a near-zero baseline, which is a separate defect from saturation and would
need its own correction. Null: median C margin ≤ .20 of the A1 margin.

## Stated null

The writer arm is saturated (≥ .90) on both target and controls — in which case §2817's, §2840's and §2819's "zero behaviours
are writer-selective" is the §2849 artifact at another granularity, and I will write that correction against those sections
rather than around them.

## Price

≤ 8 behaviours × 3 families × 24 SELECT rows × (1 native + 2 arms), batched by token length.
Literal budget: ≤ 1,500 GPU forwards, 0 backwards, **0 fitted parameters**, < 90 GPU-seconds. Uses `ops/fastload.py`.

## What this does NOT claim

Saturation is measured against the native MARGIN, the battery's own metric; an arm could be unsaturated in margin and still
distort the output distribution in ways margin does not see. One split (SELECT, the one the battery scores selectivity on),
attn8-writer behaviours only. Nothing installs; no L2. Does not satisfy Codex's four-phase integration contract; updates no
circuit record.
