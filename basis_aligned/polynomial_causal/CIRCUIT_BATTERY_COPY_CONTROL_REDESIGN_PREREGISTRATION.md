# CIRCUIT BATTERY — COPY-CONTROL REDESIGN (preregistration)

Registered 2026-09-04 07:38Z (box clock, read immediately before this line was written). Claude, LANE 1 CUDA.
Rung `circuit_battery_copy_control_redesign`. Script: `ops/circuit_battery_copy_control_redesign.py`.
Input receipts: `circuit_battery_calibrated_selectivity_results.json` (§2852, sha bb493ffa74ac41381155a8c72a915aa92cc6714200fee0b9e19e590be892ee4f)
and `circuit_battery_v2_bank21_results.json` (§2840, sha 7c1db82061bb6cac010fa7d2141cfc8e2faa4330759bdfe5aed299bab9a94d50).
IMMUTABLE.

## The defect this repairs

§2851 measured the copy control's OWN native margin at a median **.39** — **.09** on the paren list, **.20** on the numbered list, and
**−.72** on month, where the model does not natively give the copy answer at all. §2852's protocol v3 gated unusable controls out of
the ratio, which was the right instrument fix and left a bank defect in plain view: **five of eight capable behaviours are now scored
on the answer-preserving family alone.** A selectivity stage running on one control for most of the bank is not measuring selectivity,
and §2852 recorded this as "bank work, not instrument work" and stopped there.

Three copy-control DESIGNS are measured per behaviour, and one is adopted only where it clears the **.50** usability bar §2852 already
registered. Only the CONTROL family changes; the target family A1 is untouched, and pred_d measures that rather than asserting it.

- **v1_current** — the bank's existing repeated-item form (`"48. a / 48. b" → "48"`), the baseline to beat.
- **v2_triple** — the same idea with one more identical-label line (`"48. a / 48. b / 48. c" → "48"`): more repetitions of the token to
  copy, so the copy reading is more strongly cued without changing what is being controlled for.
- **v3_adjacent** — the token to copy placed immediately before the answer position, the form §2841's `verbatim_repeat.copy` behaviour
  showed the model performs at capability 1.00.

Sign convention: native margin m = logit(answer) − max logit(other candidate in the task's vocabulary), **HIGHER means the model
natively gives that answer**; a control family is USABLE when its own native margin is ≥ .50 and positive. **No CE and no §312 L2 —
the frontier's L2 is CE ADDED ABOVE THE REAL MODEL where LOWER IS BETTER (§2135, norm-2304 at 2.6735); nothing here installs.**

## Predictions

```
BARS  = {n_usable: 6, beat_current: 2.0, shared_design: 5, target_drift: .10, floor: .5}
NULLS = {n_usable_le: 3, beat_current_le: 1.0, shared_design_le: 2}
USABLE = .50 (from §2852)
```

**pred_a_a_design_clears_the_bar_widely** — at least **6** of the live behaviours have some design with native margin ≥ .50.
*Worked example:* three clear it today (§2852). If the copy reading simply needs stronger cueing, v2 or v3 should carry most of the
rest and this reads 6–8. If it reads ≤ 3, the failure is not cueing — the model does not do verbatim copying in these surfaces at all —
and the C family needs replacing with a different control concept rather than a stronger version of the same one. Count over live
behaviours. Null: ≤ 3.

**pred_b_the_best_design_beats_the_current_one** — median over behaviours of |best design's margin| / |v1_current's margin| ≥ 2.0.
*Worked example:* the current margins are .09–.20 on the failing behaviours, so a design that reaches .5–1.5 gives ratios of 3–10.
A ratio near 1 means the designs are interchangeable and the defect is not about form. Both operands are margins in the same units;
the denominator is floored at 1e-6 and the raw margins are reported alongside so a tiny denominator is visible rather than inflating
the ratio. Null: ≤ 1.0.

**pred_c_one_design_wins_across_behaviours** — the same design is best on at least **5** of the live behaviours.
*Worked example:* a shared winner is a protocol fix; a per-behaviour patchwork is task-fitting dressed as one, and would have to be
declared as such. If v2 wins on the list-shaped tasks and v3 on the run-shaped ones, this fails at 4/4 and the honest outcome is two
designs chosen by surface class, recorded rather than hidden. Null: ≤ 2.

**pred_d_the_target_family_is_untouched** — max over behaviours of |this rung's A1 native margin − §2852's| / §2852's ≤ .10.
*Worked example:* the redesign touches only the C family, so the A1 margins must reproduce §2852's to within sampling noise on the
same frozen SELECT rows. A drift above .10 means something other than the control changed and no comparison in this rung is readable.
Instrument check.

**pred_e_the_control_answer_is_not_the_successor** — on every live behaviour, ZERO control rows have an answer equal to the A1 answer
of the SAME generated situation, and ZERO have an answer absent from their own prompt.
*Worked example:* a copy control must be copyable (its answer appears in its prompt) and must not coincide with the successor answer,
or it is not controlling for anything. The bank groups families by `group_id`, so both are exact checks rather than task-specific
heuristics. **This predicate replaced a first draft whose name claimed to test the successor clash while its body only checked that
some rows were scorable — caught and fixed before this document existed.**

## Stated null

No design clears the bar widely (≤ 3), the best is no better than the current one, and no shared design wins. That would say verbatim
copying is not a behaviour this model performs in these surfaces, the C family cannot be repaired by re-cueing, and the battery's
selectivity stage should be redefined around a different second control — which is a larger protocol amendment and would be registered
as one.

## Price

≤ 8 behaviours × 3 designs × 24 SELECT control rows + 8 × 24 A1 rows, batched by token length, single forward per batch.
Literal budget: ≤ 400 GPU forwards, 0 backwards, **0 fitted parameters**, < 60 GPU-seconds. Uses `ops/fastload.py`.
**Per §2853–§2856: every figure in the resulting ledger section, including the price, will be read from this receipt in the same
command the section is written from, and none from a smoke run.**

## What this does NOT claim

Native margin only — this rung does not re-run the selectivity stage with the new controls, which is a separate protocol amendment
once a design is chosen. Designs are re-shapings of the existing C rows, so a behaviour whose C concept is wrong in kind cannot be
rescued here. attn8-writer capable behaviours only. Nothing installs; no L2. Does not satisfy Codex's four-phase integration contract;
updates no circuit record.
