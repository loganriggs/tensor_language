# CIRCUIT BATTERY — CLASS MASS LOCALISATION (preregistration)

Registered 2026-09-04 05:12Z (box clock, read immediately before this line was written). Claude, LANE 1 CUDA.
Rung `circuit_battery_class_mass_localisation`. Script: `ops/circuit_battery_class_mass_localisation.py`.
Input receipt: `circuit_battery_v2_results.json` (§2817, sha 5924b2549d…). IMMUTABLE.

## Object

§2828 refuted the type-gate hypothesis for the read: removing mlp10/mlp11's entire read of attention 8's write costs .0148 nats of
candidate-class mass while moving the within-class margin by 2.6 units. The read decides WHICH member of the answer class, not whether
the class applies. So something else maintains the type — "a list label goes here", "a weekday goes here" — and it has never been
measured in this campaign, because every rung so far has scored the MARGIN, which is by construction a within-class quantity.

This rung localises the type gate exactly the way the battery localises writers, but with the class-mass metric: each of the 36
components has its final-position write ablated outright and the loss of `log Σ_{v∈candidates} p(v)` is measured on OOD rows. The
margin damage of the same arms is recorded alongside, so the class map and the member map can be compared component by component. The
central question is whether they are the SAME components.

Fixed before the run: all 36 components, behaviours = §2817's capable attn8-writer set, split OOD, family A1, zero fitted parameters.
Sign convention: d_c = logmass_NATIVE − logmass_arm in NATS, POSITIVE = the arm REMOVES class mass; d_m = m_NATIVE − m_arm, POSITIVE =
the arm HURTS the answer.

## Predictions

```
BARS  = {top3_share: .60, overlap: .50, early_layer: 8, shared_tasks: 4, floor: .05}
NULLS = {top3_share_le: .30, overlap_ge: .80, shared_tasks_le: 1}
```

**pred_a_class_mass_is_localised** — median over behaviours of `(top-3 components' class damage) / max(Σ positive class damage, .05)`
≥ .60. *Worked example:* if a few components decide that a class member goes here, three of 36 hold .6–.9 of the positive class damage;
if the type is maintained by the whole stack, three of 36 hold ~.15–.25. Sum of positive damages in the denominator so a component whose
removal ADDS class mass cannot inflate the share; floored at .05 nats. Null: ≤ .30.

**pred_b_class_gate_is_not_the_member_selector** — median over behaviours of the fraction of the top-3 class components that are also
top-3 margin components ≤ .50 (i.e. at most one of three shared). *Worked example:* §2828 already showed the two roles come apart
WITHIN a block; if they also come apart across components, the overlap is 0/3 or 1/3. If the same components lead both maps (3/3), the
type and the member are one mechanism and §2828's within-block split is a peculiarity of that block rather than a division of labour.
Fraction of a fixed count of three. Null: ≥ .80.

**pred_c_class_gate_is_early** — on at least 4 of the ≤7 behaviours the component with the largest class damage sits at layer ≤ 8.
*Worked example:* a type decision that later blocks refine should be made no later than the write it conditions, so the hypothesis is
layers 0–8; if the class is decided late, the leaders sit at 12–17 and this reads 0–2. Count over behaviours.

**pred_d_class_gate_is_shared_across_behaviours** — the same component leads the class map on at least 4 of the ≤7 behaviours.
*Worked example:* the answer classes differ across behaviours (digits, weekday names, roman numerals), so a SHARED leader would be a
strong claim — a general "emit a member of the salient class" component — while a per-behaviour leader would say the type gate is
class-specific. Either outcome is informative; the bar is set where a shared gate is the claim. Null: ≤ 1.

**pred_e_attention8_is_not_the_class_gate** — median over behaviours of attention 8's RANK in the class map is > 3.
*Worked example:* attention 8 is the writer for all seven of these behaviours and leads the margin map; if it also led the class map,
the whole circuit would be one component doing both jobs and §2828's division of labour would be internal to it. The hypothesis is that
it does not — its write is a context-blind copy of the last item (§2808, §2820), which is a member, not a type. Rank is an integer in
[1, 36].

## Stated null

The class map is diffuse (top-3 ≤ .30), coincides with the margin map (overlap ≥ .80), and has no shared leader. That would say the
type is not localised at all and §2828's division of labour does not generalise beyond the one block it was measured in.

## Price

≤ 7 behaviours × 36 component ablations × 1 forward per length-batch of 16 OOD rows, plus one native forward per batch.
Literal budget: ≤ 1,600 GPU forwards, 0 backwards, **0 fitted parameters**, < 2 GPU-minutes.

## What this does NOT claim

The candidate class is the bank's own answer vocabulary, so class mass is defined relative to a design choice; a different vocabulary
would give different nats (this is stated in §2828 too). Whole-component ablation only — no per-head or per-edge decomposition of the
class gate. Target family only, so nothing here is a selectivity claim. Does not satisfy Codex's four-phase integration contract;
updates no circuit record.
