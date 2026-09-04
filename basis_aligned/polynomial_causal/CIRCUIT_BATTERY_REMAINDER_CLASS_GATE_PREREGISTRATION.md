# CIRCUIT BATTERY — REMAINDER CLASS GATE (preregistration)

Registered 2026-09-04 05:09Z (box clock, read immediately before this line was written). Claude, LANE 1 CUDA.
Rung `circuit_battery_remainder_class_gate`. Script: `ops/circuit_battery_remainder_class_gate.py`.
Input receipt: `circuit_battery_v2_results.json` (§2817, sha 5924b2549d…). IMMUTABLE.

## The question

§2826 and §2827 split the read of attention 8's write by mlp10/mlp11 into a rank-1 unfitted low-energy task-specific component along
`u = W_U[answer] − W_U[competitor]` (a fifth of the block's margin damage, 2.4× its specificity) and a generic remainder of four
fifths. Five decompositions have now failed to find structure in that remainder — but all five asked WHERE it lives. This asks what it
DOES, using a measurement the campaign has not used: the **candidate-class mass** `log Σ_{v ∈ candidates} p(v)`, i.e. how much
probability the model puts on the task's answer class at all, independent of which member it picks.

**Hypothesis:** the remainder maintains the TYPE ("a list label goes here") while the rank-1 axis selects the MEMBER ("the next one").
If so, the remainder's damage is class-mass damage and the causal axis's is not — and the mechanism of this circuit becomes sayable in
one sentence rather than as a list of negative results.

Fixed before the run: writer attn8, readers mlp10 and mlp11, behaviours = §2817's capable attn8-writer set, split OOD, family A1,
random seed 2827. Arms: ALL (whole removal effect), CAUSAL (component along u), REMAINDER (the orthogonal complement of u within the
effect), RANDOM (a seeded random unit direction). Zero fitted parameters. Sign convention: margin damage d_m = m_NATIVE − m_arm,
POSITIVE = the arm HURTS; class-mass damage d_c = logmass_NATIVE − logmass_arm in NATS, POSITIVE = the arm REMOVES class mass.

## Predictions

```
BARS  = {class_ratio: 3.0, causal_class_nats: .15, within_gain: .10, random_nats: .05, additivity: .20, floor: .05}
NULLS = {class_ratio_le: 1.0, causal_class_ge: .50, within_gain_le: 0.0, random_ge: .30}
```

**pred_a_remainder_carries_the_class_mass** — median over (behaviour, reader) cells of
`|d_c(REMAINDER)| / max(|d_c(CAUSAL)|, .05 nats)` ≥ 3.0. *Worked example:* if the remainder is a type gate, removing it collapses the
class mass by a few tenths of a nat while the causal axis barely touches it, giving 5–30; if both parts damage the class equally, ~1.
Denominator floored at .05 nats so it cannot pass through zero, and both operands are absolute values, so no sign flip. Null: ≤ 1.0.

**pred_b_causal_axis_spares_the_class** — median over cells of `|d_c(CAUSAL)|` ≤ .15 nats. *Worked example:* the axis
W_U[answer] − W_U[competitor] is a DIFFERENCE of two members of the same class, so removing the effect's component along it should move
mass between members and not out of the class: .00–.10 nats. If it reads ≥ .50 the axis is not within-class and pred_c's framing is
wrong. Absolute nats, no ratio. Null: ≥ .50.

**pred_c_causal_axis_is_within_class** — median over cells of
`d_m(CAUSAL)/max(|d_c(CAUSAL)|, .05) − d_m(REMAINDER)/max(|d_c(REMAINDER)|, .05)` ≥ .10. *Worked example:* this is margin damage per
nat of class damage — the causal axis should buy margin damage cheaply in class-mass terms and the remainder expensively, so a
positive difference of ~.5–4. If both parts trade margin for class mass at the same rate, ~0, and the "type versus member" story is
wrong. A DIFFERENCE of two floored ratios, not a ratio of ratios; both numerators may be negative and are kept with sign. Null: ≤ 0.

**pred_d_random_direction_does_neither** — median over cells of `|d_c(RANDOM)|` ≤ .05 nats. *Worked example:* §2826 measured a random
direction at .0006 of the margin damage; it should likewise be inert on class mass, ~.00–.02. This keeps pred_b from being satisfied by
an arm that simply does nothing. Null: ≥ .30.

**pred_e_the_two_parts_add_up** — median over cells of `|d_c(CAUSAL) + d_c(REMAINDER) − d_c(ALL)|` ≤ .20 nats.
*Worked example:* CAUSAL and REMAINDER are exact orthogonal parts of the same effect vector, so if the network's response to removing
them were linear this would be 0; it is not linear, so the gap measures how much the decomposition interacts, and .0–.15 nats would say
the split is close to additive at this scale. A large gap (≥ .5) means the two parts cannot be interpreted separately at all, which
would qualify every reading above and is registered here rather than discovered later.

## Stated null

The remainder is not a type gate: class ratio ≤ 1, the causal axis costs ≥ .5 nats of class mass, no within-class advantage, and the
random control is not inert. That would leave the generic four fifths uncharacterised by function as well as by location.

## Price

≤ 7 behaviours × 2 readers × 5 forwards per length-batch of 16 OOD rows.
Literal budget: ≤ 900 GPU forwards, 0 backwards, **0 fitted parameters**, < 90 GPU-seconds.

## What this does NOT claim

Class mass is defined by the task's own answer vocabulary, which is a design choice of the bank, not a property of the model; a
different candidate set would give different nats. Target family only. Two readers. Does not satisfy Codex's four-phase integration
contract; updates no circuit record.
