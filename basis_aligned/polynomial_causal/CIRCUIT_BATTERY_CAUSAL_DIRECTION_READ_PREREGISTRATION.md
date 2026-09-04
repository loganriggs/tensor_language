# CIRCUIT BATTERY — CAUSAL DIRECTION READ (preregistration)

Registered 2026-09-04 05:02Z (box clock, read immediately before this line was written). Claude, LANE 1 CUDA.
Rung `circuit_battery_causal_direction_read`. Script: `ops/circuit_battery_causal_direction_read.py`.
Input receipts: `circuit_battery_v2_results.json` (§2817, sha 5924b2549d…) and
`circuit_battery_removal_effect_geometry_results.json` (§2825). IMMUTABLE: any change gets a new document, not an edit.

## The move

Four rungs have now failed to localise the read of attention 8's write below the block, and §2825 diagnosed why: every ranking used was
a SIZE ranking. §2822 ranked hidden units by magnitude, §2823 by exact lens magnitude, §2824/§2825 by singular energy — and the
in-sample rank-4 subspace holds .700 of the removal effect's energy while delivering .139 of its damage. Energy is not causality in
this model.

So rank by causality, with NOTHING FITTED. For each row the axis that matters for the measured quantity is the one the margin is
defined on: `u = W_U[answer] − W_U[best competing candidate]`, read straight off the unembedding and normalised — the competitor is the
top non-answer candidate in that row's own native logits. The arm removes ONLY the component of the removal effect along u. If the
block's read acts on the answer-versus-competitor axis, this rank-1 parameter-free direction should beat §2825's fitted rank-4 energy
subspace outright.

Fixed before the run: writer attn8, readers mlp10 and mlp11, behaviours = §2817's capable attn8-writer set, split OOD, random-direction
seed 2825, comparison value = §2825's per-cell in-sample rank-4 share (already published, not re-fitted here). Zero fitted parameters.
Sign convention: d_m = m_NATIVE − m_arm, POSITIVE = the arm HURTS; ratio = max(|d_P|, |d_C|) / max(d_A1, .5), LOWER IS MORE SPECIFIC.
Admissibility gate (§2821): the causal arm counts as specific only if its A1 damage is ≥ .25 × the block's.

## Predictions

```
BARS  = {causal_share: .50, beat_energy: .20, random_share: .05, causal_energy: .25, specific_gain: .10, admit_block: .25, floor: .5}
NULLS = {causal_share_le: .15, beat_energy_le: 0.0, random_share_ge: .30, causal_energy_ge: .60}
```

**pred_a_causal_direction_carries_the_damage** — median over (behaviour, reader) cells of
`(causal rank-1 arm's A1 damage) / max(block A1 damage, .5)` ≥ .50. *Worked example:* if the block's read moves the answer against its
competitor and little else that matters, one direction of 1152 carries .5–.9; if the read damages the margin through many axes (for
instance by changing which tokens are candidates at all), .1–.3. Damages in margin units over a floored denominator. Null: ≤ .15.

**pred_b_causal_beats_fitted_energy** — median over cells of `(causal rank-1 share) − (§2825's in-sample fitted rank-4 share)` ≥ .20.
*Worked example:* §2825's fitted rank-4 delivered .139 with four fitted directions per cell; if one unfitted causal direction reaches
.6, the difference is ~.46. If it reads ≤ 0, size-ranking was not the problem and the campaign's four negative sub-block results are
about the model rather than about my choice of statistic — which would itself be worth knowing. A DIFFERENCE of two shares, both
floored on the same denominator. Null: ≤ 0.

**pred_c_random_direction_is_inert** — median over cells of the same share for a seeded RANDOM unit direction ≤ .05.
*Worked example:* a random direction of 1152 should capture ≈ 1/1152 of any effect and essentially none of the damage, ~.00–.02. This
control is what makes pred_a non-trivial: without it, "removing one direction hurts" could just mean the block is fragile.

**pred_d_the_causal_direction_is_low_energy** — median over cells of the fraction of the removal effect's squared norm lying along u is
≤ .25. *Worked example:* this is §2825's finding stated positively — if the causal direction were also the big one, an SVD would have
found it and §2824 would have succeeded; so the expectation is .01–.20, with a random direction at ≈ 1/1152 ≈ .0009 for scale. A
fraction of non-negative energies. Null: ≥ .60.

**pred_e_causal_arm_is_more_specific** — median over ADMISSIBLE cells of `(block ratio) − (causal arm ratio)` ≥ .10.
*Worked example:* the answer-versus-competitor axis is defined by the TARGET family's answer, so restricting the intervention to it
should spare the controls somewhat, giving .1–.4; if the same axis is what the copy control uses too, ~0. Only cells whose causal arm
does at least a quarter of its block's damage are eligible — the gate that caught inert arms in §2820, §2822, §2823 and §2824.
Null: not registered separately; pred_e's failure is reported as measured.

## Stated null

The causal direction is no better than a fitted energy subspace and no better than chance: causal share ≤ .15, no beat over §2825,
random ≥ .30, and the causal direction is high-energy after all. That would mean the read cannot be reduced to any one-dimensional
axis, and combined with §2822–§2825 would close sub-block localisation for this circuit in every way this campaign can test.

## Price

≤ 7 behaviours × 2 readers × [3 arms (all, causal, random) over A1/P/C + an energy-fraction pass] at 16 OOD rows per cell.
Literal budget: ≤ 1,400 GPU forwards, 0 backwards, **0 fitted parameters**, < 2 GPU-minutes.

## What this does NOT claim

`u` is the exact axis of the MARGIN, but the map from residual to logits passes through a final RMS norm, so `u` is the right direction
only up to that norm's Jacobian — a registered approximation, not an identity. The competitor token is chosen per row from the native
logits, which is data-dependent but not fitted. Two readers only. Does not satisfy Codex's four-phase integration contract; updates no
circuit record.
