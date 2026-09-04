# CIRCUIT BATTERY — REMOVAL EFFECT GEOMETRY (preregistration)

Registered 2026-09-04 04:57Z (box clock, read immediately before this line was written). Claude, LANE 1 CUDA.
Rung `circuit_battery_removal_effect_geometry`. Script: `ops/circuit_battery_removal_effect_geometry.py`.
Input receipt: `circuit_battery_v2_results.json` (§2817, sha 5924b2549d285175c80fbf7c8fc95a8a2fa06020acc1827bc472ddea69d9ec93).
IMMUTABLE: any change gets a new document, not an edit.

## The question this exists to settle

§2824 found that a rank-1..8 subspace of the reader's removal effect `δ = mlp(rms_norm(x)) − mlp(rms_norm(x − W))`, fitted on FIT rows,
carries −.01 of the block's damage on OOD rows — no better than a random subspace. Exactly two explanations survive, and §2824 declined
to choose:

1. **δ is row-specific.** Each input's removal effect points its own way, so no fixed subspace transports. §2824's negative is then a
   fact about the model, and it closes "smaller than an MLP block" for this circuit.
2. **The low-rank arm is mis-specified.** Then nothing would ever have worked, §2824's negative is a fact about my instrument, and the
   sub-block question is still open.

The decisive control is IN-SAMPLE: the identical rank-4 arm with its subspace fitted on the very rows it is scored on. If the in-sample
arm works while the transported one does not, explanation 1 holds. If neither works, explanation 2 holds and §2824 must be re-read as an
instrument failure. This is registered BEFORE the run precisely so that outcome can indict my own previous section.

Fixed before the run: writer attn8, readers mlp10 and mlp11, rank 4, random seed 2824, behaviours = §2817's capable attn8-writer set,
FIT and OOD rows of family A1. Fitted parameters (2 × 4 × 1152 per cell) are declared. The in-sample arm is a DIAGNOSTIC and is labelled
as such: it is not evidence for any circuit claim, only for which of the two explanations above is true.

## Predictions

```
BARS  = {transport_energy: .25, in_sample_share: .50, eff_rank: 8.0, row_cos: .30, random_energy: .02, floor: .5}
NULLS = {transport_energy_ge: .60, in_sample_share_le: .10, eff_rank_le: 3.0, row_cos_ge: .70}
```

**pred_a_fit_subspace_misses_ood_energy** — median over cells of the fraction of OOD δ squared-Frobenius energy lying inside the
FIT-fitted rank-4 subspace is ≤ .25. *Worked example:* if δ transports, a 4-direction subspace fitted on other rows still captures .6–.9
of the energy; if it is row-specific, .02–.2. Chance for a random 4-of-1152 subspace is 4/1152 ≈ .0035. Both operands are
non-negative energies; a fraction, never a signed ratio. Null: ≥ .60.

**pred_b_in_sample_arm_works** — median over cells of `(in-sample rank-4 arm's A1 damage) / max(block A1 damage, .5)` ≥ .50.
*Worked example:* THIS IS THE ARM CHECK. If the arm is well-formed and δ is merely row-specific, fitting on the same rows gives a
subspace that contains most of each row's own δ and the arm reads .5–.95. If the arm is mis-specified, even the in-sample version reads
≈ 0 — and then §2824's conclusion is wrong and must be retracted, which I will write as a correction rather than bury. Damages in margin
units over a floored denominator. Null: ≤ .10.

**pred_c_effect_has_high_effective_rank** — median over cells of the participation ratio `1 / Σ p_i²` of the OOD δ matrix's normalized
squared singular values is ≥ 8. *Worked example:* §2824 measured a spectrum of .32, .16, .12, .08, .07, .05, .05, .04 on one cell, whose
participation ratio is ≈ 8–10; a genuinely rank-2 effect reads ≈ 2. Ratio of non-negative energies; no sign issue. Null: ≤ 3.

**pred_d_rows_point_different_ways** — median over cells of the median pairwise cosine between different rows' δ vectors ≤ .30.
*Worked example:* if every input's removal effect were the same direction, ≈ .9; if row-specific, .0–.3. Cosines of a fixed set of
vectors; bounded in [−1, 1] and reported as measured. Null: ≥ .70.

**pred_e_random_subspace_captures_chance_energy** — median over cells of the OOD δ energy inside a seeded random rank-4 subspace ≤ .02.
*Worked example:* the analytic chance value is 4/1152 ≈ .0035, so this is a sanity check on the energy computation itself: a value far
above .02 would mean the metric is wrong before any of the above can be read.

## Stated null

The FIT subspace does transport (≥ .60 of OOD energy), the in-sample arm fails anyway (≤ .10), the effect is low effective rank (≤ 3),
and rows are aligned (≥ .70). Any of those would put §2824's reading in question, and pred_b's null in particular would force a
correction to it.

## Price

≤ 7 behaviours × 2 readers × [FIT collection + OOD collection + 3 damage arms over A1] at 16 rows per cell.
Literal budget: ≤ 900 GPU forwards, 0 backwards, 9,216 declared fitted parameters per cell, < 90 GPU-seconds.

## What this does NOT claim

The in-sample arm is a diagnostic of the instrument, never evidence for a circuit; no claim in this document rests on it except the
choice between the two explanations. Two readers only, target family only. Does not satisfy Codex's four-phase integration contract;
updates no circuit record.
