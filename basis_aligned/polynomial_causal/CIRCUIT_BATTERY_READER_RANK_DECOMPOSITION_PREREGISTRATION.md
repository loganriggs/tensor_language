# CIRCUIT BATTERY — READER RANK DECOMPOSITION (preregistration)

Registered 2026-09-04 04:54Z (box clock, read immediately before this line was written). Claude, LANE 1 CUDA.
Rung `circuit_battery_reader_rank_decomposition`. Script: `ops/circuit_battery_reader_rank_decomposition.py`.
Input receipt: `circuit_battery_v2_results.json` (§2817, sha 5924b2549d285175c80fbf7c8fc95a8a2fa06020acc1827bc472ddea69d9ec93).
IMMUTABLE: any change gets a new document, not an edit.

## Object

§2822 and §2823 established, with two independent selectors and a random control, that the causal read of attention 8's write is DENSE
in the hidden-unit basis: the top 1.4% of units by the exactly correct statistic are indistinguishable from a random 1.4%. Unit
coordinates are basis-dependent, so the honest follow-up asks the same question basis-free. For reader block ℓ the removal effect is
`δ(row) = mlp(rms_norm(x)) − mlp(rms_norm(x − W))` in R^1152. This rung fits the top singular directions of δ over FIT rows and then
intervenes with only the rank-r part — the arm subtracts `P_r δ` instead of δ — scoring on OOD.

**This rung has FITTED PARAMETERS, unlike every previous battery rung, and they are declared rather than hidden:** r × 1152 per
(behaviour, reader), fitted on FIT rows only, with OOD never opened for the fit. Fixed before the run: writer attn8, readers mlp10 and
mlp11, ranks 1/2/4/8, random-basis seed 2823, behaviours = §2817's capable attn8-writer set. Admissibility gate (§2821 correction): a
rank-4 arm counts as specific only if its own A1 damage is ≥ .25 × its BLOCK's A1 damage. Sign convention: d_m = m_NATIVE − m_arm,
POSITIVE = the arm HURTS; ratio = max(|d_P|, |d_C|) / max(d_A1, .5), LOWER IS MORE SPECIFIC.

## Predictions

```
BARS  = {rank1: .50, rank4: .80, overlap: .50, specific_gain: .20, random_share: .15, admit_block: .25, floor: .5}
NULLS = {rank1_le: .20, rank4_le: .40, overlap_le: .10, specific_gain_le: 0.0, random_share_ge: .40}
```

**pred_a_rank_one_carries_half** — median over (behaviour, reader) cells of `(rank-1 arm's A1 damage) / max(block A1 damage, .5)` ≥ .50.
*Worked example:* if the block's read of the write is essentially one direction (a "the answer is the successor" direction), rank 1
reads .5–.9; if the read is spread over many directions as it is over many units, rank 1 of 1152 reads ~.05–.2. Damages in margin units
over a floored denominator. Null: ≤ .20.

**pred_b_rank_four_carries_most** — the same share at rank 4 is ≥ .80. *Worked example:* a genuinely low-rank read saturates fast, so
if pred_a holds at ~.6 then rank 4 should reach .85–.95; a dense read climbs roughly linearly in r and reads ~.1–.3 at rank 4. This is
the prediction that distinguishes "low-rank" from "one lucky direction". Null: ≤ .40.

**pred_c_the_subspace_is_shared_across_behaviours** — median over readers of the median, over behaviour pairs, of the mean squared
principal cosine between their fitted rank-4 subspaces ≥ .50. *Worked example:* two random 4-dimensional subspaces of R^1152 have mean
squared principal cosine ≈ 4/1152 ≈ .003; if the same directions serve every surface form, .5–.9. §2818 found component re-use but only
weakly shared INTERACTION structure (.293), so this tests whether the shared object is the subspace rather than the pattern. Null: ≤ .10.

**pred_d_low_rank_arm_is_more_specific** — median over ADMISSIBLE cells of `(block ratio) − (rank-4 arm ratio)` ≥ .20.
*Worked example:* if the task-specific part of the read lives in a few directions, restricting the intervention to them should damage
the controls less relative to the target, giving ~.2–.5; if specificity is a property of the whole block, ~0. Only cells whose rank-4
arm does at least a quarter of its block's damage are eligible — §2820 and §2822 both produced "perfectly specific" arms that were
simply inert, and this gate is why §2822's .031 won nothing. A difference of two floored ratios. Null: ≤ 0.

**pred_e_random_subspace_does_not_carry_the_read** — median over cells of `(random rank-4 subspace's A1 damage) / max(block A1 damage, .5)`
≤ .15. *Worked example:* a random 4-of-1152 subspace should capture ≈ .003 of a dense effect and ~0 of a low-rank one; this is the
control that makes pred_a and pred_b interpretable, exactly as the random unit set did in §2822/§2823 — where it read .0006 and the
ranked sets read no better. Seeded (2823), orthonormalised by QR. Null: ≥ .40.

## Stated null

The read is dense in every basis: rank 1 ≤ .20, rank 4 ≤ .40, subspaces at chance overlap, no specificity gain. Combined with §2823
that would say the block's read of a write is irreducibly 1152-dimensional at the granularity this campaign can measure, and the MLP
block is the smallest honest unit of this circuit — a conclusion the 03:21Z directive's "smaller than an MLP block" would then have
received a definite negative answer to, in both a coordinate and a subspace sense.

## Price

≤ 7 behaviours × 2 readers × [FIT SVD pass + OOD arms (block, 4 ranks, random) over A1/P/C] at 16 rows per cell.
Literal budget: ≤ 3,500 GPU forwards, 0 backwards, **fitted parameters r × 1152 per cell (declared, ≈ 100 k total)**, < 5 GPU-minutes.

## What this does NOT claim

The subspace is fitted per behaviour and per reader on that behaviour's own FIT rows, so pred_a, pred_b and pred_d are two-phase
results, not parameter-free ones; only pred_c (overlap of independently fitted subspaces) and pred_e (random control) are free of that.
Two readers only. Does not satisfy Codex's four-phase integration contract; updates no circuit record.
