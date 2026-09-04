# CIRCUIT BATTERY — CAUSAL BASIS RANK (preregistration)

Registered 2026-09-04 05:06Z (box clock, read immediately before this line was written). Claude, LANE 1 CUDA.
Rung `circuit_battery_causal_basis_rank`. Script: `ops/circuit_battery_causal_basis_rank.py`.
Input receipt: `circuit_battery_v2_results.json` (§2817, sha 5924b2549d…); compares against §2826's published per-cell numbers.
IMMUTABLE: any change gets a new document, not an edit.

## Object

§2826 found the first positive sub-block result: one unfitted direction `u = W_U[answer] − W_U[best competing candidate]` carries .199
of the reader block's damage and is 2.4× more task-specific than the block, while holding .0021 of the removal effect's energy. It is
partial — four fifths of the block's effect on the margin travels elsewhere. The margin is a comparison against ONE competitor, but the
model chooses among many candidates, so the natural question is whether the remainder is also causally structured along further
answer-versus-competitor axes. This rung spans `u_i = W_U[answer] − W_U[competitor_i]` for each row's top k competitors (k = 1, 2, 4, 8),
orthonormalises per row, and removes only the component of the removal effect inside that span. Still ZERO fitted parameters: every
direction comes from the unembedding and that row's own native logits.

Fixed before the run: writer attn8, readers mlp10 and mlp11, behaviours = §2817's capable attn8-writer set, split OOD, k ∈ {1,2,4,8},
random-basis seed 2826, admissibility gate at .25 × the block's A1 damage (§2821). Sign convention: d_m = m_NATIVE − m_arm, POSITIVE =
the arm HURTS; ratio = max(|d_P|, |d_C|) / max(d_A1, .5), LOWER IS MORE SPECIFIC.

## Predictions

```
BARS  = {k4_share: .40, growth: .10, random_share: .05, specific_gain: .10, basis_energy: .05, admit_block: .25, floor: .5}
NULLS = {k4_share_le: .20, growth_le: 0.0, random_share_ge: .30, basis_energy_ge: .30}
```

**pred_a_four_causal_directions_carry_more** — median over (behaviour, reader) cells of the k=4 causal basis's share of the block's A1
damage ≥ .40. *Worked example:* §2826 measured .199 at k=1; if the remainder is spread over further competitor axes, four of them reach
.4–.6; if the k=1 axis was the only causally aligned direction, k=4 stays near .2 and this fails. Damages in margin units over a
floored denominator. Null: ≤ .20 (i.e. no better than k=1).

**pred_b_share_grows_with_k** — median over cells of `share(k=8) − share(k=1)` ≥ .10. *Worked example:* if further competitor axes carry
effect, the curve rises .2 → .3 → .45 → .55 and this reads ~.35; if they do not, the extra directions add noise and it reads ~0 or
slightly negative. A DIFFERENCE of two shares on the same floored denominator, not a ratio. Null: ≤ 0.

**pred_c_random_basis_is_inert** — median over cells of a seeded random rank-4 basis's share ≤ .05. *Worked example:* §2825 measured a
random rank-4 subspace at .0005 of the damage and §2826 a random direction at .0006; this control is what keeps pred_a from being
"removing any four directions hurts". Null: ≥ .30.

**pred_e_the_causal_basis_stays_low_energy** — median over cells of the fraction of the removal effect's squared norm inside the k=4
causal basis ≤ .05. *Worked example:* §2826 measured .0021 for k=1 against .00087 for a random direction; four such axes should hold
.004–.02, so this asks whether the causal basis remains a low-energy object as it widens — the property that made every energy-ranked
method in §2822–§2825 miss it. If instead it reads ≥ .30, the wider basis has started to capture the generic high-energy remainder and
pred_a's share would be uninformative. Null: ≥ .30.

**pred_d_specificity_survives_the_wider_basis** — median over ADMISSIBLE cells of `(block ratio) − (k=4 arm ratio)` ≥ .10.
*Worked example:* §2826 measured a gain of .169 at k=1 on four admissible cells. Widening the basis adds directions defined by
competitors the target answer beats, which are still target-defined, so the gain should persist at .1–.4; if the gain collapses to ~0,
the extra axes are generic and only the single best-competitor axis is task-specific — a sharper and more useful result than pred_a
passing. Difference of two floored ratios; only cells whose k=4 arm does ≥ .25 of its block's damage are eligible.

## Stated null

The k=1 axis of §2826 is the only causally aligned direction: k=4 ≤ .20, no growth with k, and the specificity gain does not survive.
Combined with §2822–§2825 that would say the read's causal structure is exactly one dimension wide and everything else is generic.

## Price

≤ 7 behaviours × 2 readers × [1 block arm + 4 causal arms on A1 + 2 causal arms on the controls at k=4 + 1 random arm] × 2 forwards per
length-batch of 16 OOD rows. Literal budget: ≤ 2,600 GPU forwards, 0 backwards, **0 fitted parameters**, < 3 GPU-minutes.

## What this does NOT claim

The competitor list is taken from each row's native logits, which is data-dependent but not fitted. As in §2826, `u_i` is the exact axis
of a logit difference only up to the final RMS norm's Jacobian. Two readers, target family plus the two controls at k=4 only. Does not
satisfy Codex's four-phase integration contract; updates no circuit record.
