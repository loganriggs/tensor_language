# CIRCUIT BATTERY — READER UNIT LOCALISATION, LENS-RANKED (preregistration)

Registered 2026-09-04 04:45Z (box clock, read immediately before this line was written). Claude, LANE 1 CUDA.
Rung `circuit_battery_reader_unit_lens_ranked`. Script: `ops/circuit_battery_reader_unit_lens_ranked.py`.
Input receipt: `circuit_battery_v2_results.json` (§2817, sha 5924b2549d285175c80fbf7c8fc95a8a2fa06020acc1827bc472ddea69d9ec93).
IMMUTABLE: any change gets a new document, not an edit.

## Why this is a separate document

The predecessor rung (`circuit_battery_reader_unit_localisation`, preregistration sha 8b9de76b…) ranked hidden units by the MAGNITUDE
of their change when the write is removed, `|Δh_u| · ‖Down[:,u]‖`, and its top-64 set turned out to carry ≈ 0 of the block's damage.
That run was executed and is reported as registered — the failed ranking is preserved, not overwritten. **The one thing changed here is
the ranking statistic**; every bar, gate, control and phase is identical, so the two runs are a clean A/B of the selector.

The new statistic is the unit's exact SIGNED contribution to the answer direction:
`score_u = Δh_u · (Down[:,u] · W_U[answer])`, summed with + on the target family A1 and − on the copy control C over FIT rows.
Because `mlp(u) = Down(Left(u) ⊙ Right(u)) + b` is exactly bilinear, `Δh_u · Down[:,u]` IS unit u's additive contribution to the block's
output change, and dotting it with the answer's unembedding row gives its contribution to the answer logit exactly. A magnitude ranking
cannot distinguish a unit that pushes the answer from one that pushes an irrelevant direction, or from two units that cancel; this one
can. Everything else — writer attn8, readers mlp10 and mlp11, TOPK 64 of 4,608, random seed 2821, FIT ranks / OOD scores, the
admissibility gate at .25 × the block's own A1 damage — is unchanged and was fixed before either run.

## Predictions

```
BARS  = {exact_tol: 1e-3, topk_share: .50, specific_gain: .20, jaccard: .15, random_share: .15, admit_block: .25, floor: .5}
NULLS = {topk_share_le: .20, specific_gain_le: 0.0, jaccard_le: .02, random_share_ge: .40}
```

**pred_a_unit_decomposition_is_exact** — max over cells of `|logits(all 4,608 units removed) − logits(block read removed)|` ≤ 1e-3.
*Worked example:* the predecessor measured exactly 0.0 here, so this is a re-check of the instrument after changing the ranking code,
not a new claim; a bug introduced by the edit reads O(1).

**pred_b_lens_ranked_units_carry_the_read** — median over (behaviour, reader) cells of
`(top-64 units' A1 damage) / max(block A1 damage, .5)` ≥ .50. *Worked example:* the magnitude ranking read −.0003 — its top-64 carried
nothing at all. If the read is sparse in the unit basis and the lens statistic finds it, this reads .5–.9; if the read is genuinely
DENSE (thousands of units each contributing a little), even a perfect ranking of 1.4% of units reads ~.05–.15 and this fails, which is
then a real statement about the model rather than about the selector. Null: ≤ .20.

**pred_c_the_unit_set_is_more_specific_than_its_block** — median over ADMISSIBLE cells of `(block ratio) − (top-64 ratio)` ≥ .20, where
a cell is admissible only if the top-64 set's A1 damage is ≥ .25 × its block's. *Worked example:* the predecessor had ZERO admissible
cells, so its top-64 "ratio" of .031 was the inert-arm artifact the gate exists to catch (§2820); this prediction can only be awarded on
sets that actually do something. A difference of two floored ratios. Null: ≤ 0.

**pred_d_the_lens_unit_set_is_shared** — median over readers of the median pairwise Jaccard overlap between behaviours' top-64 sets
≥ .15. *Worked example:* chance is ≈ .007 and the magnitude ranking read .008 (mlp10) and .016 (mlp11) — i.e. at chance. If the same
units serve different surface forms, .15–.5. Null: ≤ .02.

**pred_e_random_units_do_not_carry_the_read** — median over cells of `(random 64 units' A1 damage) / max(block A1 damage, .5)` ≤ .15.
*Worked example:* the predecessor measured .0006; this is the control that keeps pred_b non-trivial and it is expected to pass again.
Null: ≥ .40.

## Stated null

The read is dense in the unit basis and no ranking statistic can localise it: top-64 share ≤ .20, no admissible cell shows a
specificity gain, overlap at chance. Given the predecessor's result, this null is the LIKELY outcome and would establish something
worth stating — that below the block, bilin18's read of a write is not unit-sparse in the network's own coordinates.

## Price

Identical to the predecessor: ≤ 7 behaviours × 2 readers × [FIT ranking over A1 and C + OOD arms (block, top-64, random) over A1/P/C +
an exactness check] at 16 rows per cell. Literal budget: ≤ 3,000 GPU forwards, 0 backwards, 0 fitted parameters, < 4 GPU-minutes.

## What this does NOT claim

Unit coordinates are basis-dependent; a negative result here bounds sparsity in the NETWORK's basis only and says nothing about a
rotated one. The lens statistic is a first-order read of the answer direction and ignores how the block's output is further transformed
downstream. Does not satisfy Codex's four-phase integration contract; updates no circuit record.
