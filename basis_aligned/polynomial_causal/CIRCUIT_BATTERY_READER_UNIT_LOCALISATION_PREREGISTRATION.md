# CIRCUIT BATTERY — READER UNIT LOCALISATION (preregistration)

Registered 2026-09-04 04:42Z (box clock, read immediately before this line was written). Claude, LANE 1 CUDA.
Rung `circuit_battery_reader_unit_localisation`. Script: `ops/circuit_battery_reader_unit_localisation.py`.
Input receipt: `circuit_battery_v2_results.json` (§2817, sha 5924b2549d285175c80fbf7c8fc95a8a2fa06020acc1827bc472ddea69d9ec93).
IMMUTABLE: any change gets a new document, not an edit.

## Object

bilin18's block MLP is `Bilinear` with `gated=False`: `mlp(u) = Down(Left(u) ⊙ Right(u)) + b`. The elementwise product means the reader's
response to removing a write decomposes ADDITIVELY over its 4,608 hidden units — removing unit u's read of W is taking that one
coordinate of the hidden vector from the removed-input forward and every other coordinate from the native one, then applying `Down`.
Removing all 4,608 must equal removing the write from the block outright. This is the decomposition strictly finer than an MLP block
that the campaign has wanted since the 03:21Z directive, and it is exact rather than approximate.

Fixed before the run: writer attn8; readers mlp10 and mlp11 (the specificity peak in §2819/§2821); TOPK = 64 units of 4,608 (1.4%);
random-set seed 2821; behaviours = §2817's capable attn8-writer set. **Phases:** FIT ranks units by `Σ_families ±(|Δh_u| · ‖Down[:,u]‖)`
with + on the target family A1 and − on the copy control C — a selection, hence FIT only. OOD scores the resulting set. Sign
convention: d_m = m_NATIVE − m_arm, POSITIVE = the arm HURTS that family's answer; ratio = max(|d_P|, |d_C|) / max(d_A1, .5), LOWER IS
MORE SPECIFIC. **Admissibility gate (the §2821 correction): a unit set counts as specific only if its own A1 damage is ≥ .25 × its
BLOCK's A1 damage** — gated on the block, not on READS, because §2821 showed a READS-relative gate admits almost nothing.

## Predictions

```
BARS  = {exact_tol: 1e-3, topk_share: .50, specific_gain: .20, jaccard: .15, random_share: .15, admit_block: .25, floor: .5}
NULLS = {topk_share_le: .20, specific_gain_le: 0.0, jaccard_le: .02, random_share_ge: .40}
```

**pred_a_unit_decomposition_is_exact** — max over cells of `|logits(all 4,608 units removed) − logits(block read removed)|` ≤ 1e-3.
*Worked example:* the two are the same computation, so the hypothesis reads fp32 round-off ~1e-5. The bar is 1e-3 rather than §2820's
1e-4, deliberately: §2820's pred_a failed at 1.83e-4 purely on fp32 accumulation over many masked projections, and this arm accumulates
over 4,608 coordinates, so the bar is set from the dtype instead of copied. An indexing or ordering bug reads O(1).

**pred_b_a_few_units_carry_the_read** — median over (behaviour, reader) cells of `(top-64 units' A1 damage) / max(block A1 damage, .5)`
≥ .50. *Worked example:* if the read of the write is carried by a sparse set of units, 1.4% of them reach half the block's damage and
this reads .5–.9; if the read is dense, 64 of 4,608 units read ~.014 of it. Both operands are damages in margin units with a floored
denominator. Null: ≤ .20.

**pred_c_the_unit_set_is_more_specific_than_its_block** — median over ADMISSIBLE cells of `(block ratio) − (top-64 ratio)` ≥ .20.
*Worked example:* §2819 measured mlp10 at .24–.94 and mlp11 at .14–.59; if specificity concentrates further within the block, the unit
set sits ~.2–.5 below its block's ratio. If the block's specificity is a property of the whole block rather than of a subset, ~0. A
DIFFERENCE of two floored ratios, not a ratio of ratios, and only admissible cells (top-64 A1 damage ≥ .25 × block's) are eligible —
an inert unit set cannot win, which is exactly the §2820 failure this gate exists to prevent. Null: ≤ 0.

**pred_d_the_unit_set_is_shared** — median over readers of the median pairwise Jaccard overlap between behaviours' top-64 sets ≥ .15.
*Worked example:* two independent 64-of-4,608 draws overlap at Jaccard ≈ .007, so .15 is twenty times chance and indicates the same
units serve different surface forms; a value near .01 says each behaviour recruits its own units inside a shared block, which would be
a genuinely different picture from §2818's component-level re-use. Null: ≤ .02.

**pred_e_random_units_do_not_carry_the_read** — median over cells of `(random 64 units' A1 damage) / max(block A1 damage, .5)` ≤ .15.
*Worked example:* a uniform random 1.4% of a dense read would carry ~.014, and of a sparse read ~0; this is the control that makes
pred_b non-trivial — without it, a large top-64 share could simply mean any 64 units matter. Seeded (2821) and drawn once per cell.
Null: ≥ .40.

## Stated null

The read is dense and unit selection buys nothing: top-64 share ≤ .20, no specificity gain over the block, Jaccard at chance, and a
random set carrying ≥ .40. Each null is reported separately.

## Price

≤ 7 behaviours × 2 readers × [ FIT ranking passes over A1 and C + OOD arms (block, top-64, random) over A1/P/C + an exactness check ]
at 16 rows per cell. Literal budget: ≤ 3,000 GPU forwards, 0 backwards, 0 fitted parameters, expected < 4 GPU-minutes.

## What this does NOT claim

Units of the hidden layer are a basis-dependent decomposition: this rung localises the read in the network's OWN coordinates, and says
nothing about whether a rotated basis would be sparser. Two readers only. FIT-selected sets scored on OOD is a valid two-phase design
but the SET is still chosen, so pred_d's overlap is the only claim here that is independent of that choice. Does not satisfy Codex's
four-phase integration contract; updates no circuit record.
