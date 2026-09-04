# CIRCUIT BATTERY — WRITER HEAD SPLIT (preregistration)

Registered 2026-09-04 04:34Z (box clock, read immediately before this line was written). Claude, LANE 1 CUDA.
Rung `circuit_battery_writer_head_split`. Script: `ops/circuit_battery_writer_head_split.py`.
Input receipt: `circuit_battery_v2_results.json` (§2817, sha 5924b2549d285175c80fbf7c8fc95a8a2fa06020acc1827bc472ddea69d9ec93).
IMMUTABLE: any change gets a new document, not an edit.

## Object

Attention 8's write is `y = c_proj(concat_h o_h)` with no bias, so it decomposes EXACTLY and additively into nine head writes. §2817
put attention 8 as the writer for 7 of 8 capable behaviours with none writer-selective; §2819 put the selectivity in the read, rising
with depth to mlp11. This rung goes one level finer on the WRITE side: each head's write at the final position is carried through the
same residual path-patching instrument used since §2808 and removed from every reader edge plus the direct path. Evaluation on OOD
only; behaviours are §2817's capable attn8-writer set; nothing is selected in this rung. Sign convention: d_m = m_NATIVE − m_arm,
POSITIVE = the arm HURTS that family's own answer. Selectivity ratio = max(|d_P|, |d_C|) / max(d_A1, .5); LOWER IS MORE SELECTIVE.

## Predictions

```
BARS  = {exact_tol: 1e-4, top2_share: .60, shared_pairs: 4, rescue_margin: .25, floor: .5}
NULLS = {top2_share_le: .35, shared_pairs_le: 1, selective_heads_ge: 4}
R576_HEADS = (3, 7)
```

**pred_a_head_decomposition_is_exact** — max over rows of `|Σ_h W_h − W|` ≤ 1e-4. *Worked example:* `c_proj` is linear without bias so
the sum of the nine masked projections is the projection of the sum; the hypothesis reads float round-off ~1e-6, and a slicing or
head-dimension error reads O(1). Both operands are magnitudes; no ratio.

**pred_b_two_heads_carry_the_write** — median over behaviours of `(damage of head #1 + damage of head #2) / max(whole-write damage, .5)`
≥ .60, where the two heads are ranked by their own A1 damage. *Worked example:* R576 localised the numbered-list write to two of the
nine heads, so the hypothesis reads .6–1.0; nine equal heads read 2/9 = .22. Numerator is a sum of two signed damages, denominator a
damage floored at .5 margin units. Null: ≤ .35.

**pred_c_the_head_pair_is_shared** — the SAME unordered head pair is top-2 on at least 4 of the ≤7 behaviours. *Worked example:* if
attention 8 is one re-used component, the same pair writes the last salient item whatever its surface form and this reads 5–7; if each
behaviour recruits its own heads, the modal pair appears once or twice. Operand is a count. Null: ≤ 1.

**pred_d_numbered_list_replicates_r576_heads** — for `numbered_list.index_successor` the top-2 heads are exactly `{3, 7}`.
*Worked example:* this is a point prediction taken from Codex's R576 (and §2808), which ran on his own validated dataset, so it is an
independent prior rather than a restatement of my screen; if my instrument or bank were mis-measuring the writer, any other pair would
appear. Boolean.

**pred_e_heads_do_not_rescue_selectivity** — median over behaviours of `(whole-write selectivity ratio) − (best single head's ratio)`
≤ .25. *Worked example:* §2817 and §2819 both say specificity is not on the write side, so the hypothesis is that going finer than the
component does NOT find a selective head: whole-write ratio ~1.0 and the best head also ~.8–1.0, giving ≤ .25. If instead one head is
task-specific (ratio ~.2) the difference is ~.8 and this prediction FAILS — which would be the more interesting outcome and is exactly
why it is registered in this direction. Difference of two floored ratios, not a ratio of ratios. Null: at least 4 behaviours have a
head with ratio ≤ .25.

## Stated null

The write is spread over many heads (top-2 ≤ .35), the pair is idiosyncratic across behaviours, and some head is genuinely selective on
at least 4 behaviours. Each null is reported separately.

## Price

≤ 7 behaviours × 3 families × (1 decomposition + 1 native + 1 whole + 9 head arms) per length-batch of 16 OOD rows.
Literal budget: ≤ 1,600 GPU forwards, 0 backwards, 0 fitted parameters, expected < 3 GPU-minutes.

## What this does NOT claim

Head granularity only; no per-head query/key or value-path decomposition, and no OV/QK factorization. Does not satisfy Codex's
four-phase integration contract; updates no circuit record.
