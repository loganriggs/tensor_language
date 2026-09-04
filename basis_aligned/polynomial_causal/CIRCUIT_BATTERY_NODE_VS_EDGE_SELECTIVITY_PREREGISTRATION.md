# CIRCUIT BATTERY — NODE VERSUS EDGE SELECTIVITY (preregistration)

Registered 2026-09-04 06:58Z (box clock, read immediately before this line was written). Claude, LANE 1 CUDA.
Rung `circuit_battery_node_vs_edge_selectivity`. Script: `ops/circuit_battery_node_vs_edge_selectivity.py`.
Input receipts: `circuit_battery_successor_full_sweep_results.json` (§2849, sha 6e20b244de4a96f1158d5fbbcdc704926048f6b87b1c524480c8b6f3c2a7b3be)
and `circuit_battery_reader_selectivity_results.json` (§2819, sha 8170669c13428850aa07c4539de28dd7d8164f25eb100f11996b63b02121735f).
IMMUTABLE: any change gets a new document, not an edit.

## The question, and why it is about my own reading

§2849 swept all 36 components on the numbered-list successor with the answer-preserving family P and the copy control C, and
**every one of the seven ADMISSIBLE components came out at selectivity ratio exactly 1.00** — attn1, attn5, attn6, attn8,
mlp0, mlp1, mlp4. §2819 measured the EDGES of attention 8's write on the same task at **.59** (mlp11), .90 (mlp10), 1.06
(mlp9), 1.12 (mlp8). Read naively that says task specificity in this model lives in edges rather than nodes, which would be a
real structural claim about how small a compiled program can be.

**But a ratio of exactly 1.00 across seven very different components is also precisely what SATURATION looks like.** If
removing a component destroys the answer on every family, each family's damage equals its own native margin and the ratio is
1 by construction — carrying no information about selectivity at all. Nothing in §2849 distinguished those two readings, and
the entire edge-versus-node conclusion depends on which holds. This rung measures saturation directly, as damage over each
family's NATIVE margin, for nodes and edges on the same rows, and only then compares selectivity.

Sign convention: damage d_m = m_NATIVE − m_arm, POSITIVE = the arm HURTS that family's own answer; saturation fraction =
d_m / max(m_NATIVE, .5), where 1.0 means the arm removed the whole native margin; selectivity ratio =
max(|d_P|, |d_C|) / max(d_A1, .5), LOWER IS MORE SELECTIVE. **No CE and no §312 L2 — the frontier's L2 is CE ADDED ABOVE THE
REAL MODEL where LOWER IS BETTER (§2135, norm-2304 at 2.6735); nothing here installs or may be quoted as one.**

Fixed before the run: task `numbered_list.index_successor`, split OOD, the seven admissible NODES from §2849
{attn1, attn5, attn6, attn8, mlp0, mlp1, mlp4} ablated at every position, and the four EDGES of attention 8's write
{mlp8, mlp9, mlp10, mlp11} removed as §2808's path-patching arms.

## Predictions

```
BARS  = {node_sat: .90, ctrl_sat: .90, edge_sat: .60, gap: .30, repro: .20, floor: .5}
NULLS = {node_sat_le: .60, ctrl_sat_le: .60, edge_sat_ge: .90, gap_le: 0.0}
```

**pred_a_node_arms_are_saturated** — median over the seven nodes of (A1 damage / native A1 margin) ≥ .90.
*Worked example:* this is registered as the DEFLATING hypothesis about my own §2849 reading. If whole-component ablation
destroys the answer, the fraction sits .95–1.05 and the 1.00 ratios are an artifact. If instead nodes remove only part of
the margin (.3–.6), the ratios are informative and §2849's "no node is selective" stands as a real result. Both operands
are margin quantities with a floored denominator. Null: ≤ .60.

**pred_b_controls_are_saturated_too** — median over the seven nodes of max(P saturation, C saturation) ≥ .90.
*Worked example:* the ratio is 1.00 only if BOTH numerator and denominator are at ceiling; if the target saturates but the
controls do not, the ratios would not have come out at exactly 1.00 and something else is going on. This is the second half
of the artifact test and it must pass with pred_a for the artifact reading to hold.

**pred_c_edges_are_not_saturated** — median over the four edges of (A1 damage / native A1 margin) ≤ .60.
*Worked example:* §2808 measured single-reader edge damages well below the whole-term deletion, so the edges should sit
.1–.5 and their ratios are therefore measuring something. If edges saturate too (≥ .90) then neither instrument is
informative on this task and the comparison in pred_d is empty. Null: ≥ .90.

**pred_d_edges_are_more_selective_than_nodes** — (minimum node ratio) − (minimum edge ratio) ≥ .30.
*Worked example:* §2849's minimum node ratio is 1.00 and §2819's minimum edge ratio is .59 (mlp11), giving .41. This is the
comparison the structural claim rests on, and it is registered to be read ONLY IF pred_a and pred_b hold — because if the
node arms are saturated, a gap says nothing about nodes and everything about the bluntness of the instrument. A DIFFERENCE
of two ratios, both floored. Null: ≤ 0.

**pred_e_ratios_replicate_the_earlier_rungs** — max over arms of |this rung's ratio − the published one| ≤ .20, against
§2849 for nodes and §2819 for edges. *Worked example:* both are re-measurements on the same frozen OOD rows with the same
arms, so agreement should be within sampling noise; a larger gap means this rung's arms differ from theirs and nothing here
is comparable. Instrument check.

## Stated null

The nodes are NOT saturated (≤ .60) — in which case §2849's seven exact 1.00s are a real and surprising result rather than a
ceiling, and the edge-versus-node structure stands on its own. That would be the more interesting outcome and it is what
pred_a's failure would mean.

## Price

3 families × 24 OOD rows × (1 native + 7 node ablations + 4 edge removals), batched by token length.
Literal budget: ≤ 900 GPU forwards, 0 backwards, **0 fitted parameters**, < 60 GPU-seconds. Uses `ops/fastload.py`.

## What this does NOT claim

One task, one split, seven nodes and four edges chosen by §2849's admissibility and §2808's writer. Saturation is measured
against the native MARGIN, which is the battery's own metric; an arm could be non-saturated in margin and still destroy the
model's output distribution in ways margin does not see. Nothing installs; no L2. Does not satisfy Codex's four-phase
integration contract; updates no circuit record.
