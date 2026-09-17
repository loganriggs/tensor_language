# Aspectual has/had — definition-of-done scorecard

Living document (Claude circuit lane). Rubric: `basis_aligned/better_circuits.md` §1. Style:
`basis_aligned/communicating_results.md`. Updated 2026-09-17.

**Path now.** `since/by` cue → MLP4 two-term write at `last`+period+`the` → attention5 heads
7/1/6/8 → carried blocks 6–8 → block9 heads 1/4 at the final query → resid10 → block11 head 3,
block15 head 5 (+ MLP12/14) → resid18 → has/had. **Delta today:** the five named components are
now *selective* on fresh rows against a norm-matched random null (edit); their midpoint
removals compose additively and beat a random-split null (edit), but random pieces of the
same removal are just as "selective", so the gate certifies the cue-defined delta, not the
module partition (row 11).

**Metrics box.** *Damage*: native has−had margin minus the edited margin, oriented toward the
native answer, mean over 64 rows (logits). *Fraction*: damage / native mean margin (2.07).
*Null*: 16 random directions of the same removed norm at the same slice; we report the max.
*Unrelated excess*: mean |move| of an unrelated reader minus its null mean. *Fresh*: rows unused
at any prior stage (the 16-agent/16-period lexicon introduced in v1); *opened*: rows seen by an
earlier removal run.

## Claim table

| # | claim | tag | rows | key numbers | status |
|---|---|---|---|---|---|
| 1 | Removing the cue-dependent half of attention9 H1/H4's write at the final query damages has/had | edit | fresh (v2) | 0.72 logits (35%); null max 0.03; positive on 64/64 | passes |
| 2 | Same for the MLP4 write at `last`+period+`the` | edit | fresh (v2) | 0.53 (26%); null max 0.05 | passes |
| 3 | Block11 H3 and block15 H5 midpoint removals are specific but small | edit | fresh (v2) | 0.19 each (9%); null max ≤ 0.02 | established (below the 10% LIVE bar) |
| 4 | Attention5 H7/H1/H6/H8 midpoint removal is specific and small | edit | fresh (v2) | 0.09 (4%); null max 0.02 | established |
| 5 | Each of the five is selective: was−were, who−which, night−day move ≤ null + 0.25×damage | edit | fresh (v2) | max excess +0.06 (mlp4) | passes, 5/5 |
| 6 | Whole-write zeroing of attention5 heads is a norm effect, not aspect content | edit | fresh (v1) | 0.94 damage vs null median 1.05 | falsified as a component test |
| 7 | Joint midpoint removal of the five reproduces the sum of singles | edit | opened (v2, v3) | gap 0.06 ≤ bar 0.13; normalized gap 0.033 vs random-split median 0.063 (min 0.062, max 0.064) | passes, and beats the random-split null 2× — but random splits are also near-additive, so the regime is close to linear |
| 10 | Pairwise Möbius terms are below 25% of the smaller piece | edit | opened (v3) | 9/10 pass; mlp4+attn5 = +0.025 vs bar 0.023 (serial upstream→transport interaction) | fails as registered; the one failure is the mechanistically expected pair |
| 11 | Per-piece selectivity distinguishes the module split from random pieces | edit | opened (v3) | 80/80 random coordinate pieces are live and pass the v2 gate | falsified: selectivity here is a property of the cue-defined delta, not of the module partition |
| 12 | Weight-only readout direction `O_h^T(u_has−u_had)` at attention9 H1/H4, removed at the final query (no partner row, no fit) | edit | opened (v4) | 0.70 (34%), 97% of the midpoint damage; null max 0.012; positive 64/64; unrelated excess +0.10/−0.01/+0.03 | passes: first donor-free selective removal |
| 13 | The readout direction lies along the cue-carrying write | fold | opened (v4) | cosine 0.46 (H1), 0.63 (H4); 0.39 (11.3), 0.51 (15.5); attn5 heads 0.10–0.31 | established |
| 14 | Attention5 heads write the answer directly | edit | opened (v4) | readout removal 0.04 vs null max 0.04 | falsified (transport heads, as the path says) |
| 15 | Head 11.3's readout direction is aspect-private | edit | opened (v4) | damage 0.12; was−were moves 0.22 vs null 0.03 | falsified: shared with subject number |
| 16 | The attention9 H1/H4 weight-only removal transfers to three template-varying constructions (cue moved, no `last`, agent-first with comma-final, began/ended cue) | edit | fresh templates (v5) | damage 0.72 / 0.98 / 0.80 logits = 34% / 36% / 46% of each native margin; positive 32/32 in each; null max ≤ 0.018; selective in all three; all six capability cells 1.00 | passes |
| 17 | Block11 H3's has/had readout contribution depends on the template | edit | fresh templates (v5) | 0.21 (7%) in the began/ended frame, ≈0 in the two `last`-free frames; was−were excess +0.12 to +0.32 everywhere | established: 11.3 is not part of the template-independent component |
| 8 | Joint removal is selective | edit | opened (v2) | unrelated moves 0.15/0.12/0.11 vs joint damage 1.79 | passes |
| 9 | Head 11.3 (subject-number head) serves has/had separably from was−were at this position | edit | fresh (v1→v2) | v1 whole-write moved was−were 0.34 (null 0.27); v2 midpoint 0.05 (null 0.08) | established |

## Five-property status

| property | status | what would close it |
|---|---|---|
| Simple | not yet counted | count readers/products/writers of the released program vs a random component of matched effect |
| Predicts OOD | partial: the attention9 removal effect is predicted (sign, positivity, 34–46% band) on new lexicon and new templates (row 16); strong form from token IDs not licensed | token-only generator for the source write; a frozen numeric prediction of the removal effect before running it |
| Extracted | partial (program v12 runs with ports resid10/resid18/basis/lm_head; paired states required) | close ports by folding, not fitting |
| Selective | **held for attention9 H1/H4** with a weight-only direction, random null and 3 readers, on the discovery shape and three new templates (rows 12, 16); held for the cue-defined delta at the other slices (rows 1–5); 11.3 fails (rows 15, 17) | — |
| Composes | held at five-way grain with a random-split null (row 7); one serial pairwise term above gate (row 10) | model the mlp4→attn5 term explicitly (five-arm design, better_circuits §3.6) |

## Limitations that change the reading

- Midpoint removal uses the partner row's native write, so every edit here is paired-causal; it
  is not a donor-free removal of a stored component.
- The MLP4 / attention5 results (rows 2, 4, 6, 14) still rest on one template shape; only the
  attention9 removal has been template-varied (row 16).
- The five components account for 87% of the fresh-row margin jointly, but the released path's
  own recovery from MLP4 was ~27–33%; the difference is the carried/parallel routes the
  midpoint edit also removes at those slices.

## Appendix: receipts

- v1 whole-write removal: `bilinear_quotient/circuits/followups/aspectual_anchor_dod_removal_v1_result.json`
- v2 midpoint removal: `.../aspectual_anchor_dod_removal_v2_result.json`
- v3 composition null: `.../aspectual_anchor_dod_composition_v3_result.json`
- v4 weight-only readout removal: `.../aspectual_anchor_dod_readout_removal_v4_result.json`
- v5 template transfer: `.../aspectual_anchor_dod_template_transfer_v5_result.json`
- Code: `bilinear_quotient/ops/aspectual_dod_lib.py`, `run_aspectual_dod_removal_v{1,2}.py`,
  `run_aspectual_dod_composition_v3.py`, `run_aspectual_dod_readout_removal_v4.py`, `run_aspectual_dod_template_transfer_v5.py`
