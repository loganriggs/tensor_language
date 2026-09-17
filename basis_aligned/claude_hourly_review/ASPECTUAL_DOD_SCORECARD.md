# Aspectual has/had — definition-of-done scorecard

Living document (Claude circuit lane). Rubric: `basis_aligned/better_circuits.md` §1. Style:
`basis_aligned/communicating_results.md`. Updated 2026-09-17.

**Path now.** `since/by` cue → MLP4 two-term write at `last`+period+`the` → attention5 heads
7/1/6/8 → carried blocks 6–8 → **readout component {8.1, 9.1, 9.4}** at the final query, each
head writing along `O_h^T(u_has−u_had)` → read directly by the unembedding (57%) and amplified by
MLPs 9–15 (43%), mlp17 opposing → has/had. Block11 H3 / block15 H5 are small, template-dependent
extras; 11.3 is shared with subject number. **Delta today:** the five named components are
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
| 18 | The attention9 removal acts 57% directly and 43% through downstream responses; attention modules 10–17 are inert responders; MLPs 9–15 amplify (mlp11 largest, −0.11), mlp17 opposes (+0.07, the calibrator sign) | response | opened (v6, 160 rows, 4 constructions) | exact −0.78 logits pooled; recurrence closure ≤ 2e-5; linear remainder 1e-4; downstream net same sign as direct in all 4 constructions | established; registered guess of the largest responder (mlp12/14/17) was wrong — it is mlp11 |
| 19 | Blind sweep of the same weight-only readout removal over all 162 heads: only three are live — 9.1 (0.36), **8.1 (0.34, not named by the released path)**, 9.4 (0.33); then 15.5 (0.18), 11.3 (0.12); every other head < 0.04 | edit | opened (v7) | 5 of 162 heads exceed 0.05; pair 9.1+9.4 additive (joint 0.70 vs sum 0.69) | established; registered prediction that nothing outside blocks 9–15 reaches half of 9.4 is FALSIFIED by 8.1 |
| 20 | Head 8.1's weight-only readout removal is live, null-beating and selective in all four constructions | edit | opened (v8) | 0.32–0.37 logits (14–18%); positive ≥ 31/32 everywhere; was−were excess ≤ +0.09 | passes |
| 21 | The three-head readout component {8.1, 9.1, 9.4} is live, null-beating, selective and additive with the block-9 pair in all four constructions | edit | opened (v8) | 1.04 / 1.41 / 1.08 / 1.14 logits = 50 / 52 / 52 / 66% of each margin (pooled 54%); positive 160/160; additivity gap ≤ 0.06 vs bars ≥ 0.08 | passes |
| 22 | Keeping ONLY the readout projection at the three heads (discarding everything else they write at the final query) retains the heads' whole has/had contribution | edit | opened (v9, 160 rows) | zeroing the three slices: 1.25 logits; keep-only readout: 1.01 retention pooled (0.94 / 1.05 / 1.02 / 1.03 per construction); keep-only a random direction: ≤ 0.015 retention over 16 seeds | passes: the component's declared input is three scalars `c_h = w_h·v̂_h` |
| 23 | Head 8.1's coefficient contrast is a cue-token read: 83% from the since/by position, 78% through the block-0 (token-only) value branch; heads 9.1/9.4 read the contextual `last`+period+`the` bank through the current-block value (cue 15% / 2%; inherited ≈ 0) | fold | opened (v10, 32 aligned pairs × 2 constructions) | closure 5e-5; contrasts 46 / 32 / 21 (coefficient units); block-8 λ = 4.0, block-9 λ = −0.66 | established; registered guesses that the cue leads for every head and that top-2 sources carry ≥ 70% failed for 9.1 |
| 24 | Head 8.1's final-query output can be replaced by one term, its native pattern on the since/by token × λ × the block-0 value of that token (a token-only table), and keeps 76% of the head's has/had service (83% with both value branches); every other single source keeps ≤ 14% | edit | opened (v11b, 64 rows) | zero slice 0.46 logits; full recomputation replays native to 5e-6 | passes; v11 executed the same arms but withheld its receipt because I mis-counted its price (30 vs 38 forwards); log preserved |
| 25 | Head 8.1's has/had service is a two-entry lookup table on the cue token: constant × λ × block-0 value(cue), the constant being the median native pattern per cue from the *other* construction | edit | cross-construction (v12) | retention 0.78 (fronted constant on report rows), 0.87 (report constant on fronted rows), 0.83 pooled — above the native-pattern term (0.76); the native scalar's CV is 0.37–0.45, so it is not literally constant but its variation does not matter | passes (4/5; the CV bar failed as registered) |
| 26 | The bank states heads 9.1/9.4 read (on the weight-only reader direction `V_h^T v̂_h` at `last`/period/`the`) are written mainly by attention8 (29% / 17%), MLP8 (28% / 28%), MLP7 (10% / 17%) and MLP6 (10% / 12%); MLP4 contributes 4% / 9%; the embedding 2% | fold | opened (v13) | writer closure 1e-7; top-3 writers 66% / 62% | established; the registered prediction that MLP4 is the decisive bank writer (from the released path's paired-interchange story) FAILED on this reader direction |
| 8 | Joint removal is selective | edit | opened (v2) | unrelated moves 0.15/0.12/0.11 vs joint damage 1.79 | passes |
| 9 | Head 11.3 (subject-number head) serves has/had separably from was−were at this position | edit | fresh (v1→v2) | v1 whole-write moved was−were 0.34 (null 0.27); v2 midpoint 0.05 (null 0.08) | established |

## Five-property status

| property | status | what would close it |
|---|---|---|
| Simple | the readout component is three (layer, head) indices plus the has/had contrast over native weights; the blind sweep (row 19) shows only 3 of 162 heads carry it, so a random component of matched effect size does not exist at head grain | count the MLP-cascade readers (v6) the same way |
| Predicts OOD | partial: the removal effect is predicted (sign, positivity, 34–66% band) on new lexicon and new templates (rows 16, 21); strong form (token IDs in) holds for the 8.1 sub-component only (row 25) | token-only generator for the block-9 bank; a frozen numeric prediction of the removal effect before running it |
| Extracted | held at the head boundary with three scalar ports (row 22); **8.1's port is closed** (row 25); 9.1/9.4's ports are bank reads written by blocks 6–8, not by a token-only source (row 26) | fold attention8's write at the bank positions by block-8 head and source token (is it the cue?); if that is not token-dominated either, stop folding and report three open ports |
| Selective | **held for the three-head readout component {8.1, 9.1, 9.4}** with weight-only directions, random null and 3 readers, on the discovery shape and three new templates (rows 12, 16, 20, 21); 11.3 fails (rows 15, 17) | — |
| Composes | held: the three heads are additive in all four constructions (row 21); five-way midpoint grain also holds with a random-split null (row 7); one serial pair (mlp4→attn5) above gate (row 10) | five-arm design for the mlp4→attn5 term |

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
- v6 response census: `.../aspectual_anchor_dod_response_census_v6_result.json`
- v7 all-head sweep: `.../aspectual_anchor_dod_head_sweep_v7_result.json`
- v8 three-head promotion: `.../aspectual_anchor_dod_triple_v8_result.json`
- v9 keep-only sufficiency: `.../aspectual_anchor_dod_keep_only_v9_result.json`
- v10 source fold: `.../aspectual_anchor_dod_source_fold_v10_result.json`
- v11b cue-source edit: `.../aspectual_anchor_dod_cue_source_v11b_result.json` (v11: price mis-registration, log only)
- v12 constant pattern: `.../aspectual_anchor_dod_constant_pattern_v12_result.json`
- v13 bank writer fold: `.../aspectual_anchor_dod_bank_writer_fold_v13_result.json`
- Code: `bilinear_quotient/ops/aspectual_dod_lib.py`, `run_aspectual_dod_removal_v{1,2}.py`,
  `run_aspectual_dod_composition_v3.py`, `run_aspectual_dod_readout_removal_v4.py`, `run_aspectual_dod_template_transfer_v5.py`, `run_aspectual_dod_response_census_v6.py`, `run_aspectual_dod_head_sweep_v7.py`, `run_aspectual_dod_triple_v8.py`, `run_aspectual_dod_keep_only_v9.py`, `run_aspectual_dod_source_fold_v10.py`, `run_aspectual_dod_cue_source_v11b.py`, `run_aspectual_dod_constant_pattern_v12.py`, `run_aspectual_dod_bank_writer_fold_v13.py`
