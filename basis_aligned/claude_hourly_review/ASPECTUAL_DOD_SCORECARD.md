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
| 27 | Attention8's bank write on the block-9 reader directions is head 8.1 again (1.03 / 0.99 of the contrast for readers 9.1 / 9.4), reading the since/by token (0.87 / 0.83) through the block-0 value branch (inherited 0.82 / 0.83) | fold | opened (v14) | closure 2e-5 | established: the route cue → 8.1 at `last`/period/`the` → 9.1/9.4 has the same token-only form as 8.1's direct route |
| 28 | Head 8.1 replaced by an 8-constant token-only table (per cue × query position, constants from the other construction) at `last`, period, `the` and the final query retains 79% of everything the head contributes to has/had (zeroing it at those four positions costs 0.88 logits); the native cue-only term retains 83% | edit | cross-construction (v15) | full recomputation replays native to 7e-6 | passes |
| 29 | MLP8's bank write on the reader directions expands exactly into writer-pair terms; it is diffuse: the top pair (mlp6 × mlp7) carries 10% / 14%, pairs involving head 8.1 carry 43% / 21%, 8.1 × 8.1 alone 4% / 0% | fold | opened (v16, 676 pair terms) | closure 2e-4 | established; all three registered 8.1-dominance predictions FAILED. Kill criterion from review 2 tripped for this port: folding stops here; the MLP5–8 polynomial part of the bank stays an open native port |
| 30 | Strong-form OOD: head 8.1's final-query write replaced by a 2-constant token table (since 0.220, by 0.182; medians over the discovery rows) on the three template-varying constructions | edit, token IDs in | fresh templates (v17) | retention agent_first_comma: 0.99 / clear_that_began_ended: 0.77 / ever_since_by_end: 1.08; pooled 0.93; native-pattern cue term 0.85–0.89 | passes |
| 31 | Relay edit: zeroing head 8.1's write at the three bank positions (final query left native) removes 27% of the block-9 readout effect (0.70 → 0.51 logits), and is itself live (0.37) | edit | opened (v18) | fold (row 27) nominated 17–29%; bank-slice recomputation replays native | passes: the cue → 8.1@bank → 9.1/9.4 relay is causal, ~¼ of the block-9 readout |
| 32 | Frozen numeric prediction: the triple readout removal was predicted to remove 0.54 ± 0.15 of the margin on a third lexicon and a new subordinate-clause construction, and the 8.1 token table (since 0.220, by 0.182) to retain ≥ 0.50 | edit, frozen before access | fresh lexicon + fresh template (v19, 96 rows) | fractions clause_comma 0.43 / fronted 0.53 / report 0.54; positive on every row; all three readers within gate; table retention 1.24 / 0.90 / 0.75; all six capability cells ≥ 0.94 | passes |
| 33 | Natural FineWeb rows (outcome-blind: a has/had target with a since/by token in the previous 10 tokens, any sense): removing the component lowers has−had on `since` rows (−0.25 logits, 69% of rows; head 8.1 alone −0.15, 88%) as the token-only mechanism predicts; on `by` rows the predicted rise is only +0.09 (53%; 8.1 alone +0.02) — most mined `by` are agentive, and the frozen +0.15 / 60% bar fails | edit | fresh corpus (v20) | native accuracy 0.84 / 0.94; unrelated readers move 0.04–0.06, but my gate was written against the pooled signed shift (0.08, cancelling across cues) and fails as registered | 2/5: `since` transfers to corpus; `by` does not at any-sense grain; gate mis-designed (preserved) |
| 34 | On natural rows the shift caused by removing head 8.1 scales with 8.1's attention pattern on the cue: Pearson −0.89 (since rows), +0.63 (by rows); the top-tercile since rows shift −0.30, the bottom −0.02. Natural `by` gets little 8.1 attention (median pattern 0.01 vs 0.18 in templates) | fold + edit | fresh corpus (v21) | on an outcome-blind temporal-`by` panel the triple removal shifts has−had by +0.36 (66% of rows), so the component fires on deadline `by` in the wild through the block-9 relay while 8.1's direct term stays small (+0.04); top-tercile any-sense `by` +0.06 < 0.10 bar | 4/5: the pattern, not the value, carries the sense |
| 35 | Recomputing head 8.1's cue key from the embedding-only cue state collapses its attention (median pattern 0.005 on since rows, −0.002 on by rows, vs 0.06 / 0.01 native) and the pattern no longer tracks the shift (r −0.14 / −0.31 vs −0.89 / +0.63) | fold | fresh corpus (v22) | pattern replay exact | 2/4: the registered guess that context *suppresses* attention on agentive `by` was wrong — context *builds* the key everywhere; only the value is token-only |
| 8 | Joint removal is selective | edit | opened (v2) | unrelated moves 0.15/0.12/0.11 vs joint damage 1.79 | passes |
| 9 | Head 11.3 (subject-number head) serves has/had separably from was−were at this position | edit | fresh (v1→v2) | v1 whole-write moved was−were 0.34 (null 0.27); v2 midpoint 0.05 (null 0.08) | established |

## Five-property status

| property | status | what would close it |
|---|---|---|
| Simple | the readout component is three (layer, head) indices plus the has/had contrast over native weights; the blind sweep (row 19) shows only 3 of 162 heads carry it, so a random component of matched effect size does not exist at head grain | count the MLP-cascade readers (v6) the same way |
| Predicts OOD | held on authored panels (rows 16, 21, 32; strong form for 8.1, rows 30, 32) and on natural corpus for both cues once the cue is temporal (rows 33, 34); the token-only table for `by` is an authored-template fact — in the wild the sense is carried by 8.1's attention pattern | fold 8.1's QK product on the cue into its sources (what makes it attend to deadline `by`) |
| Extracted | held at the head boundary (row 22); head 8.1 is a token-only generator wherever it matters (rows 25, 28); 9.1/9.4 relay a bank that is about half 8.1's token-only write and half a diffuse MLP5–8 polynomial (rows 26, 29) — **two open native ports remain: that polynomial and the block-9 patterns** | stop folding at this grain (kill criterion); report the ports |
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
- v14 attention8 bank fold: `.../aspectual_anchor_dod_attn8_bank_fold_v14_result.json`
- v15 token-only 8.1 at four positions: `.../aspectual_anchor_dod_token_only_8_1_v15_result.json`
- v16 MLP8 pair fold: `.../aspectual_anchor_dod_mlp8_pair_fold_v16_result.json`
- v17 token-only OOD: `.../aspectual_anchor_dod_token_only_ood_v17_result.json`
- v18 relay mediation: `.../aspectual_anchor_dod_relay_mediation_v18_result.json`
- v19 frozen prediction: `.../aspectual_anchor_dod_frozen_prediction_v19_result.json`
- v20 natural rows + result: `.../aspectual_anchor_dod_natural_rows_v20.json`, `.../aspectual_anchor_dod_natural_v20_result.json`
- v22 cue key token-vs-context fold: `.../aspectual_anchor_dod_cue_pattern_fold_v22_result.json`
- v21 pattern vs shift + temporal-by panel: `.../aspectual_anchor_dod_natural_rows_v21.json`, `.../aspectual_anchor_dod_natural_pattern_v21_result.json`
- Code: `bilinear_quotient/ops/aspectual_dod_lib.py`, `run_aspectual_dod_removal_v{1,2}.py`,
  `run_aspectual_dod_composition_v3.py`, `run_aspectual_dod_readout_removal_v4.py`, `run_aspectual_dod_template_transfer_v5.py`, `run_aspectual_dod_response_census_v6.py`, `run_aspectual_dod_head_sweep_v7.py`, `run_aspectual_dod_triple_v8.py`, `run_aspectual_dod_keep_only_v9.py`, `run_aspectual_dod_source_fold_v10.py`, `run_aspectual_dod_cue_source_v11b.py`, `run_aspectual_dod_constant_pattern_v12.py`, `run_aspectual_dod_bank_writer_fold_v13.py`, `run_aspectual_dod_attn8_bank_fold_v14.py`, `run_aspectual_dod_token_only_8_1_v15.py`, `run_aspectual_dod_mlp8_pair_fold_v16.py`, `run_aspectual_dod_token_only_ood_v17.py`, `run_aspectual_dod_relay_mediation_v18.py`, `run_aspectual_dod_frozen_prediction_v19.py`, `aspectual_dod_natural_rows.py`, `run_aspectual_dod_natural_v20.py`, `aspectual_dod_natural_rows_v21.py`, `run_aspectual_dod_natural_pattern_v21.py`, `run_aspectual_dod_cue_pattern_fold_v22.py`
