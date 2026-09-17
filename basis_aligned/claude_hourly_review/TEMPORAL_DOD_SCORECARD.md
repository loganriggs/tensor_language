# Temporal will/had — definition-of-done scorecard

Living document (Claude circuit lane), opened 2026-09-17 23:09 UTC after review 3. Rubric:
`basis_aligned/better_circuits.md` §1; style: `basis_aligned/communicating_results.md`; battery and
metrics as in `ASPECTUAL_DOD_SCORECARD.md` (damage, fraction, norm-matched random null, retention).

**Path now.** Cue token (tomorrow/earlier) → readout set S = {11.3, 9.1, 15.5, 9.4}, each head writing along
`O_h^T(u_will − u_had)` at the final query → will/had. S was selected by the blind 162-head sweep on the
line's authored rows (v27) and is scored here on fresh rows only. **Delta:** first rung landed.

## Claim table

| # | claim | tag | rows | key numbers | status |
|---|---|---|---|---|---|
| 1 | Removing S along its weight-only readout directions damages will/had | edit | fresh places, 2 constructions; agents fresh except 3 of 16 (broker, tenant, author) from the corpus reporter-alternate list — CORRECTION 23:32 (v28) | 3.45 logits = 81% of the margin; positive 64/64; null max 0.05 | passes |
| 2 | S is selective on was−were, who−which, night−day | edit | fresh (v28) | all three within null + 0.25×damage (11.3 is the subject-number head, yet was−were stays within gate) | passes |
| 3 | S is additive | edit | fresh (v28) | joint 3.45 vs Σ singles 3.34: gap 0.11 vs bar 0.09 (25% of the smallest single, 9.4 = 0.36) | fails as registered (superadditive by 3% of the joint) |
| 4 | 11.3 leads on fresh rows | edit | fresh (v28) | singles 11.3 1.39, 9.1 1.18, 15.5 0.41, 9.4 0.36 | passes |

| 5 | Keeping only the readout projection at the four heads keeps their will/had service | edit | opened (v29) | zero four slices 3.45 logits; keep-only retention 1.19; random-direction keep ≤ 0.015 | passes |
| 6 | The S removal transfers to three new constructions (adverb phrase, expected-that frame, agent-first comma-final) | edit | fresh templates (v29) | fractions 0.82 / 0.78 / 0.36; positive 32/32 each; null max ≤ 0.10; selective in all three; all six capability cells 1.00 | 2 of 3 inside the registered [0.40, 1.21] band; the comma-final construction (0.36) is live, selective and null-beating but below the band — fails as registered |
| 7 | Source fold of the four coefficients (oriented tomorrow−earlier): 11.3 reads the subject NP (the1 0.47, agent 0.31, cue 0.15; inherited 0.03); 9.1: the1 0.53, cue 0.28, agent 0.13; inherited -0.06; 9.4: cue 0.53, the1 0.48, place -0.01; inherited 0.12; 15.5 reads the cue (cue 0.46, agent 0.15, place 0.14; inherited 0.25) | fold | opened (v30) | closure 4e-7 | established; registered guesses (cue leads for 11.3; block-9 heads read context not cue) failed as written |
| 8 | Matched-count null: 16 random four-head sets along their own will/had readout directions | edit | opened (v31) | S 3.45 vs random max 0.075, median 0.038; none live | passes |
| 9 | Pairwise Möbius terms among the four heads are below 25% of the smaller piece; the head split is more additive than random coordinate splits of the same removal | edit | opened (v32) | largest pair 11.3×9.1 +0.08 (bar 0.29); four-way gap 0.11 vs bar 0.09 (replays row 3's failure); normalized gap 0.032 vs random-split median 0.038 | 3/4: pairwise and null held; the strict four-way bar fails by 0.02 |
| 10 | Frozen numeric prediction 0.80 ± 0.15 for the S removal on a fourth lexicon and a new comma-adverb construction | edit, frozen before access | fresh agents + fresh template; CORRECTION: 13 of the 16 place nouns belong to the corpus's shared object list that the temporal line's own authored PP slots draw from, so the places were not fresh (v33, 96 rows) | fractions bare_frame 0.79 / comma_adverb 0.81 / report_frame 0.83; positive on every row; all reader gates pass; all six capability cells 1.00 | passes |
| 11 | Natural rows (outcome-blind; tomorrow/earlier within 10 tokens of a will/had target): removing S lowers will−had on tomorrow rows by −1.34 (FineWeb, 90% of rows) / −1.57 (Pile, 86%) and raises it on earlier rows by +0.54 (72%) / +0.59 (70%); unrelated readers 0.06–0.09 vs mean |shift| 1.2–1.5 | edit, frozen bars | natural in-distribution (FineWeb) + OOD (Pile) (v34) | 11.3 alone carries the tomorrow shift (−0.37 / −0.33) but not the earlier one (+0.03 / −0.01): registered pred_d failed | 4/5 |
| 12 | The subject-NP state 11.3 reads (on `V_h^T v̂_h` at `the`/agent) is written by few mid-block writers (mlp:09 0.21, attn:09 0.17, mlp:10 0.16, mlp:08 0.16, attn:08 0.12; top-4 0.70); embedding 0.00 | fold | opened (v35) | closure 1e-7 | passes 3/3 |
| 13 | Relay edit: zeroing the five nominated NP writers (mlp8, attn8, mlp9, attn9, mlp10) at `the`/agent removes 56% of 11.3's readout effect (1.39 → 0.62 logits) and is itself live (1.51, 35%, 64/64) | edit | opened (v36) | fold (row 12) nominated 70%; identity replacement replays native | passes 4/4: the NP relay blocks 8–10 → 11.3 is causal |
| 14 | Attention8/9's NP-state writes on 11.3's reader direction are carried by two heads per block (8.1 1.00, 8.2 0.00; cue source 0.98 — attn8; 9.1 0.48, 9.4 0.45; cue source 0.49 — attn9) and sourced from the cue token | fold | opened (v37) | closure 4e-6 | passes 3/3: the adverb reaches 11.3 by cue → 8.1/9.1 at the NP → 11.3 |
| 15 | Cue-only terms at the readout heads: 15.5 keeps 0.48 of its service (inherited-only 0.27); 9.1+9.4 keep 0.36 (inherited-only -0.02) | edit | opened (v38) | full recomputation replays native | 2/4: no token-only adverb reader at the readout heads (unlike aspectual 8.1); the inherited branch is the weaker one, as registered |
| 16 | Head 8.1 at the NP positions is a token-only adverb reader: replacing its two slices by the cue-only inherited term (native pattern × λ × block-0 value of tomorrow/earlier) keeps 0.83 of its service (1.01 with both branches); `the`/agent/prefix single sources keep 0.00 / 0.00 / -0.04 | edit | opened (v39) | zero two slices 0.78 logits (18%, positive 100%); full recomputation replays native | passes 5/5 — the same head 8.1 as on the aspectual line, same mechanism, different cue token |
| 17 | Head 8.1's NP-position service is an 8-constant token table (per cue × NP position, constants from the other construction): retention 1.19 (bare constants on report rows), 0.57 (report constants on bare rows), pooled 0.77; native-pattern term 0.83 | edit | cross-construction (v40) | 22 forwards | passes 4/4 — 8.1's port is closed on this line as on the aspectual line |
| 18 | MLP8/9/10's NP-state writes on 11.3's reader direction expand exactly into writer-pair terms; they are diffuse: top pairs attn:07 x attn:08 0.03, attn:07 x attnhead:08:1 0.03, attn:08 x mlp:08 0.02, mlp:06 x mlp:07 0.02; pairs with head 8.1 as a factor carry 0.13 | fold | opened (v41) | closure 1e-4; the pooled MLP contrast is NEGATIVE on this direction (the MLP part opposes the attention part) | 1/3: kill criterion tripped — the MLP part of the NP state is a declared open port |

## Five-property status

| property | status | next |
|---|---|---|
| Simple | held at head grain: 4 heads + the will/had contrast; blind sweep (v27) and random-quadruple null (row 8) | — |
| Predicts OOD | **held**: authored panels (rows 6, 10) and natural rows on FineWeb and the Pile OOD corpus with frozen bars (row 11) | — |
| Extracted | held at the head boundary with four scalar ports (row 5); 11.3's port is the NP state written by blocks 8–10 (56% causal), whose attention part is heads 8.1/9.1 reading the cue token (rows 12–14); no readout head is a token-only adverb reader (row 15), but the NP-position writer 8.1 is (rows 16, 17): the adverb enters the chain as a token-only value with 8 stored constants; **two declared open ports**: the MLP8–10 part of the NP state (diffuse, row 18) and the block-9 pair's own reads | — (folding stops at this grain) |
| Selective | held on fresh rows and three templates (rows 2, 6) | — |
| Composes | held pairwise and against the random-split null (row 9); the four-way sum overshoots the strict bar by 0.02 (3% of the joint), recorded as a failure (rows 3, 9). Interpretation from rows 14–16: the largest pair term (11.3×9.1, +0.08) is a serial relay term — 9.1 also writes the NP state 11.3 reads — the same kind of term as the aspectual mlp4→attn5 pair | — |

## Receipts
- v27 sweep (aspectual lane receipt): `bilinear_quotient/circuits/followups/aspectual_anchor_dod_reuse_temporal_v27_result.json`
- v28: `.../temporal_auxiliary_dod_removal_v28_result.json`; code `ops/run_temporal_dod_removal_v28.py`
- v29: `.../temporal_auxiliary_dod_keep_and_templates_v29_result.json`; code `ops/run_temporal_dod_keep_and_templates_v29.py`
- v30 source fold: `.../temporal_auxiliary_dod_source_fold_v30_result.json`
- v41 MLP NP pair fold: `.../temporal_auxiliary_dod_mlp_np_pair_fold_v41_result.json`
- v40 8.1 constant patterns at NP: `.../temporal_auxiliary_dod_8_1_np_constant_v40_result.json`
- v39 8.1 token-only at NP: `.../temporal_auxiliary_dod_8_1_np_token_only_v39_result.json`
- v37 NP head fold: `.../temporal_auxiliary_dod_np_head_fold_v37_result.json`
- v38 cue-term test: `.../temporal_auxiliary_dod_cue_term_v38_result.json`
- v36 NP mediation: `.../temporal_auxiliary_dod_np_mediation_v36_result.json`
- v34 natural rows + result: `.../temporal_auxiliary_dod_natural_rows_v34.json`, `.../temporal_auxiliary_dod_natural_v34_result.json`
- v35 NP writer fold: `.../temporal_auxiliary_dod_np_writer_fold_v35_result.json`
- v33 frozen prediction: `.../temporal_auxiliary_dod_frozen_prediction_v33_result.json`
- v32 composition: `.../temporal_auxiliary_dod_composition_v32_result.json`
- v31 random-set null: `.../temporal_auxiliary_dod_random_set_null_v31_result.json`
