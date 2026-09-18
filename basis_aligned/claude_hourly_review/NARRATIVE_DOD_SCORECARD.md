# Narrative tense (was/is) — definition-of-done scorecard

Living document (Claude circuit lane), opened 2026-09-17 23:30 UTC (third path). Rubric: `better_circuits.md` §1;
battery and metrics as in `ASPECTUAL_DOD_SCORECARD.md`.

**Path now.** Tense cue spread over an earlier sentence ("Last winter … stood" / "Every winter … stands") → readout
set {15.5, 11.3, 9.4, 9.1} at the final query, each head writing along `O_h^T(u_was − u_is)` → was/is. **Delta:**
the blind sweep on fresh rows returns exactly the temporal will/had set, with 15.5 leading instead of 11.3.

## Claim table

| # | claim | tag | rows | key numbers | status |
|---|---|---|---|---|---|
| 1 | Blind 162-head readout sweep (was−is): 15.5 0.57, 11.3 0.54, 9.4 0.14, 9.1 0.09, then 7.7 0.06 and nothing above 0.04 | edit | fresh subjects/places/focus (v42) | 4 capability cells 1.00 | established: same four heads as the temporal set |
| 2 | The top-4 set removes 74% of the was/is margin, beats the norm-matched null (max 0.03), positive 64/64, selective on all three readers | edit | opened by the same sweep (v42) | 1.39 logits | passes; fresh-row confirmation pending (v43) |
| 3 | Head 8.1 is not in the top six | edit | v42 | 8.1 absent | established: no single cue token to read here |
| 4 | Frozen prediction 0.74 ± 0.15 on new places/focus and subjects new to this behaviour (5 never used; 11 lexicon-3 agents from the temporal panels, declared) | edit, frozen | fresh to this path (v43) | fractions direct 0.72, relative 0.74; positive 64/64; null max 0.024; all reader gates pass; capability 4/4 cells 1.00 | passes |
| 5 | Keep-only readout at the four heads keeps their was/is service | edit | v43 | retention 1.21; random keep ≤ 0.022 | passes |
| 6 | Additivity: singles 15.5 0.55, 11.3 0.57, 9.4 0.19, 9.1 0.08; joint 1.46; gap 0.061 vs bar 0.021 (25% of the smallest single) | edit | v43 | superadditive by 4% of the joint | fails as registered (same pattern as the temporal set) |
| 7 | Source fold (oriented past−present): 15.5 reads rest1 0.31, verb 0.25; 11.3 reads tail 0.90, subject_np 0.10; 9.4 rest1 0.35, verb 0.29; 9.1 tail 1.39, season -0.21; inherited branches ≈ 0 | fold | v44 | closure 7e-6 | established; registered guess (tensed verb or adverb leads for 15.5/11.3) FAILED — the readout heads read the tense from the second sentence's contextual state, not from the tense-bearing tokens |
| 8 | Matched-count null: 16 random four-head sets on their own was/is readout directions | edit | v45 | set 1.46 vs random max 0.059, median 0.005; none live | passes |
| 9 | Set removal transfers to three new tense frames (years ago / these days; once / still; back then / right now) | edit | fresh templates (v46) | fractions back_then_now 0.69 / once_still 0.75 / years_ago 0.75; positive every row; null max ≤ 0.03; selective everywhere; all six capability cells 1.00 | passes |
| 10 | Pairwise Möbius terms all below 25% of the smaller piece (largest attn15_h5+attn11_h3 +0.025 vs bar 0.138); head split normalized gap 0.042 vs random-split median 0.050; four-way gap 0.061 vs bar 0.021 | edit | v47 | 156 forwards | 3/4: pairwise and null held; strict four-way bar fails (same pattern as the temporal set) |

| 11 | Natural FineWeb rows (a natural ANALOGUE of the panel cue: one temporal adverb — yesterday / ago / formerly / previously / earlier vs today / now / currently / nowadays / presently — within 12 tokens of was / is; be-forms excluded): congruent removal 1.08 of 2.88 (38%), positive 29/32, null max 0.03, selective (who−which, night−day, they−he); past/was 1.53 of 2.82 (16/16), present/is 0.64 of 2.93 (13/16) | edit | natural (v151) | v20 bars held | passes 5/6 |
| 11f | **Failed:** counter-case rows were predicted to shift toward the text; pooled they are hurt (+0.21 FineWeb, +0.28 Pile) — asymmetrically: past-adverb + is rows move toward the text (−0.16 / −0.20) while present-adverb + was rows are hurt (+0.58 / +0.77): the set carries the 'was' side of the decision whatever the adverb, and the present side only with a present cue | edit | natural (v151, v152) | | falsified as registered |
| 12 | Pile rows: congruent 0.75 of 2.37 (32%), positive 25/32, null 0.03, selective; capability 0.81 / 0.88 | edit | natural OOD (v152) | | passes 5/6 |

## Five-property status
| property | status | next |
|---|---|---|
| Simple | held at head grain: 4 heads + contrast; blind sweep (row 1) and random-quadruple null (row 8) | — |
| Predicts OOD | held on a fresh-to-path panel with a frozen number (row 4) and on three new tense frames (row 9) | natural rows (tense cue is spread; miner not built) |
| Extracted | held at the head boundary (row 5); the ports are contextual states of the current sentence (row 7) — tense has already been propagated across the sentence boundary by earlier layers | writer fold of the tail state (kill criterion applies) |
| Selective | held on the selecting rows, the fresh panel and three templates (rows 2, 4, 9) | — |
| Composes | held pairwise and against the random-split null (row 10); the strict four-way bar fails by 4% (rows 6, 10); reuse across lines is the headline (same set as temporal) | — |

## Receipts
- v151 / v152: `.../narrative_dod_natural_v151_result.json`, `.../narrative_dod_pile_v152_result.json`; rows `narrative_dod_{natural,pile}_rows_v15{1,2}.json`; miner config `ops/narrative_dod_natural_rows.py`
- v47 composition: `.../narrative_tense_dod_composition_v47_result.json`
- v45 random-set null: `.../narrative_tense_dod_random_set_null_v45_result.json`
- v46 templates: `.../narrative_tense_dod_templates_v46_result.json`
- v44 source fold: `.../narrative_tense_dod_source_fold_v44_result.json`
- v43: `.../narrative_tense_dod_confirm_v43_result.json`; code `ops/run_narrative_dod_confirm_v43.py`
- v42: `bilinear_quotient/circuits/followups/narrative_tense_dod_sweep_and_set_v42_result.json`; code `ops/run_narrative_dod_sweep_and_set_v42.py`
