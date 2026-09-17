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

## Five-property status
| property | status | next |
|---|---|---|
| Simple | 4 heads + contrast (sweep-selected; shared with the temporal set) | random 4-set null |
| Predicts OOD | held on a fresh-to-path panel with a frozen number (row 4) | templates; natural rows |
| Extracted | held at the head boundary (row 5) | source folds (what do 15.5 and 11.3 read here?) |
| Selective | held on the selecting rows and on the fresh panel (rows 2, 4) | — |
| Composes | strict four-way bar fails by 4% (row 6); pairwise terms not yet; reuse across lines is the headline (same set as temporal) | pairwise terms; cross-line comparison |

## Receipts
- v43: `.../narrative_tense_dod_confirm_v43_result.json`; code `ops/run_narrative_dod_confirm_v43.py`
- v42: `bilinear_quotient/circuits/followups/narrative_tense_dod_sweep_and_set_v42_result.json`; code `ops/run_narrative_dod_sweep_and_set_v42.py`
