# Modal would/will — definition-of-done scorecard

Opened 2026-09-18 00:47 UTC (fifth path; Claude lane). Rubric: `better_circuits.md` §1; battery as in the other scorecards.

**Path now.** Conditional cue (if + past antecedent vs when + present antecedent, one clause earlier) → readout set
{9.4, 11.3, 9.1, 15.5} along `O_h^T(u_would − u_will)` at the final query → would/will. The temporal family again.

## Claim table

| # | claim | tag | rows | key numbers | status |
|---|---|---|---|---|---|
| 1 | Blind sweep on the line's authored rows returns the temporal family {9.4 0.52, 11.3 0.51, 9.1 0.29, 15.5 0.21}; set 59%, selective | edit | opened (v49b) | null max 0.04 | established |
| 2 | Frozen 0.59 ± 0.15 on fresh rows (new places, new adjectives, both constructions) | edit, frozen | fresh (v66) | 0.57 / 0.59; positive 64/64; null max 0.05; 4/4 capability cells 1.00 | passes |
| 3 | Selective (was−were, who−which, night−day within null + 0.25×damage) | edit | fresh (v66) | all three pass | passes |
| 4 | Additive | edit | fresh (v66) | singles 0.56 / 0.49 / 0.35 / 0.19; gap 0.017 ≤ bar 0.047 | passes |
| 5 | Keep-only readout at the four heads keeps their service | edit | fresh (v66) | retention 0.96; random keep ≤ 0.01 | passes |
| 6 | Matched-count random four-head-set null | edit | v67 | set 1.60 vs random max 0.12; none live | passes |
| 7 | Template transfer: fractions had_once 0.62 / perfect 0.54 / should_whenever 0.51; the 'Should … / Whenever …' frame fails native capability (no past/present antecedent) and is excluded; the two capable frames transfer within band, positive every row, selective | edit | fresh templates (v67) | | passes on the capable frames |

| 8 | Natural FineWeb rows (two-token cue: If / if + had within the window → remote; When / when / Once + has / have → present; next token would / will; other modals excluded): congruent removal 1.55 of 5.08 (31%), positive 31/32, null max 0.04, selective; remote/would 2.13 of 5.59 (16/16), present/will 0.98 of 4.57 (15/16); the 4 counter-case rows shift toward the text (−0.63) | edit | natural (v153) | v20 bars held | passes 6/6 |
| 9 | Pile rows: congruent 1.50 of 5.26 (28%), positive 31/31, null 0.04, selective; remote/would 2.19 of 5.35, present/will 0.76 of 5.16; 5 counter-cases −0.01 | edit | natural OOD (v154) | | passes 6/6 |

## Five-property status
| property | status | next |
|---|---|---|
| Simple | held at head grain (row 6) | — |
| Predicts OOD | held on a fresh panel, two new conditional frames, and natural FineWeb / Pile rows (rows 2, 7, 8, 9) | — |
| Extracted | held at the head boundary (row 5) | source folds |
| Selective | held (row 3) | — |
| Composes | additive (row 4) | pairwise + random-split null |

## Receipts
- v153 / v154: `.../modal_dod_natural_v153_result.json`, `.../modal_dod_pile_v154_result.json`; rows `modal_dod_{natural,pile}_rows_v15{3,4}.json`; miner config `ops/modal_dod_natural_rows.py` (two-token cue via `dod_natural_miner.mine(second=...)`)
- v49b: `bilinear_quotient/circuits/followups/modal_remoteness_dod_reuse_census_v49b_result.json`
- v67 null + templates: `.../modal_remoteness_dod_null_and_templates_v67_result.json`
- v66: `.../modal_remoteness_dod_battery_v66_result.json`; code `ops/run_modal_dod_battery_v66.py`
