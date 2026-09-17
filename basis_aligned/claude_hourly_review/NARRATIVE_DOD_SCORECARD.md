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

## Five-property status
| property | status | next |
|---|---|---|
| Simple | 4 heads + contrast (sweep-selected; shared with the temporal set) | random 4-set null |
| Predicts OOD | not yet | fresh-lexicon confirmation with a frozen number (v43); templates; natural rows |
| Extracted | not yet | keep-only; source folds (what state does 15.5 read?) |
| Selective | held on the selecting rows only (row 2) | fresh rows |
| Composes | not yet; reuse across lines is the headline (same set as temporal) | pairwise terms; cross-line comparison of the readout directions |

## Receipts
- v42: `bilinear_quotient/circuits/followups/narrative_tense_dod_sweep_and_set_v42_result.json`; code `ops/run_narrative_dod_sweep_and_set_v42.py`
