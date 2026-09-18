# Number family (agreement at the auxiliary slot) — definition-of-done scorecard

Opened 2026-09-18 00:22 UTC (fourth path; Claude lane). Rubric: `better_circuits.md` §1; battery as in the other
scorecards. Readers for this family: has−had (tense), who−which, night−day — was−were is the target axis.

**Path now.** Number of the subject noun (singular / plural token) → readout set {11.3, 5.7, 7.8, 9.7} writing along
`O_h^T(u_were − u_was)` at the final query → was/were. Selected by the reuse censuses on four number lines (v50b–v53).

## Claim table

| # | claim | tag | rows | key numbers | status |
|---|---|---|---|---|---|
| 1 | Removing the set along its readout directions damages was/were | edit | fresh lexical-number rows (v55) | 2.67 logits = 70%; positive 64/64; null max 0.18 | passes |
| 2 | The set is selective on has−had, who−which, night−day | edit | v55 | has−had moves 0.88 vs null 0.20 (gate fails); who−which 0.22 vs 0.25 and night−day 0.25 vs 0.15 pass | fails: the number readout shifts tense |
| 3 | The set is additive | edit | v55 | singles 11.3 1.43, 5.7 0.50, 7.8 0.42, 9.7 0.28; gap 0.06 ≤ bar 0.07 | passes |
| 4 | Keeping only the readout projection keeps the heads' was/were service | edit | v55 | retention 0.47 (random keep ≤ 0.02) | fails: these heads' was/were service is not one direction each |
| 5 | 11.3 leads | edit | v55 | 1.43 vs 0.50 | passes |
| 6 | The number heads' were−was and has−had readout directions overlap in weight space: cos -0.54 (11.3), -0.57 (7.8), -0.41 (9.7), -0.16 (5.7); temporal heads -0.06 / 0.01 / -0.07; unembedding -0.26 | fold (weights only) | v56 | 0 forwards | passes 3/3: the tense movement in row 2 is a weight-level overlap of the two contrasts at the number heads |
| 7 | Removing only the part of each were−was direction orthogonal to its has−had direction (weights only, no fit) keeps 68% of the set's damage (1.82 logits, 48%, positive 100%, null max 0.13) and spares has−had (0.32 vs null 0.18); who−which / night−day within gate | edit | fresh (v57) | 38 forwards | passes 5/5 |
| 8 | Rank-2 keep (span of the were−was and has−had directions), implemented as replacement by the projection of the NATIVE slices: rank-1 0.62, rank-2 0.32, random rank-2 ≤ 0.06 | edit | v58 | identity replay exact | 2/4: pred_c (rank-1 replays v55's 0.47) failed because this protocol fixes later heads' slices at native values while v55's in-forward keep-only projects the slices the edited forward actually produces — a protocol difference, not a science result; rank-2 claim unsupported here; rerun in-forward as v58b |

## Five-property status
| property | status | next |
|---|---|---|
| Simple | 4 heads + contrast (sweep-selected on 4 lines) | random-set null |
| Predicts OOD | first fresh panel only | frozen prediction on new templates |
| Extracted | NOT held: keep-only 0.47 | per-head keep-only; rank-2 readout? (declared, not fitted) |
| Selective | held with the orthogonalized (weight-only) number direction (row 7); the raw were−was direction is entangled with tense at the number heads (rows 2, 6) | — |
| Composes | additive (row 3) | pairwise + random-split null |

## Receipts
- v58 rank-2 keep (replace protocol; protocol mismatch noted): `.../number_family_dod_keep_rank2_v58_result.json`
- v56 cosines (CPU): `.../number_family_dod_readout_cosines_v56_result.json`
- v57 orthogonalized readout: `.../number_family_dod_orthogonal_readout_v57_result.json`
- v55: `bilinear_quotient/circuits/followups/number_family_dod_battery_v55_result.json`; code `ops/run_number_dod_battery_v55.py`
