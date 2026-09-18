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

## Five-property status
| property | status | next |
|---|---|---|
| Simple | 4 heads + contrast (sweep-selected on 4 lines) | random-set null |
| Predicts OOD | first fresh panel only | frozen prediction on new templates |
| Extracted | NOT held: keep-only 0.47 | per-head keep-only; rank-2 readout? (declared, not fitted) |
| Selective | fails on the tense reader | weight-only check: cosine of the were−was and has−had readout directions per head |
| Composes | additive (row 3) | pairwise + random-split null |

## Receipts
- v55: `bilinear_quotient/circuits/followups/number_family_dod_battery_v55_result.json`; code `ops/run_number_dod_battery_v55.py`
