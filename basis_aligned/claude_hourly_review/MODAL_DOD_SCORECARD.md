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

## Five-property status
| property | status | next |
|---|---|---|
| Simple | 4 heads + contrast (sweep) | random-set null (v67) |
| Predicts OOD | frozen number held on one fresh panel (row 2) | templates (v68); natural rows |
| Extracted | held at the head boundary (row 5) | source folds |
| Selective | held (row 3) | — |
| Composes | additive (row 4) | pairwise + random-split null |

## Receipts
- v49b: `bilinear_quotient/circuits/followups/modal_remoteness_dod_reuse_census_v49b_result.json`
- v66: `.../modal_remoteness_dod_battery_v66_result.json`; code `ops/run_modal_dod_battery_v66.py`
