# Pronoun gender he/she — definition-of-done scorecard

Opened 2026-09-18 01:55 UTC (sixth path; Claude lane). Rubric: `better_circuits.md` §1; battery as in the other scorecards.

**Path now.** Gendered noun one clause earlier (king/queen …) → readout set {10.1, 9.6, 12.4, 15.1} along
`O_h^T(u_he − u_she)` at the final query (the conjunction / comma before the pronoun) → he/she. The triple {9.6, 12.4, 15.1}
is the readout atlas's most recurrent top-4 core (16 live lines, all pronoun gender / number / person cells), so this line
opens the PRONOUN readout family alongside the temporal and number families at the auxiliary slot.

## Claim table

| # | claim | tag | rows | key numbers | status |
|---|---|---|---|---|---|
| 1 | Blind 162-head sweep on the line's authored rows (128, capability 1.00 in all four cells) ranks 10.1 0.60, 9.6 0.58, 12.4 0.50, 15.1 0.29 (next: 8.1 0.22, 6.1 0.17); the top-4 set removes 89% of the margin, positive on 128/128, null max 0.05 | edit | opened (atlas v68) | readers moved 0.07 / 0.04 / 0.03 vs null 0.03 / 0.03 / 0.02 (all within null + 0.25×2.10) | established |
| 2 | Frozen 0.89 ± 0.15 on fresh rows (10 fresh gender pairs, 10 fresh objects, three unused frames) with capability ≥ 0.85 in every cell | edit, frozen | fresh (v71) | fractions lost 0.99 / because 0.96 / later 1.11 (pooled 1.02, positive 60/60, null max 0.02); capability 1.00 / 0.90 per side | **band failed upward**: 'later' exceeds 1.04; the set is live and null-beating everywhere, the frozen number under-predicted it |
| 3 | Selective on fresh rows | edit | fresh (v71) | readers moved 0.02 / 0.02 / 0.04 vs null-level bars | passes |
| 4 | Additive | edit | fresh (v71) | singles 12.4 0.70 / 10.1 0.43 / 9.6 0.30 / 15.1 0.26; gap 0.063 ≤ bar 0.064 | passes (at the bar) |
| 6 | Matched-count random four-head-set null (16 quadruples from the other 158 heads, each along its own he−she readout direction) | edit | v71 rows (v72) | set 1.75 (fraction 1.02, replay of v71) vs random max 0.12 (fraction 0.07); no random quadruple live | passes |
| 5 | Keep-only readout at the four heads keeps their service | edit | fresh (v71) | zero 1.69; keep-only damage −0.62 (retention 1.37: keeping only the readout direction raises the margin above native); random keep ≤ 0.013 | passes |

## Five-property status
| property | status | next |
|---|---|---|
| Simple | held at head grain (row 6) | — |
| Predicts OOD | fresh panel live/null-beating/selective in all three frames, but the frozen band failed upward on one frame (row 2) | natural rows; a re-frozen band from v71 (1.02 ± 0.15) tested on a further fresh panel |
| Extracted | held at the head boundary (row 5) | source folds |
| Selective | held (rows 1, 3) | — |
| Composes | additive, gap at the bar (row 4) | pairwise + random-split null |

## Receipts
- atlas v68: `bilinear_quotient/circuits/followups/atlas_pronoun_gender_v68_result.json` (724 forwards, rows sha e0c60bf6a3c4…)
- v72: `.../pronoun_gender_dod_random_set_null_v72_result.json`; code `ops/run_pronoun_gender_dod_random_set_null_v72.py`
- v71: `.../pronoun_gender_dod_battery_v71_result.json`; code `ops/run_pronoun_gender_dod_battery_v71.py` (via `ops/dod_battery.py`)
