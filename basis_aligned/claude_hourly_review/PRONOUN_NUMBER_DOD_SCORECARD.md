# Pronoun number they/he — definition-of-done scorecard

Opened 2026-09-18 02:10 UTC (seventh path; Claude lane; the pronoun FAMILY test). Rubric: `better_circuits.md` §1.

**Path now.** Plural vs singular agent noun one clause earlier → readout set {9.6, 12.4, 15.1, 10.5} along
`O_h^T(u_they − u_he)` at the final query → they/he. Shares the core {9.6, 12.4, 15.1} with the pronoun-gender line
(`PRONOUN_GENDER_DOD_SCORECARD.md`); the contrast is they vs he (module vocabulary), so number is confounded with gender on
the singular side — the shared core is the claim under test.

## Claim table

| # | claim | tag | rows | key numbers | status |
|---|---|---|---|---|---|
| 1 | Blind 162-head sweep on the authored rows ranks 9.6 0.81, 12.4 0.28, 15.1 0.15, 10.5 0.14 (next 10.1 0.10); top-4 set 71%, live, selective | edit | opened (atlas v68) | | established |
| 2 | Frozen 0.71 ± 0.15 on fresh rows (16 fresh agent nouns with single-token plurals, 16 fresh objects, three unused frames) | edit, frozen | fresh (v76) | fractions lost 0.75 / because 0.75 / later 0.77 (pooled 0.76); positive 96/96; null max 0.03; capability 1.00 / 0.94–1.00 | passes |
| 3 | Selective (was−were, who−which, night−day) | edit | fresh (v76) | all three within null + 0.25×1.55 | passes |
| 4 | Additive | edit | fresh (v76) | singles 9.6 0.66 / 12.4 0.51 / 15.1 0.21 / 10.5 0.18; gap 0.015 ≤ bar 0.046 | passes |
| 5 | Keep-only readout at the four heads keeps their service | edit | fresh (v76) | zero 1.39; retention 1.53 (keep-only raises the margin above native); random keep ≤ 0.02 | passes |

## Five-property status
| property | status | next |
|---|---|---|
| Simple | head grain; random-set null not run for this line (the gender line's v72 covers the core, not this set) | random four-head-set null |
| Predicts OOD | held on a fresh panel with a frozen number (row 2) | natural FineWeb/Pile rows via `dod_natural_line` |
| Extracted | held at the head boundary (row 5) | response census |
| Selective | held (rows 1, 3) | — |
| Composes | additive (row 4) | — |

## Receipts
- atlas v68: `bilinear_quotient/circuits/followups/atlas_pronoun_number_v68_result.json`
- v76: `.../pronoun_number_dod_battery_v76_result.json`; code `ops/run_pronoun_number_dod_battery_v76.py` (via `ops/dod_battery.py`)
