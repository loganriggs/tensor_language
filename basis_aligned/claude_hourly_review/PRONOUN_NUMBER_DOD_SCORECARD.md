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
| 6 | Matched-count random four-head-set null (16 quadruples from the other 158 heads, each along its own they−he direction) | edit | v76 rows (v79) | set 1.55 (fraction 0.76, replay of v76) vs random max 0.08 (fraction 0.04); none live | passes 4/4 |
| 7 | Natural FineWeb rows (outcome-blind miner, agent noun singular/plural within 12 tokens, next token they/he, 16 per cell): congruent rows (plural/they, singular/he; capability 0.94 / 0.88) removal 2.27 of 3.49 (65%), positive 30/32, null max 0.07, selective by the registered gate | edit | natural (v77) | was−were moved 0.23 vs null 0.04 (gate bar 0.60): a number readout touches the number reader far above null, passes only because the gate scales with damage | passes 5/6 |
| 7f | **Failed:** incongruent rows (singular noun then 'they', plural noun then 'he'; capability 0.69 / 0.75) were predicted to shift toward the text's pronoun under removal; they shift away (+0.77, positive 18/32) | edit | natural (v77) | the set carries the number the model itself resolved (usually the text's), not the surface noun's number — unlike the gender set, whose incongruent rows moved toward the text | falsified as registered |
| 8 | Pile rows (out-of-corpus, same miner and bars): congruent 2.04 of 3.61 (56%), positive 30/32, null max 0.02, selective by the gate (was−were 0.20 vs null 0.03); incongruent again shift away (+0.99, positive 24/32; capability 0.81 / 0.94) | edit | natural OOD (v78) | | passes 5/6, pred_f falsified again |
| 9 | Source fold on the v76 fresh rows (closure 3e-5): every head reads the number-marked noun at 24–48% and the other positions at 40–57%, almost entirely through the contextual branch (inherited 2–11%); no token-only reader in this set | fold | fresh (v80) | pooled noun share 0.34 (gender line 0.61), inherited 0.05 | registered "number noun share below gender" held; consistent with the falsified counter-case (row 7f): the set reads a resolved, contextual number |

## Five-property status
| property | status | next |
|---|---|---|
| Simple | held at head grain (row 6) | — |
| Predicts OOD | held on fresh, natural FineWeb and Pile rows for the congruent readout (rows 2, 7, 8); the registered counter-case prediction was wrong twice (row 7f): the set reads a resolved number, not the noun's (row 9 agrees) | — |
| Extracted | held at the head boundary (row 5); sources folded, contextual (row 9) | response census; finer position split (v81) |
| Selective | held (rows 1, 3) | — |
| Composes | additive (row 4) | — |

## Receipts
- atlas v68: `bilinear_quotient/circuits/followups/atlas_pronoun_number_v68_result.json`
- v80 (both lines): `.../pronoun_dod_source_fold_v80_result.json`
- v79: `.../pronoun_number_dod_random_set_null_v79_result.json` (scored key `pred_d_set_fraction_within_band_of_v71` is v72's literal name reused; the band is v76's 0.76 ± 0.05)
- v77 / v78: `.../pronoun_number_dod_natural_v77_result.json`, `.../pronoun_number_dod_pile_v78_result.json`; rows `..._natural_rows_v77.json`, `..._pile_rows_v78.json`; miner `ops/pronoun_number_dod_natural_rows.py`; body `ops/dod_natural_line.py`
- v76: `.../pronoun_number_dod_battery_v76_result.json`; code `ops/run_pronoun_number_dod_battery_v76.py` (via `ops/dod_battery.py`)
