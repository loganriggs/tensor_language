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
| 10 | Position-grain fold (v81): 9.6 noun 0.48 / verb 0.50; 10.5 noun 0.31 / verb 0.34; 12.4 noun 0.24 / verb 0.32 / det-before-object 0.15 / final 0.12; 15.1 noun 0.31 / verb 0.17 / final 0.20 / conj 0.15 | fold | fresh (v81) | registered "noun + verb ≥ 0.60 for every head" false (12.4 0.56, 15.1 0.48): number is spread over the clause for the later heads | 3/5 (shared receipt with the gender line) |
| 11 | Writer fold of what 9.6 reads (v82, closure 2e-7): the number feature at the verb is 94% MLP-written (MLP 8 0.54, MLP 6 0.11, MLP 4 0.10, MLP 5 0.08), heads 6%; at the noun MLP 87% (MLP 8 0.50), embedding 10%, heads 3% — no attention copier; the plural morphology is resolved by the MLP stack at the noun and again at the verb | fold | fresh (v82) | | MLP 8 is the declared port on this line too |
| 12 | Pair-term fold of MLP 8's write (v83, closure 4e-4): diffuse products of MLPs 4–7 (largest pair 7%; pairs with MLP 6/7 0.41 at the verb, 0.58 at the noun); registered "MLP 6/7 pairs ≥ 0.50 at the verb" false | fold | fresh (v83) | | 1/5 — **declared port**: MLP 8 (fed by MLPs 4–7) |
| 13 | Response census of the set removal (λ-recurrence exact split, blocks 9–17, closure 2e-5): the set's own attention writes carry 103% of the linear attribution (attn:09 −0.60, attn:12 −0.56, attn:15 −0.24, attn:10 −0.20 of −1.56); downstream net +0.04 (MLP 17 +0.17 counteracts, MLP 9 −0.09 amplifies); remainder 0.4% | response | fresh (v84) | | passes 5/5 — a direct readout, like the gender set |
| 14 | Shared number axis? Removal at the same four heads along the verb-agreement direction `O_h^T(u_were − u_was)` (weight-level |cos| 0.4 with they−he at 9.6 / 15.1, v95) on the v76 fresh rows: 0.28 logits (14% of the margin; 18% of the own-direction 1.55), positive 91/96, above the null (0.02), live; the gender direction he−she removes 0.09 | edit | fresh (v96) | registered "were−was carries ≥ 50% of own" false; live/null-beating and gender < number held; own replay 0.76 exact | 4/5 — a small shared number component, not one axis |
| 15 | Shared heads, separate directions: at the same four heads the PERSON family's direction O_h^T(u_myself − u_yourself) (the person set shares 15.1 and 10.5) removes 0.00 of the they/he margin on the v76 rows (positive 45/96 — noise), own direction 1.55 (replay of v76), null max 0.01 | edit | fresh (v134) | | passes 5/5 — the two families use the same heads along orthogonal readout directions |
| 16 | MLP 8 at unit grain on 9.6's they−he reader direction at the noun (exact, closure 8e-5): unit 829 carries 39% of the plural−singular contrast, 953 17%, 1030 9%, 1484 5%, 3152 (the gender unit) 4.6%, 1738 −4%; top-10 78%, top-50 89%, top-500 97%; frame-wise top-50 Jaccard ≥ 0.70 | fold | fresh (v168) | all four registered readings held | 4/4 — the number feature 9.6 reads has its own MLP-8 units (829, 953, 1030), distinct from the gender unit 3152 (rank 5 here, 4.6%); edit test v169 |

## Five-property status
| property | status | next |
|---|---|---|
| Simple | held at head grain (row 6) | — |
| Predicts OOD | held on fresh, natural FineWeb and Pile rows for the congruent readout (rows 2, 7, 8); the registered counter-case prediction was wrong twice (row 7f): the set reads a resolved number, not the noun's (row 9 agrees) | — |
| Extracted | held at the head boundary (row 5); sources folded to the MLP stack 4–8 at the noun and verb, which does not close by pair folding (rows 9–12) | MLP 8 declared port; readout direct (row 13) — closed at head grain |
| Selective | held (rows 1, 3) | — |
| Composes | additive (row 4) | — |

## Receipts
- atlas v68: `bilinear_quotient/circuits/followups/atlas_pronoun_number_v68_result.json`
- v168 (MLP 8 unit census): `.../pronoun_number_dod_mlp8_unit_census_v168_result.json`; code `ops/run_pronoun_number_dod_mlp8_unit_census_v168.py`
- v134: `.../pronoun_number_dod_cross_family_direction_v134_result.json`; code `ops/run_pronoun_number_dod_cross_family_direction_v134.py`
- v96: `.../pronoun_number_dod_shared_number_axis_v96_result.json`; code `ops/run_pronoun_number_dod_shared_number_axis_v96.py`
- v84: `.../pronoun_number_dod_response_census_v84_result.json`; code `ops/run_pronoun_number_dod_response_census_v84.py`
- v83: `.../pronoun_dod_mlp8_pair_fold_v83_result.json`
- v82: `.../pronoun_dod_verb_writer_fold_v82_result.json`
- v81: `.../pronoun_dod_source_fold_positions_v81_result.json`
- v80 (both lines): `.../pronoun_dod_source_fold_v80_result.json`
- v79: `.../pronoun_number_dod_random_set_null_v79_result.json` (scored key `pred_d_set_fraction_within_band_of_v71` is v72's literal name reused; the band is v76's 0.76 ± 0.05)
- v77 / v78: `.../pronoun_number_dod_natural_v77_result.json`, `.../pronoun_number_dod_pile_v78_result.json`; rows `..._natural_rows_v77.json`, `..._pile_rows_v78.json`; miner `ops/pronoun_number_dod_natural_rows.py`; body `ops/dod_natural_line.py`
- v76: `.../pronoun_number_dod_battery_v76_result.json`; code `ops/run_pronoun_number_dod_battery_v76.py` (via `ops/dod_battery.py`)
