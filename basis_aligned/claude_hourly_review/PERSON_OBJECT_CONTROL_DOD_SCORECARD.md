# Reflexive object control me/you — definition-of-done scorecard

Opened 2026-09-18 03:05 UTC (Claude lane; the PERSON readout family). Rubric: `better_circuits.md` §1.

**Path now.** An object pronoun's person (me / you) carried into a controlled infinitive's reflexive → readout set {13.1, 8.1, 10.5, 15.1} along `O_h^T(u_myself − u_yourself)` at the final query → myself/yourself. Five live atlas
lines (reflexive person, object control, their plural forms, possessive my/your) share this set or three of it; 8.1, the temporal
family's token-only cue reader, leads or co-leads — here it reads a person-marked pronoun token. Readers avoid person / number:
will−would, who−which, night−day.

## Claim table

| # | claim | tag | rows | key numbers | status |
|---|---|---|---|---|---|
| 1 | Blind sweep on the authored rows: 13.1 0.73, 8.1 0.57, 10.5 0.48, 15.1 0.44 (next 8.5 0.22); top-4 set 54%, live, selective | edit | opened (atlas v68) | | established |
| 2 | Frozen 0.54 ± 0.15 on fresh rows (16 fresh agents, 16 fresh objects, three unused frames: 'The {agent} by the {obj} warned me/you to protect', 'Yesterday the {agent} reminded me/you to trust', 'Once again the {agent} near the {obj} urged me/you to forgive') | edit, frozen | fresh (v105) | fractions warned 0.44 / reminded 0.43 / urged 0.50 (pooled 0.46); positive 96/96; null max 0.01; capability 1.00 in all six cells | passes |
| 3 | Selective (will−would, who−which, night−day) | edit | fresh (v105) | moves 0.13 / 0.05 / 0.03 vs null 0.05 / 0.06 / 0.04 (gate bar 0.60) | passes |
| 4 | Additive | edit | fresh (v105) | singles 13.1 0.67 / 8.1 0.56 / 10.5 0.53 / 15.1 0.48; gap 0.016 ≤ bar 0.121 | passes |
| 5 | Keep-only readout at the four heads keeps their service | edit | fresh (v105) | zero 1.89; retention 1.29; random keep ≤ 0.02 | passes |
| 6 | Matched-count random four-head-set null | edit | fresh rows (v108) | set 2.23 (fraction 0.46) vs random max 0.15 (fraction 0.03); none live | passes 4/4 |
| 7 | Response census (exact λ-recurrence split from block 8): the set's own writes carry 90% (attn:13 −0.58, attn:15 −0.56, attn:10 −0.48, attn:08 −0.39 of −2.23); downstream net −0.23 (MLP 10 −0.14, MLP 11 −0.12; MLP 17 +0.26); remainder 0.03% | response | fresh (v109) | | passes 5/5 — a direct readout |
| 8 | Source fold (exact, closure 6e-7): the four coefficients come from the me / you token — 8.1 1.01 (95% token-only), 13.1 0.98 (95%), 15.1 0.76 (64%), 10.5 0.52 (28%; 0.22 final, 0.26 other); pooled 0.88 from the pronoun, 0.80 token-only | fold | fresh (v115) | all five registered readings held, including "pooled token-only ≥ 0.50" registered the way v112 came out | 5/5 — the same token-reader structure as the subject-antecedent line: the person family reads the pronoun token, whichever slot it sits in |
| 9 | Token-only generator for {8.1, 13.1, 15.1}: p × λ × v1(me / you) with the native pattern retains 94% of the three heads' zeroed service; a constant pattern per (head, cue) from the other two constructions 94% pooled (warned 0.82, reminded 1.29, urged 0.81); pattern CV ≤ 0.15 | edit | fresh (v117) | | passes 5/5 — three heads close to a two-entry lookup on the pronoun token; 10.5 open |

| 10 | Natural FineWeb rows (two-token cue: me / you followed by 'to' within the window, next token myself / yourself; other person pronouns excluded; the corpus yields 8 me/myself and 16 you/yourself rows in 20000 documents and no counter-cases): congruent removal 2.06 of 5.27 (39%), positive 24/24, null max 0.02, selective (will−would moves 0.17 vs null 0.04, inside the gate) | edit | natural (v155) | v20 bars held; the counter-case prediction unscorable (no rows); me-cell short, recorded not padded | passes 5/5 scorable |
| 11 | Pile rows (1 me/myself + 16 you/yourself): congruent 2.17 of 5.55 (39%), positive 17/17, null 0.06, selective | edit | natural OOD (v156) | | passes 5/5 scorable |

## Five-property status
| property | status | next |
|---|---|---|
| Simple | held at head grain (row 6) | — |
| Predicts OOD | held on fresh and natural FineWeb / Pile rows (rows 2, 10, 11); the natural me-cell is short | — |
| Extracted | held at the head boundary (row 5); direct (row 7); 8.1 / 13.1 / 15.1 close to a token-only generator with constant patterns (rows 8–9); 10.5 contextual, open | 10.5's source |
| Selective | held (row 3) | — |
| Composes | additive (row 4) | — |

## Receipts
- v155 / v156: `.../person_object_control_dod_natural_v155_result.json`, `.../person_object_control_dod_pile_v156_result.json`; rows `person_object_control_dod_{natural,pile}_rows_v15{5,6}.json`; miner config `ops/person_object_control_dod_natural_rows.py`
- v117: `.../person_object_control_dod_token_only_generator_v117_result.json`; code `ops/run_person_object_control_dod_token_only_generator_v117.py` (v113's body)
- v115: `.../person_object_control_dod_source_fold_v115_result.json`; code `ops/run_person_object_control_dod_source_fold_v115.py`
- v108: `.../person_object_control_dod_random_set_null_v108_result.json`; v109: `.../person_object_control_dod_response_census_v109_result.json`
- atlas v68: `bilinear_quotient/circuits/followups/atlas_reflexive_object_control_v68_result.json`
- v105: `.../person_object_control_dod_battery_v105_result.json`; code `ops/run_person_object_control_dod_battery_v105.py` (via `ops/dod_battery.py`)
