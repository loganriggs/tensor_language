# Reflexive person I/you — definition-of-done scorecard

Opened 2026-09-18 03:05 UTC (Claude lane; the PERSON readout family). Rubric: `better_circuits.md` §1.

**Path now.** A subject pronoun's person (I / you) one clause earlier → readout set {8.1, 13.1, 10.5, 15.1} along `O_h^T(u_myself − u_yourself)` at the final query → myself/yourself. Five live atlas
lines (reflexive person, object control, their plural forms, possessive my/your) share this set or three of it; 8.1, the temporal
family's token-only cue reader, leads or co-leads — here it reads a person-marked pronoun token. Readers avoid person / number:
will−would, who−which, night−day.

## Claim table

| # | claim | tag | rows | key numbers | status |
|---|---|---|---|---|---|
| 1 | Blind sweep on the authored rows: 8.1 0.71, 13.1 0.62, 10.5 0.58, 15.1 0.40 (next 5.7 0.30); top-4 set 46%, live, selective | edit | opened (atlas v68) | | established |
| 2 | Frozen 0.46 ± 0.15 on fresh rows (16 fresh agents, 16 fresh objects, three unused frames: 'After the {obj} fell, I/you clearly blamed', 'The {agent}s say that I/you rarely trust', 'By the {obj} I/you then reminded') | edit, frozen | fresh (v104) | fractions after 0.47 / by 0.40 / say 0.40 (pooled 0.42); positive 96/96; null max 0.02; capability 1.00 in all six cells | passes |
| 3 | Selective (will−would, who−which, night−day) | edit | fresh (v104) | moves 0.10 / 0.04 / 0.05 vs null 0.05 / 0.06 / 0.04 (gate bar 0.62) | passes |
| 4 | Additive | edit | fresh (v104) | singles 13.1 0.73 / 8.1 0.63 / 10.5 0.48 / 15.1 0.37; gap 0.055 ≤ bar 0.092 | passes |
| 5 | Keep-only readout at the four heads keeps their service | edit | fresh (v104) | zero 2.03; retention 1.21; random keep ≤ 0.02 | passes |
| 6 | Matched-count random four-head-set null | edit | fresh rows (v106) | set 2.27 (fraction 0.42) vs random max 0.10 (fraction 0.02); none live | passes 4/4 |
| 7 | Response census (exact λ-recurrence split from block 8): the set's own writes carry 85% (attn:13 −0.61, attn:15 −0.44, attn:08 −0.43, attn:10 −0.43 of −2.24); downstream net −0.33 (MLP 10 −0.23, MLP 11 −0.11; MLP 17 +0.30 counteracts); remainder 1% | response | fresh (v107) | | passes 5/5 — a direct readout |
| 8 | Natural FineWeb rows (an I / you token within 12 tokens, next token myself / yourself; other person pronouns excluded): congruent cells (I/myself, you/yourself; capability 1.00) removal 2.18 of 5.93 (37%), positive 32/32, null max 0.03, selective (will−would moves 0.14 vs null 0.04, inside the gate); the 11 counter-case rows (a quoted speaker) shift toward the text's reflexive (−0.23) as registered | edit | natural (v110) | frozen v20 bars held | passes 6/6 |
| 9 | Pile rows (out-of-corpus): congruent 2.07 of 6.29 (33%), positive 32/32, null max 0.03, selective; 4 counter-case rows −0.05 | edit | natural OOD (v111) | | passes 6/6 |

## Five-property status
| property | status | next |
|---|---|---|
| Simple | held at head grain (row 6) | — |
| Predicts OOD | held on fresh, natural FineWeb and Pile rows (rows 2, 8, 9); the set follows the cue token (counter-cases move toward the text, as the gender set did and the number sets did not) | — |
| Extracted | held at the head boundary (row 5); direct readout (row 7) | source fold |
| Selective | held (row 3) | — |
| Composes | additive (row 4) | — |

## Receipts
- v110 / v111: `.../person_reflexive_dod_natural_v110_result.json`, `.../person_reflexive_dod_pile_v111_result.json`; rows `person_reflexive_dod_{natural,pile}_rows_v11{0,1}.json`; miner config `ops/person_dod_natural_rows.py`
- v106: `.../person_reflexive_dod_random_set_null_v106_result.json`; v107: `.../person_reflexive_dod_response_census_v107_result.json`
- atlas v68: `bilinear_quotient/circuits/followups/atlas_reflexive_person_v68_result.json`
- v104: `.../person_reflexive_dod_battery_v104_result.json`; code `ops/run_person_reflexive_dod_battery_v104.py` (via `ops/dod_battery.py`)
