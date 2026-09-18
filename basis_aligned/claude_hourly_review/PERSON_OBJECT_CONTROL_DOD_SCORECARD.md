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

## Five-property status
| property | status | next |
|---|---|---|
| Simple | held at head grain (row 6) | — |
| Predicts OOD | held on a fresh panel with a frozen number (row 2) | natural rows |
| Extracted | held at the head boundary (row 5); direct readout (row 7) | source fold |
| Selective | held (row 3) | — |
| Composes | additive (row 4) | — |

## Receipts
- v108: `.../person_object_control_dod_random_set_null_v108_result.json`; v109: `.../person_object_control_dod_response_census_v109_result.json`
- atlas v68: `bilinear_quotient/circuits/followups/atlas_reflexive_object_control_v68_result.json`
- v105: `.../person_object_control_dod_battery_v105_result.json`; code `ops/run_person_object_control_dod_battery_v105.py` (via `ops/dod_battery.py`)
