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

## Five-property status
| property | status | next |
|---|---|---|
| Simple | head grain | random four-head-set null |
| Predicts OOD | held on a fresh panel with a frozen number (row 2) | natural rows |
| Extracted | held at the head boundary (row 5) | response census |
| Selective | held (row 3) | — |
| Composes | additive (row 4) | — |

## Receipts
- atlas v68: `bilinear_quotient/circuits/followups/atlas_reflexive_person_v68_result.json`
- v104: `.../person_reflexive_dod_battery_v104_result.json`; code `ops/run_person_reflexive_dod_battery_v104.py` (via `ops/dod_battery.py`)
