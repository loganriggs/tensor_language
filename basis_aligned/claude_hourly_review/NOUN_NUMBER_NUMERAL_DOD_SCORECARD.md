# Numeral number three/one (ones vs one) — definition-of-done scorecard

Opened 2026-09-18 10:44 UTC (Claude lane; the NOUN-NUMBER family's second line — the line that made it the seventh family). Rubric: `better_circuits.md` §1.

**Path now.** A numeral's number (three / one) across an adjective → readout set {11.2, 7.8, 15.1, 10.5} along `O_h^T(u_ones − u_one)` at the adjective
→ ones / one. Shares 11.2, 7.8, 15.1 with the demonstrative line (`NOUN_NUMBER_DEMONSTRATIVE_DOD_SCORECARD.md`). Readers explicit and number-free:
will−would, who−which, night−day.

## Claim table

| # | claim | tag | rows | key numbers | status |
|---|---|---|---|---|---|
| 1 | Blind sweep on the authored rows: 11.2 0.63, 7.8 0.10, 15.1 0.10, 10.5 0.06; top-4 set 22%, live, selective | edit | opened (atlas v68) | | established |
| 2 | Frozen 0.22 ± 0.15 on fresh rows (16 fresh agents, 16 fresh objects — second object pool, 16 fresh adjectives; the v157 frames with the numeral pair) | edit, frozen | fresh (v158) | fractions wanted 0.27 / kept 0.26 / counted 0.26 (pooled 0.26); positive 96/96; null max 0.05; capability 0.94–1.00 | passes |
| 3 | Selective (will−would, who−which, night−day) | edit | fresh (v158) | moves 0.05 / 0.09 / 0.07 vs null 0.04 / 0.07 / 0.06 | passes |
| 4 | Additive | edit | fresh (v158) | singles 11.2 0.55 / 7.8 0.20 / 15.1 0.11 / 10.5 0.07; gap 0.012 ≤ bar 0.019 | passes |
| 5 | Keep-only readout at the four heads keeps their service | edit | fresh (v158) | zero 1.80; retention **0.62** (registered ≥ 0.70); random keep ≤ 0.02 | **failed**: the readout directions carry 62% of the heads' ones/one service |
| 6 | Matched-count random four-head-set null | edit | v158 rows (v161) | set 0.94 (fraction 0.26) vs random max 0.08 (fraction 0.02); none live | passes 4/4 |
| 7 | Response census (exact split from block 7, closure 1e-5): the set's own writes carry 82% (attn:11 −0.56, attn:15 −0.10, attn:07 −0.06, attn:10 −0.05 of −0.94); downstream net −0.17 (MLP 17 −0.13, MLP 16 −0.08; MLP 13 +0.06); remainder 0.8% | response | fresh (v162) | | passes 5/5 — direct, 11.2 dominant |

## Five-property status
| property | status | next |
|---|---|---|
| Simple | held at head grain (row 6) | — |
| Predicts OOD | held on a fresh panel with a frozen number (row 2) | natural rows (numeral + adjective → ones/one is rare; a noun-form line would mine better) |
| Extracted | necessary at the head boundary (rows 2–4); direct (row 7); sufficiency 0.62 (row 5) — open, like the third correlative line | rank-2 keep or declared port |
| Selective | held (row 3) | — |
| Composes | additive (row 4) | — |

## Receipts
- v161: `.../noun_number_numeral_dod_random_set_null_v161_result.json`; v162: `.../noun_number_numeral_dod_response_census_v162_result.json` (emitted by `ops/dod_line.py`)
- atlas v68: `bilinear_quotient/circuits/followups/atlas_numeral_number_v68_result.json`
- v158: `.../noun_number_numeral_dod_battery_v158_result.json`; code `ops/run_noun_number_numeral_dod_battery_v158.py`
