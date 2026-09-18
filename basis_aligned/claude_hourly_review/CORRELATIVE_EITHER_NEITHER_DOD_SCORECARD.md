# Correlative either/neither (or vs nor) — definition-of-done scorecard

Opened 2026-09-18 04:34 UTC (Claude lane; the correlative family's third line). Rubric: `better_circuits.md` §1.

**Path now.** 'either' obliges 'or', 'neither' obliges 'nor' after the object → readout set {14.8, 8.1, 16.8, 5.7} along `O_h^T(u_or − u_nor)`
at the object. Three heads shared with the two batteried correlative lines ({16.8, 14.8, 7.8, 8.1}); 5.7 (a number-core head) replaces 7.8 here.
Readers set explicitly: was−were, who−which, night−day.

## Claim table

| # | claim | tag | rows | key numbers | status |
|---|---|---|---|---|---|
| 1 | Blind sweep on the authored rows: 14.8 0.80, 8.1 0.78, 16.8 0.50, 5.7 0.39 (next 7.8 0.28); top-4 set 55%, live, selective | edit | opened (atlas v68) | | established |
| 2 | Frozen 0.55 ± 0.15 on fresh rows (16 fresh agents, 16 fresh objects from the pools, the v119 frames with the either/neither pair) | edit, frozen | fresh (v147) | fractions accept 0.54 / bought 0.53 / chose 0.55 (pooled 0.54); positive 96/96; null max 0.23 (the largest null in the collection: the or/nor margin is sensitive to any direction at these heads); capability 1.00 | passes |
| 3 | Selective (was−were, who−which, night−day) | edit | fresh (v147) | moves 0.42 / 0.61 / 0.26 vs null 0.30 / 0.37 / 0.28 — inside the damage-scaled gates (bar 0.90–0.97) but who−which moves 1.6× its null; stated | passes by the registered gate |
| 4 | Additive | edit | fresh (v147) | singles 14.8 0.81 / 8.1 0.63 / 16.8 0.49 / 5.7 0.48; gap 0.023 ≤ bar 0.121 | passes |
| 5 | Keep-only readout at the four heads keeps their service | edit | fresh (v147) | zero 2.68; retention **0.53** (registered ≥ 0.70); random keep ≤ 0.04 | **failed**: the four readout directions carry about half of the heads' or/nor service; the rest of what these heads write matters for or/nor off those directions |
| 6 | Rank-2 keep (v58b's in-forward `keep_span`): the or−nor readout direction alone retains 0.525 (replay of row 5 within 0.001); or−nor + and−nor per head 0.60; 16 random rank-2 spans ≤ 0.13 | edit | fresh (v148) | registered "rank-2 ≥ 0.70" false | 2/3 — the two correlative directions carry 60% of the heads' or/nor service; the rest is off both; per the registered kill criterion the line is left as "necessary at the head boundary, sufficiency open" |
| 7 | Natural FineWeb rows (either / neither within 12 tokens, next token or / nor; the corpus has one either…nor row in 20000 docs): congruent removal 2.52 of 5.93 (43%), positive 32/32, null max 0.20, selective; either/or 2.15 of 9.48 (23%), neither/nor 2.89 of 2.37 (122% — the set is the whole 'nor' decision); the 16 neither…or counter-cases shift toward the text (−0.73) as registered | edit | natural (v149) | v20 bars held | passes 6/6 |
| 8 | Pile rows: congruent 2.22 of 4.62 (48%), positive 32/32, null 0.21, selective; either/or 27%, neither/nor 114%; counter-cases −1.00 | edit | natural OOD (v150) | | passes 6/6 |

## Five-property status
| property | status | next |
|---|---|---|
| Simple | head grain | random four-head-set null |
| Predicts OOD | held on fresh, natural FineWeb and Pile rows (rows 2, 7, 8); the set follows the cue token (counter-cases toward the text) | — |
| Extracted | necessary at the head boundary (rows 2–4); rank-1 keep 0.53, rank-2 (with and−nor) 0.60, random rank-2 0.13 (rows 5–6) — sufficiency open, declared | — |
| Selective | held by the gate with a large who−which move noted (row 3) | — |
| Composes | additive (row 4) | — |

## Receipts
- v149 / v150: `.../correlative_either_neither_dod_natural_v149_result.json`, `.../correlative_either_neither_dod_pile_v150_result.json`; rows `correlative_either_neither_dod_{natural,pile}_rows_v1{49,50}.json`
- v148: `.../correlative_either_neither_dod_keep_rank2_v148_result.json`; code `ops/run_correlative_either_neither_dod_keep_rank2_v148.py`
- atlas v68: `bilinear_quotient/circuits/followups/atlas_correlative_state_v68_result.json`
- v147: `.../correlative_either_neither_dod_battery_v147_result.json`; code `ops/run_correlative_either_neither_dod_battery_v147.py`
