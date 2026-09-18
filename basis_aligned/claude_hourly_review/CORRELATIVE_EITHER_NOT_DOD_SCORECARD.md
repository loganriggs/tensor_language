# Correlative either/not (or vs but) — definition-of-done scorecard

Opened 2026-09-18 03:24 UTC (Claude lane; the CORRELATIVE readout family). Rubric: `better_circuits.md` §1.

**Path now.** 'either' obliges 'or', 'not' obliges 'but' after the object → readout set {16.8, 14.8, 7.8, 8.1} at the final query (the object noun) → the correlative's second element. Five live atlas
lines share this set or three of it (either/neither, both/either, either/not, either/both, both/neither); 8.1 — the token-only cue reader of
the temporal and person families — leads on a token cue again. Readers as run: will−would, who−which, night−day (import side effect; the
runner's docstring said was−were and was corrected after the run).

## Claim table

| # | claim | tag | rows | key numbers | status |
|---|---|---|---|---|---|
| 1 | Blind sweep on the authored rows: 16.8 0.55, 14.8 0.47, 7.8 0.41, 8.1 0.40 (next 10.5 0.34); top-4 set 43%, live, selective | edit | opened (atlas v68) | | established |
| 2 | Frozen 0.43 ± 0.15 on fresh rows (16 fresh agents, 16 fresh objects from the screened pools, three unused frames: 'Yesterday the {agent} chose CUE the {obj}', 'The {agent} would accept CUE the {obj}', 'At dawn the {agent} bought CUE the {obj}') | edit, frozen | fresh (v119) | fractions accept 0.46 / bought 0.44 / chose 0.42 (pooled 0.44); positive 96/96; null max 0.06; capability ≥ 0.94 in every cell | passes |
| 3 | Selective (will−would, who−which, night−day) | edit | fresh (v119) | moves 0.07 / 0.22 / 0.09 (who−which moves 3× its null, inside the gate 0.56; stated) vs null 0.05 / 0.07 / 0.06 | passes |
| 4 | Additive | edit | fresh (v119) | singles 14.8 0.51 / 16.8 0.50 / 7.8 0.49 / 8.1 0.42; gap 0.076 vs bar 0.104 | passes |
| 5 | Keep-only readout at the four heads keeps their service | edit | fresh (v119) | zero 2.02; retention 1.14; random keep ≤ 0.05 | passes |
| 6 | Matched-count random four-head-set null | edit | fresh rows (v121) | set 1.99 (fraction 0.44) vs random max 0.20 (fraction 0.05); none live | passes 4/4 |
| 7 | Response census (exact λ-recurrence split from block 7): the set's own writes carry 117% of the linear attribution (attn:14 −0.87, attn:16 −0.69, attn:08 −0.43, attn:07 −0.32 of −1.96); downstream net +0.34 (MLPs 8–10 amplify −0.41, MLPs 15–17 push back +1.03); remainder 1% | response | fresh (v122) | | passes 5/5 — a direct readout with a strong late-MLP counter-response |
| 8 | Natural FineWeb rows (either / not within 12 tokens, next token or / but; any-sense cue): congruent cells removal 1.14 of 5.55 (21%), positive 29/32, null max 0.04, selective; per cell either/or 2.08 of 7.22 (29%, 16/16) but not/but only 0.21 of 3.87 (5%, 13/16) — the any-sense 'not' is a weak cue; counter-case cells shift toward the text (−0.17), as registered | edit | natural (v125) | v20 bars held | passes 6/6 |
| 9 | Pile rows: congruent 1.45 of 5.69 (25%), positive 30/32, null 0.06, selective; either/or 2.47 of 8.46, not/but 0.42 of 2.91; counter-cases −0.06 | edit | natural OOD (v126) | | passes 6/6 |
| 10 | Source fold (exact, closure ≤ 3e-5): every head reads the cue token — 16.8 1.01 (99% token-only), 14.8 0.98 (90%), 8.1 0.97 (76%), 7.8 0.84 (56%); pooled 0.98 from the cue, 0.88 token-only | fold | fresh (v129) | all five registered readings held (8.1 token-only; cue largest for every head; pooled token-only ≥ 0.50) | 5/5 — the purest token-reader family so far: all four heads copy the correlative's first element; the open port is their attention pattern |

## Five-property status
| property | status | next |
|---|---|---|
| Simple | held at head grain (row 6) | — |
| Predicts OOD | held on fresh, natural FineWeb and Pile rows (rows 2, 8, 9), the either/or half strongly and the not/but half weakly; the set follows the cue token | — |
| Extracted | held at the head boundary (row 5); direct (row 7); all four heads are token readers of the cue (row 10) | token-only generator with constant patterns (v131) |
| Selective | held (row 3) | — |
| Composes | additive (row 4) | — |

## Receipts
- v129: `.../correlative_either_not_dod_source_fold_v129_result.json`; code `ops/run_correlative_either_not_dod_source_fold_v129.py`
- v125 / v126: `.../correlative_either_not_dod_natural_v125_result.json`, `.../correlative_either_not_dod_pile_v126_result.json`; rows `correlative_either_not_dod_{natural,pile}_rows_v12{5,6}.json`; miner config `ops/correlative_dod_natural_rows.py`
- v121: `.../correlative_either_not_dod_random_set_null_v121_result.json`; v122: `.../correlative_either_not_dod_response_census_v122_result.json` (runners emitted by `ops/dod_line.py`)
- atlas v68: `bilinear_quotient/circuits/followups/atlas_correlative_disjoint_either_not_v68_result.json`
- v119: `.../correlative_either_not_dod_battery_v119_result.json`; code `ops/run_correlative_either_not_dod_battery_v119.py`
