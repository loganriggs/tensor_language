# Correlative both/neither (and vs nor) — definition-of-done scorecard

Opened 2026-09-18 03:24 UTC (Claude lane; the CORRELATIVE readout family). Rubric: `better_circuits.md` §1.

**Path now.** 'both' obliges 'and', 'neither' obliges 'nor' after the object → readout set {8.1, 7.8, 16.8, 14.8} at the final query (the object noun) → the correlative's second element. Five live atlas
lines share this set or three of it (either/neither, both/either, either/not, either/both, both/neither); 8.1 — the token-only cue reader of
the temporal and person families — leads on a token cue again. Readers as run: will−would, who−which, night−day (import side effect; the
runner's docstring said was−were and was corrected after the run).

## Claim table

| # | claim | tag | rows | key numbers | status |
|---|---|---|---|---|---|
| 1 | Blind sweep on the authored rows: 8.1 0.89, 7.8 0.71, 16.8 0.56, 14.8 0.52 (next 9.7 0.52); top-4 set 47%, live, selective | edit | opened (atlas v68) | | established |
| 2 | Frozen 0.47 ± 0.15 on fresh rows (16 fresh agents, 16 fresh objects from the screened pools, three unused frames: 'Yesterday the {agent} chose CUE the {obj}', 'The {agent} would accept CUE the {obj}', 'At dawn the {agent} bought CUE the {obj}') | edit, frozen | fresh (v120) | fractions accept 0.47 / bought 0.45 / chose 0.48 (pooled 0.46); positive 96/96; null max 0.09; capability ≥ 0.94 in every cell | passes |
| 3 | Selective (will−would, who−which, night−day) | edit | fresh (v120) | moves 0.10 / 0.12 / 0.09 vs null 0.10 / 0.11 / 0.10 | passes |
| 4 | Additive | edit | fresh (v120) | singles 8.1 1.22 / 14.8 0.82 / 16.8 0.55 / 7.8 0.52 = 3.11 vs joint 3.37; gap 0.26 vs bar 0.13 | **failed**: over-additive by 8% of the joint (the have/has shape) |
| 5 | Keep-only readout at the four heads keeps their service | edit | fresh (v120) | zero 3.84; retention 0.96; random keep ≤ 0.01 | passes |
| 6 | Matched-count random four-head-set null | edit | fresh rows (v123) | set 3.37 (fraction 0.46) vs random max 0.10 (fraction 0.01); none live | passes 4/4 |
| 7 | Response census (exact λ-recurrence split from block 7): the set's own writes carry 106% (attn:14 −1.17, attn:08 −0.92, attn:16 −0.72, attn:07 −0.50 of −3.11); downstream net +0.20 (MLPs 8–12 amplify −1.07, MLPs 15–17 push back +1.43); remainder 8% (−0.26, inside the 10% bar) | response | fresh (v124) | | passes 5/5 — a direct readout with a strong late-MLP counter-response |
| 8 | Natural FineWeb rows (both / neither within 12 tokens, next token and / nor; no both/nor rows exist in 20000 docs): congruent removal 3.43 of 7.95 (43%), positive 31/32, null 0.11, selective; per cell neither/nor 6.23 of 4.80 (130% — removal flips the decision), both/and 0.63 of 11.10 (6% — 'and' is the default continuation); the 16 neither/and counter-cases shift toward the text (−1.14) | edit | natural (v127) | v20 bars held | passes 6/6 |
| 9 | Pile rows: congruent 3.04 of 7.18 (42%), positive 30/32, null 0.07, selective; neither/nor 5.59 of 4.24, both/and 0.48 of 10.12; counter-cases −0.58 (n = 18) | edit | natural OOD (v128) | | passes 6/6 |
| 10 | Source fold (exact, closure ≤ 3e-5): every head reads the cue token — 14.8 0.99 (96% token-only), 16.8 1.00 (98%), 8.1 0.99 (85%), 7.8 0.92 (59%); pooled 0.99 from the cue, 0.91 token-only | fold | fresh (v130) | all five registered readings held (8.1 token-only; cue largest for every head; pooled token-only ≥ 0.50) | 5/5 — the purest token-reader family so far: all four heads copy the correlative's first element; the open port is their attention pattern |

## Five-property status
| property | status | next |
|---|---|---|
| Simple | held at head grain (row 6) | — |
| Predicts OOD | held on fresh, natural FineWeb and Pile rows (rows 2, 8, 9): the neither/nor half carries the whole natural decision, both/and little | — |
| Extracted | held at the head boundary (row 5); direct (row 7); all four heads are token readers of the cue (row 10) | token-only generator with constant patterns (v131) |
| Selective | held (row 3) | — |
| Composes | failed the strict bar (row 4) | pairwise terms |

## Receipts
- v130: `.../correlative_either_not_dod_source_fold_v130_result.json`; code `ops/run_correlative_either_not_dod_source_fold_v130.py`
- v127 / v128: `.../correlative_both_neither_dod_natural_v127_result.json`, `.../correlative_both_neither_dod_pile_v128_result.json`; rows `correlative_both_neither_dod_{natural,pile}_rows_v12{7,8}.json`
- v123: `.../correlative_either_not_dod_random_set_null_v123_result.json`; v124: `.../correlative_either_not_dod_response_census_v124_result.json` (runners emitted by `ops/dod_line.py`)
- atlas v68: `bilinear_quotient/circuits/followups/atlas_correlative_pair_v68_result.json`
- v120: `.../correlative_both_neither_dod_battery_v120_result.json`; code `ops/run_correlative_both_neither_dod_battery_v120.py`
