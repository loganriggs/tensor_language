# Perfect have/has — definition-of-done scorecard

Opened 2026-09-18 02:43 UTC (tenth path; Claude lane; the number FAMILY's second line at fresh grain). Rubric: `better_circuits.md` §1.

**Path now.** A head noun's number, across a PP intervener, read at the auxiliary → readout set {11.3, 7.8, 5.3, 9.7} along
`O_h^T(u_have − u_has)` at the final query → have/has. Shares {11.3, 7.8, 9.7} with the lexical were/was set of
`NUMBER_DOD_SCORECARD.md` ({11.3, 5.7, 7.8, 9.7}); disjoint answer vocabulary from that line by design (the module's purpose).
Readers avoid number: will−would, who−which, night−day.

## Claim table

| # | claim | tag | rows | key numbers | status |
|---|---|---|---|---|---|
| 1 | Blind sweep on the authored rows ranks 11.3 1.54, 7.8 0.20, 5.3 0.12, 9.7 0.12; top-4 set 66%, live (selectivity failed there only through the number-marked was−were reader) | edit | opened (v50b) | | established |
| 2 | Frozen 0.66 ± 0.15 on fresh rows (16 fresh head nouns with single-token plurals, 16 fresh objects, three unused PP frames) | edit, frozen | fresh (v97) | fractions since 0.65 / lately 0.68 / apparently 0.66 (pooled 0.66); positive 96/96; null max 0.04; capability 0.88–1.00 | passes |
| 3 | Selective (will−would, who−which, night−day) | edit | fresh (v97) | moves 0.07 / 0.13 / 0.07 vs null 0.05 / 0.06 / 0.03 (gate bar 0.58) | passes |
| 4 | **Additive** | edit | fresh (v97) | singles 11.3 1.38 / 7.8 0.26 / 9.7 0.14 / 5.3 0.09 = 1.86 vs joint 2.11; gap 0.26 vs bar 0.021 (0.25 × the smallest single) | **failed**: over-additive by 12% of the joint; 11.3 carries two thirds alone; the bar is strict because 5.3 is tiny |
| 5 | Keep-only readout at the four heads keeps their service | edit | fresh (v97) | zero 2.41; retention 1.03; random keep ≤ 0.02 | passes |
| 6 | Matched-count random four-head-set null | edit | v97 rows (v98) | set 2.11 (fraction 0.66) vs random max 0.08 (fraction 0.03); none live | passes 4/4 |
| 7 | Response census (exact split from block 5, closure 2e-5): the set's own writes carry 60% (attn:11 −1.01, attn:07 −0.14, attn:09 −0.09, attn:05 −0.05 of −2.14); the MLP suffix amplifies −0.85 (mlp:12 −0.17, mlp:16 −0.14, mlp:17 −0.10, mlp:15 −0.08, mlp:13 −0.07, mlp:09 −0.07, mlp:11 −0.06); remainder 1% | response | fresh (v99) | registered "direct ≥ 0.80" and "downstream ≤ 0.25 × direct" false, as the docstring anticipated | 3/5 — a relay through MLPs 9–17, the temporal-family shape |
| 8 | Natural FineWeb rows (a list noun in singular / plural form within 12 tokens, next token have / has; pronouns and auxiliaries excluded from the context): congruent cells (plural/have, singular/has; capability 1.00) removal 0.79 of 7.02 (11%), positive 32/32, null max 0.02, selective | edit | natural (v100) | frozen v20 bars held | passes 5/6 |
| 8f | **Failed:** incongruent rows (a far noun of the other number; capability 0.94 / 1.00 — the model follows the real subject) were predicted to shift toward the text's auxiliary; removal hurts them just as much (+0.77 FineWeb, +1.29 Pile, positive 31/32 and 30/32) | edit | natural (v100, v101) | the set carries the number the model resolved for the actual subject, not the cue noun's — the third and fourth falsification of this reading on number lines (v77, v78) | falsified as registered |
| 9 | Pile rows (out-of-corpus): congruent 1.30 of 5.11 (25%), positive 30/32, null max 0.02, selective | edit | natural OOD (v101) | | passes 5/6 |
| 10 | Pairwise Möbius terms (all six pairs, same removal): the six pair terms sum to 0.27 against the joint−singles gap 0.26 (higher-order terms negligible); pairs with 11.3 carry 82% of the interaction; the largest is 11.3 × 7.8 (+0.12), then 11.3 × 9.7 (+0.06), 11.3 × 5.3 (+0.05); all positive (synergy) | edit | fresh (v102) | registered "pairs without 5.3 pass the pair bar" false: 11.3 × 7.8 (0.12 vs 0.06) and 11.3 × 9.7 (0.06 vs 0.03) exceed it | 4/5 — the over-additivity is a serial relay: 11.3 reads what 7.8 / 9.7 / 5.3 write (the temporal family's 9.1 → 11.3 shape) |

## Five-property status
| property | status | next |
|---|---|---|
| Simple | held at head grain (row 6) | — |
| Predicts OOD | held on fresh, natural FineWeb and Pile rows (rows 2, 8, 9); the counter-case reading failed again: number sets read a resolved subject number (row 8f) | — |
| Extracted | held at the head boundary (row 5); downstream MLP relay (row 7) | MLP response part declared unless a fold closes it |
| Selective | held (row 3) | — |
| Composes | failed the strict bar (row 4); the interaction is pairwise and concentrated on 11.3's pairs (row 10): a serial relay, not an unexplained residue | source / writer fold of 11.3 (v103) |

## Receipts
- v50b: `bilinear_quotient/circuits/followups/perfect_number_dod_reuse_census_v50b_result.json`
- v102: `.../perfect_number_dod_pairwise_v102_result.json`; code `ops/run_perfect_number_dod_pairwise_v102.py`
- v100 / v101: `.../perfect_number_dod_natural_v100_result.json`, `.../perfect_number_dod_pile_v101_result.json`; rows `perfect_number_dod_{natural,pile}_rows_v10{0,1}.json`
- v98: `.../perfect_number_dod_random_set_null_v98_result.json`; v99: `.../perfect_number_dod_response_census_v99_result.json`
- v97: `.../perfect_number_dod_battery_v97_result.json`; code `ops/run_perfect_number_dod_battery_v97.py` (via `ops/dod_battery.py`)
