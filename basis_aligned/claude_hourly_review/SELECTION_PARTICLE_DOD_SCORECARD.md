# Verb particle selection up/down — definition-of-done scorecard

Opened 2026-09-18 02:31 UTC (ninth path; Claude lane; the selection FAMILY test). Rubric: `better_circuits.md` §1.

**Path now.** A verb (woke / calmed) selects its particle across a parenthetical → readout set {13.8, 14.8, 7.8, 8.8} along
`O_h^T(u_up − u_down)` at the final query → up/down. Three heads shared with the adjective-preposition set of
`SELECTION_DOD_SCORECARD.md` ({8.8, 6.3, 13.8, 7.8}).

## Claim table

| # | claim | tag | rows | key numbers | status |
|---|---|---|---|---|---|
| 1 | Blind sweep on the authored rows ranks 13.8 0.69, 14.8 0.55, 7.8 0.27, 8.8 0.27 (next 8.1 0.25, 6.3 0.21); top-4 set 52%, live, selective | edit | opened (atlas v68) | | established |
| 2 | Frozen 0.52 ± 0.15 on fresh rows (16 fresh agents, 16 fresh objects, the v85 frames with the verb pair) | edit, frozen | fresh (v86) | fractions remained 0.43 / looked 0.47 / grew 0.40 (pooled 0.43); positive 96/96; null max 0.09; capability 1.00 in all six cells | passes |
| 3 | Selective (was−were, who−which, night−day) | edit | fresh (v86) | moves 0.02 / 0.08 / 0.09 vs null 0.07 / 0.07 / 0.05 | passes |
| 4 | Additive | edit | fresh (v86) | singles 13.8 0.70 / 14.8 0.40 / 7.8 0.33 / 8.8 0.29; gap 0.036 ≤ bar 0.074 | passes |
| 5 | Keep-only readout at the four heads keeps their service | edit | fresh (v86) | zero 1.93; retention 0.97; random keep ≤ 0.03 | passes |
| 6 | Matched-count random four-head-set null | edit | v86 rows (v89) | set 1.76 (fraction 0.43) vs random max 0.09 (fraction 0.02); none live | passes 4/4 |
| 7 | Response census (exact split from block 7, closure 1e-5): the set's own writes carry 86% (attn:13 −0.69, attn:14 −0.41, attn:08 −0.22, attn:07 −0.19 of −1.75); downstream net −0.25 (MLPs 8–10 −0.18 together, MLP 17 +0.06); remainder 0.5% | response | fresh (v90) | | passes 5/5 — direct, like the pronoun sets |
| 8 | Natural FineWeb rows (woke / calmed within 12 tokens, next token up / down; the corpus yields 28 rows, no incongruent cells): congruent removal 1.48 of 8.08 (18%), positive 28/28, null max 0.08, selective; woke/up 1.35 of 9.06, calmed/down 1.65 of 6.78 | edit | natural (v93) | frozen v20 bars held; incongruent prediction unscorable (no rows) | passes 5/5 scorable |
| 9 | Pile rows (21 congruent + 1 incongruent): congruent 1.57 of 8.36 (19%), positive 21/21, null max 0.08, selective; the single incongruent row (woke … down) shifted away (+0.16) — registered ≤ 0 false on n = 1 | edit | natural OOD (v94) | | passes 5/6 |

## Five-property status
| property | status | next |
|---|---|---|
| Simple | held at head grain (row 6) | — |
| Predicts OOD | held on fresh, natural FineWeb and Pile rows by the registered bars (rows 2, 8, 9); natural fraction 18–19% against the panel's 43% | — |
| Extracted | held at the head boundary (row 5); direct readout (row 7) | source fold |
| Selective | held (row 3) | — |
| Composes | additive (row 4) | — |

## Receipts
- atlas v68: `bilinear_quotient/circuits/followups/atlas_verb_particle_up_down_v68_result.json`
- v93 / v94: `.../selection_updown_dod_natural_v93_result.json`, `.../selection_updown_dod_pile_v94_result.json`; rows `selection_updown_dod_{natural,pile}_rows_v9{3,4}.json`
- v89: `.../selection_particle_dod_random_set_null_v89_result.json`; v90: `.../selection_particle_dod_response_census_v90_result.json`
- v86: `.../selection_particle_dod_battery_v86_result.json`; code `ops/run_selection_particle_dod_battery_v86.py`
