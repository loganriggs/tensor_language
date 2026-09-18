# Adjective preposition selection in/of — definition-of-done scorecard

Opened 2026-09-18 02:29 UTC (eighth path; Claude lane; a FOURTH readout family candidate). Rubric: `better_circuits.md` §1.

**Path now.** A predicative adjective (interested / afraid) selects its preposition across a parenthetical → readout set
{8.8, 6.3, 13.8, 7.8} along `O_h^T(u_in − u_of)` at the final query (the comma before the preposition) → in/of. The same four
heads were the earlier preposition census's set for on/of (v54), and 13.8 + 7.8 recur in 9 live atlas lines that are all
complement / particle / preposition selections (`READOUT_ATLAS_TABLE.md`, family column "other").

## Claim table

| # | claim | tag | rows | key numbers | status |
|---|---|---|---|---|---|
| 1 | Blind 162-head sweep on the authored rows ranks 8.8 0.66, 6.3 0.64, 13.8 0.53, 7.8 0.30 (next 11.3 0.18); top-4 set 54%, live, selective | edit | opened (atlas v68) | | established |
| 2 | Frozen 0.54 ± 0.15 on fresh rows (16 fresh agents, 16 fresh objects, three unused frames keeping the parenthetical) | edit, frozen | fresh (v85) | fractions remained 0.67 / looked 0.62 / grew 0.49 (pooled 0.57); positive 96/96; null max 0.06; capability 1.00 in all six cells | passes |
| 3 | Selective (was−were, who−which, night−day) | edit | fresh (v85) | moves 0.05 / 0.08 / 0.20 vs null 0.05 / 0.07 / 0.05 — night−day moves 4× its null, inside the damage-scaled gate (0.59); stated | passes by the registered gate |
| 4 | Additive | edit | fresh (v85) | singles 8.8 0.82 / 6.3 0.52 / 13.8 0.44 / 7.8 0.41; gap 0.056 ≤ bar 0.103 | passes |
| 5 | Keep-only readout at the four heads keeps their service | edit | fresh (v85) | zero 2.42; retention 1.08; random keep ≤ 0.04 | passes |
| 6 | Matched-count random four-head-set null (16 quadruples from the other 158 heads, each along its own in−of direction) | edit | v85 rows (v87) | set 2.14 (fraction 0.57) vs random max 0.19 (fraction 0.05); none live | passes 4/4 |
| 7 | Response census of the set removal (exact λ-recurrence split from block 6, closure 9e-6): the set's own attention writes carry only 38% of the linear attribution (attn:13 −0.44, attn:08 −0.25, attn:06 / attn:07 −0.06 each of −2.15); the MLP suffix amplifies (mlp:09 −0.37, mlp:10 −0.28, mlp:15 −0.23, mlp:11 −0.20, mlp:12 −0.17, mlp:08 −0.22; MLP 16/17 counteract +0.15 / +0.18); remainder 0.4% | response | fresh (v88) | registered "direct ≥ 0.80" and "downstream ≤ 0.25 × direct" false | 3/5 — the early heads 6.3 / 7.8 / 8.8 act mostly through MLPs 8–15: a relay, unlike the pronoun sets and unlike the up/down set |

## Five-property status
| property | status | next |
|---|---|---|
| Simple | held at head grain (row 6) | — |
| Predicts OOD | held on a fresh panel with a frozen number (row 2) | natural rows (`dod_natural_line`); a second selection line on the core (verb particle up/down) |
| Extracted | held at the head boundary (row 5); downstream of the heads the effect is MLP-relayed (row 7) | MLP response part is a declared port unless a fold closes it; source fold |
| Selective | held by the gate (row 3), with the night−day move noted | — |
| Composes | additive (row 4) | — |

## Receipts
- atlas v68: `bilinear_quotient/circuits/followups/atlas_adjective_preposition_in_of_v68_result.json`; v54: `.../preposition_selection_dod_reuse_census_v54_result.json`
- v87: `.../selection_dod_random_set_null_v87_result.json`; v88: `.../selection_dod_response_census_v88_result.json`
- v85: `.../selection_dod_battery_v85_result.json`; code `ops/run_selection_dod_battery_v85.py` (via `ops/dod_battery.py`)
