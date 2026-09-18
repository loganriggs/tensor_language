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

## Five-property status
| property | status | next |
|---|---|---|
| Simple | head grain | random four-head-set null |
| Predicts OOD | held on a fresh panel with a frozen number (row 2) | natural rows |
| Extracted | held at the head boundary (row 5) | response census; source fold |
| Selective | held (row 3) | — |
| Composes | additive (row 4) | — |

## Receipts
- atlas v68: `bilinear_quotient/circuits/followups/atlas_correlative_disjoint_either_not_v68_result.json`
- v119: `.../correlative_either_not_dod_battery_v119_result.json`; code `ops/run_correlative_either_not_dod_battery_v119.py`
