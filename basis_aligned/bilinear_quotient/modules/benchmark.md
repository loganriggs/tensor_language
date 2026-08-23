# The understanding benchmark — per-module scores, valid stand-ins, measurement rules

Scale: 0 = mean-ablate, 1 = full model. The VALID measure is PER-MODULE-IN-ISOLATION —
per-module understanding does NOT compose (whole-model greedy substitution 12%,
compounding-dominated; content passthrough 0.39; §1070-1071). Whole-model simultaneous held-out:
**0.32 ± 0.06** (draw-sensitive; §1013-1014).

## Current per-module scores + stand-ins
| Module | Score | Valid stand-in | Ref |
|---|---|---|---|
| attn L0 (all heads) | ~1.00 | token+position pattern + per-token value table | sink arc |
| head 5.7 | ~0.985 | ONE fixed vector (global mean) | §1089/§1091 |
| other front routers L0H3/L1H1/L2H5 | 0.7-0.95 | residual-window routing | §1054/§1091 |
| mlp0 | ~0.90 | class-code writer / token map | §905/§1045 |
| mlp1 | ~0.93 | static per-token table (held-out) | §1088 |
| mlp3 | **0.95 (r256) / 0.83 (r64)** certified | own-basis projection (held-out) | §1130 |
| mlp4 | 0.82 (r256) / 0.48 (r64) certified | own-basis projection; token table HURTS held-out | §1130 |
| mlp5-14 (content) | ~0.10 as variables | none — real high-rank (3-way confirmed) | §1000/38/42 |
| middle attn (collective) | 0.58 partial | static distance-kernel (values dynamic); kernel+content-sim next | §1099 |
| readout mlp16/17 | **0.81/0.84 certified** (mlp15 0.40, tiny stakes) | fitted linear read; CEILING: top-256 own-neurons also 0.81 — functional tail is neither sparse nor linear | §1131-1132 |
| block-17 calibration | ~1.00 | rank-1 w_freq | §650-651 |
| the merge | understood | additive-linear (−W·c, cos 0.77) | §1082/§1086 |

## Measurement rules (violations produced retractions — check every design against these)
1. HELD-OUT everything: in-sample per-token means leak on singletons (~20% of positions)
   (§901 0.81→0.30; §1088 deep tok-recovery 0.59→0.01-0.14).
2. Matched nulls: keep-only needs shuffled-label matched-rank nulls (§836); random-subspace is
   far too weak (§821).
3. Perturbation-size controls: norm-match any removal before claiming direction-specificity
   (§1086); low K for specificity (destructive regime at K=256; §1067-1068).
4. Partial ≠ share: partial removals of baselines/constants cost MORE than full removal
   (§1087/§1091/§1093) — never read fractions off partials.
5. Per-part sums ≠ collective: measure both granularities (5.7× gap, §1093; middle band 4×,
   §813; front-6 §952).
6. Stand-in must match the removal point/scale (§1066).
7. Banding vs output-ablation flip on redundant parts (§1008-1009); firing ≠ function (§726).
8. Draw-sensitivity: report understanding as a band (±0.06), not a point (§1013-1014).

## The target
90% per module bottom-up (user directive). Solved: front, readout, sink, L0-attention,
calibration. Bounded by real model properties: deep-middle content (high-rank). Genuinely open:
mlp4's variable, the middle-attention collective pool.
