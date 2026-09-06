# Terminal-evidence table — six head sets (generated 2026-09-06 23:56 UTC)

Rubric rows (circuits/TIER_RUBRIC.md): row 2 extraction ≥0.80 (LB ≥0.60); row 3 selective removal LB>0 with specificity; row 4 off-target UB ≤0.01 nat; row 5 OOD (A2) keeps sign, ≥50% of ID. Removal = block diff-in-means (rank 1 per block) set to its background coordinate at the prediction position; CE damage in nats over base+donor sentences (64 docs), 97.5% document bootstrap. All-pos = answer-CE damage when the direction is removed at every position / final-only. Per-token C = 97.5% UB of the per-token CE increase over the 32 control sentences under all-position removal. Cross = worst of the other sets' A1 answer-CE shift (one-sided); cross-tok = worst per-token UB. Sorted by removal damage.

| set | heads | removal (LB) | extraction (LB) | all-pos ratio | per-token C UB | cross ans | cross tok UB | A2 | C ans | rows met | source |
|---|---|---|---|---|---|---|---|---|---|---|---|
| quantifier_number (enlarged v54) | 6 | 0.592 (0.505) | 0.720 (0.681) | — | — | 0.005 | — | 0.668 | -0.121 | 3,4,5 | v54/v59 |
| verb_complementizer | 3 | 0.583 (0.497) | 0.585 (0.569) | 1.13 | 0.001 | 0.008 | 0.022 | 0.311 | -0.001 | 3,4,5 | v57/v60 |
| dative (enlarged v54) | 9 | 0.365 (0.314) | 0.763 (0.702) | — | — | 0.020 | — | 0.165 | -0.083 | 3,4 | v54/v59 |
| polarity_licensing (enlarged v54) | 7 | 0.361 (0.235) | 0.773 (0.748) | — | — | 0.037 | — | 0.409 | -0.008 | 3,4,5 | v54/v59 |
| quantifier_number | 2 | 0.355 (0.300) | 0.511 (0.492) | 1.04 | -0.007 | -0.005 | -0.003 | 0.397 | -0.116 | 3,4,5 | v51/v52/v56 |
| dative | 5 | 0.259 (0.220) | 0.589 (0.547) | 0.91 | -0.002 | 0.018 | 0.000 | 0.144 | -0.111 | 3,4,5 | v51/v52/v56 |
| voice_frame (enlarged v54) | 8 | 0.208 (0.135) | 0.782 (0.727) | — | — | 0.036 | — | 0.331 | 0.040 | 3,5 | v54/v59 |
| verb_preposition | 3 | 0.201 (0.175) | 0.584 (0.569) | 1.10 | 0.005 | 0.013 | 0.020 | 0.152 | -0.114 | 3,4,5 | v57/v60 |
| polarity_licensing | 4 | 0.173 (0.101) | 0.597 (0.579) | 1.18 | 0.001 | 0.034 | 0.022 | 0.288 | -0.015 | 3,4,5 | v51/v52/v56 |
| voice_frame | 4 | 0.098 (0.057) | 0.603 (0.559) | 1.13 | -0.005 | 0.021 | 0.017 | 0.220 | -0.094 | 3,4,5 | v51/v52/v56 |

Notes: row 2 is not met by any heads-only set (0.51–0.60; enlarged sets 0.72–0.78 held-out, v54). Negative C/cross values are improvements of the control answer under removal (v51/v56 caveat: a two-sided bar would flag quantifier→dative −0.12). "rows met" is computed from this table only (row 4 here = C answer shift ≤0.01 one-sided; per-token row-4 passes for all six original sets). Mechanism dossiers: polarity (POLARITY_MECHANISM_2026-09-06.md, Tier 4: write-gated readout, product formed in mlp 12–17 as a dense bilinear form, v42–v55); quantifier (v43/v58: parallel products at mlp 15 and 16, mlp 17 damps).
