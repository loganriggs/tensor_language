# LATE_TAIL_CROSS_UNIT_PROBE — preregistration (Registered 2026-09-04 00:48Z (box clock))

Claude, LANE 1 (CUDA). Script ops/late_tail_cross_unit_probe.py, derived from ops/late_tail_product_term_probe.py (§2780, results sha
044ce5ad… frozen as PRIOR). FRESH split: fits docs 96–191, eval docs 0–63. Sign convention (§2135): CE ADDED ABOVE THE REAL MODEL —
LOWER IS BETTER.

## Question
§2780: the late MLPs' width cost beyond 768 is the cross term Lc∘Rt + Lt∘Rc — a core-gated linear read of the tail (.1039 of .1249).
Is that read carried by a few of the 4608 hidden units per block (a narrow sub-circuit), or spread across the hidden layer?

## Method
Fit pass (docs 96–191, exact forward): per late block l and hidden unit j, energy E_lj = mean(cross_j²) · ‖Down[:, j]‖². Rank units per
block. Arms (blocks 8–17 MLP reads through the bus top-768 of U_8, rest constant, tail×tail dropped throughout): LATE_MLP_768 (core×core
only, prior .1249), DROP_TT_768 (all units keep the cross term, prior .0087), CROSS_TOP_h (cross term on the top-h units by E, h = 256 /
512 / 1024 / 2048), CROSS_RAND_1024 (seeded random 1024 units per block). Recovery rec(h) = (LATE_MLP_768 − CROSS_TOP_h) / (LATE_MLP_768 −
DROP_TT_768). Participation ratio PR_l = (ΣE)² / ΣE², averaged over blocks (4608 = uniform).

## Predictions (bars literal; scored exactly as written)
- pred_a_instrument: baseline 3.0322401 ± 1e-4; SPLIT8_1024 (.0374), LATE_MLP_768 (.1249), DROP_TT_768 (.0087) within .015 of prior.
- pred_b_top_1024_units_recover_most_of_the_cross_gain: rec(1024) ≥ 0.60. NULL: rec(1024) ≤ 0.35 (≈ proportional, 1024/4608 = 0.22).
- pred_c_top_512_units_recover_much: rec(512) ≥ 0.40. NULL: rec(512) ≤ 0.20.
- pred_d_random_1024_units_recover_little: rec(RAND_1024) ≤ 0.35. NULL: rec(RAND_1024) ≥ 0.55.
- pred_e_cross_energy_concentrated_on_units: mean PR ≤ 1500. NULL: mean PR ≥ 2500.

## Price
2 fit passes (heads + unit energies, 96 docs each) + 64 eval docs × 9 forwards ≈ 768 GPU document-forwards; ≈ 25 s.
