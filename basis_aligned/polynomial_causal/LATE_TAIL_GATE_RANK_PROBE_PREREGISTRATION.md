# LATE_TAIL_GATE_RANK_PROBE — preregistration (Registered 2026-09-04 00:51Z (box clock))

Claude, LANE 1 (CUDA). Script ops/late_tail_gate_rank_probe.py, derived from ops/late_tail_cross_unit_probe.py (§2781, results sha
c563c548… frozen as PRIOR). FRESH split: fits docs 96–191, eval docs 0–63. Sign convention (§2135): CE ADDED ABOVE THE REAL MODEL —
LOWER IS BETTER.

## Question
§2780: the late MLPs' 768-cost is the core-gated linear read of the tail, Down[L(c)∘Rt + Lt∘R(c)]; §2779: not low-rank in t; §2781: not
sparse in hidden units. Last simple axis: the GATE. How many core directions does the gain vector (L(c), R(c)) of the read depend on?

## Arms (blocks 8–17 MLP reads; core×core exact at 768 through U_8; tail×tail dropped; everything else exact)
- SPLIT8_1024 (.0374), LATE_MLP_768 (core×core only, .1249), DROP_TT_768 (= GATE_768, exact gates, .0087)
- GATE_r: gates L(g), R(g) with g = mx + P_r(xh − mx), P_r the top-r bus directions; r = 0 (constant gates → a FIXED linear read of the
  tail, one D×384 matrix per block), 32, 64, 128, 256, 512.
- recovery rec(r) = (LATE_MLP_768 − GATE_r) / (LATE_MLP_768 − DROP_TT_768).

## Predictions (bars literal; scored exactly as written)
- pred_a_instrument: baseline 3.0322401 ± 1e-4; SPLIT8_1024, LATE_MLP_768, DROP_TT_768 within .015 of prior.
- pred_b_constant_gates_recover_little: rec(0) ≤ 0.30. NULL: rec(0) ≥ 0.60 (the read is essentially a fixed linear map).
- pred_c_256_gate_dims_recover_most: rec(256) ≥ 0.60. NULL: rec(256) ≤ 0.35.
- pred_d_64_gate_dims_recover_much: rec(64) ≥ 0.40. NULL: rec(64) ≤ 0.20.
- pred_e_512_gate_dims_nearly_exact: rec(512) ≥ 0.85. NULL: rec(512) ≤ 0.60.

## Price
1 fit pass (96 docs) + 64 eval docs × 10 forwards ≈ 736 GPU document-forwards; ≈ 22 s.
