# LATE_TAIL_PRODUCT_TERM_PROBE — preregistration (Registered 2026-09-04 00:45Z (box clock))

Claude, LANE 1 (CUDA, device-explicit). Script ops/late_tail_product_term_probe.py, derived from ops/late_tail_channel_rank_probe.py
(§2779, results sha c5a85ff6… frozen as PRIOR). FRESH split: fits docs 96–191, eval docs 0–63. Sign convention (§2135): all numbers are
CE ADDED ABOVE THE REAL MODEL — LOWER IS BETTER.

## Question
Below the MLP block: the late MLPs' width cost at k = 768 (.1249, MLP reads only) — is it paid in the CROSS terms (tail modulating the core
linearly: Lc·Rt + Lt·Rc) or in the TAIL×TAIL term (Lt·Rt, the low-variance state interacting with itself)? Exact split, bias-free Left/Right.

## Arms (blocks 8–17 MLP reads through the bus core top-k of U_8; rest constant; everything else exact)
- SPLIT8_1024 (instrument, prior .0374), LATE_MLP_768 = core×core only (prior .1249), LATE_MLP_896 (prior .0662)
- FULL_768 = all four terms (exactness check; must be ≈ 0)
- DROP_TT_768 = core×core + cross; DROP_CROSS_768 = core×core + tail×tail; likewise DROP_TT_896, DROP_CROSS_896

## Predictions (bars literal; scored exactly as written)
- pred_a_instrument: baseline 3.0322401 ± 1e-4; SPLIT8_1024, LATE_MLP_768, LATE_MLP_896 within .015 of prior; |FULL_768| ≤ .002.
- pred_b_cross_terms_carry_the_bulk: DROP_CROSS_768 ≥ .070. NULL: DROP_CROSS_768 ≤ .040.
- pred_c_tail_x_tail_is_small: DROP_TT_768 ≤ .040. NULL: DROP_TT_768 ≥ .070.
- pred_d_terms_near_additive: (DROP_TT_768 + DROP_CROSS_768) / LATE_MLP_768 ∈ [0.7, 1.3]. NULL: ratio ≤ 0.5.
- pred_e_tail_x_tail_negligible_at_896: DROP_TT_896 ≤ .015. NULL: DROP_TT_896 ≥ .040.

## Price
1 fit pass (96 docs) + 64 eval docs × 9 forwards ≈ 672 GPU document-forwards; expected ≈ 20 s.
