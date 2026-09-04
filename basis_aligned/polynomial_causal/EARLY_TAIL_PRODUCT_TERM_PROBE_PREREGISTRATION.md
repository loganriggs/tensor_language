# EARLY_TAIL_PRODUCT_TERM_PROBE — preregistration (Registered 2026-09-04 00:54Z (box clock))

Claude, LANE 1 (CUDA). Script ops/early_tail_product_term_probe.py, derived from ops/late_tail_gate_rank_probe.py (§2782, results sha
b3b494e1… frozen as PRIOR). FRESH split: fits docs 96–191, eval docs 0–63. Sign convention (§2135): CE ADDED ABOVE THE REAL MODEL —
LOWER IS BETTER.

## Question
Compositionality of the below-the-block finding across depth. In the late blocks the width beyond the 768 frame is a core-gated LINEAR
read of the tail (cross terms .1039 of .1249 = 83%; tail×tail .0087 = 7%; §2780). Do the early blocks (0–7), reading through their OWN
site frames (§2770: .002–.006 per block at 768), use their tail the same way? Because the early cost at 768 is small, the main point is
k = 512 (own frames; tail = 640 dims), with 384 and 768 as the slope.

## Arms (blocks 0–7 MLP reads only; own top-k frame, rest constant; everything else exact)
- SPLIT8_1024 (.0374), LATE_DROP_TT_768 (.0087) — instruments; EARLY_FULL_512 (all terms; must be ≈ 0)
- EARLY_MLP_k (core×core only), EARLY_DROP_TT_k (core×core + cross), EARLY_DROP_CROSS_k (core×core + tail×tail), k ∈ {384, 512, 768}
- shares at k: tt_share = DROP_TT_k / MLP_k; cross_share = DROP_CROSS_k / MLP_k; sum_over_joint = (DROP_TT_k + DROP_CROSS_k) / MLP_k

## Predictions (bars literal; scored exactly as written)
- pred_a_instrument: baseline 3.0322401 ± 1e-4; SPLIT8_1024 and LATE_DROP_TT_768 within .015 of prior; |EARLY_FULL_512| ≤ .002.
- pred_b_early_cross_share_512_large: cross_share(512) ≥ 0.60. NULL: ≤ 0.35.
- pred_c_early_tt_share_512_small: tt_share(512) ≤ 0.30. NULL: ≥ 0.50.
- pred_d_early_cross_share_768_large: cross_share(768) ≥ 0.60. NULL: ≤ 0.35.
- pred_e_early_terms_near_additive_512: sum_over_joint(512) ∈ [0.7, 1.3]. NULL: ≤ 0.5.

## Price
1 fit pass (96 docs) + 64 eval docs × 13 forwards ≈ 928 GPU document-forwards; ≈ 28 s.
