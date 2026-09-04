# ALL18_TAIL_LINEAR_PROGRAM_PROBE — preregistration (Registered 2026-09-04 00:56Z (box clock))

Claude, LANE 1 (CUDA). Script ops/all18_tail_linear_program_probe.py, derived from ops/early_tail_product_term_probe.py (§2783, results
sha 02a246b2… frozen as PRIOR). FRESH split: fits docs 96–191, eval docs 0–63. Sign convention (§2135): CE ADDED ABOVE THE REAL MODEL —
LOWER IS BETTER.

## Question
§2780 + §2783: at every depth the MLP's out-of-frame arithmetic is degree (1,1) in (core, tail); the tail's self-interaction is worth
7–10% of the tail's value per half. Put together as ONE program — "bilin18 = quadratic on the 768 core + core-gated linear on the tail,
everything else exact" — what does the whole model lose, and how does it scale down in k?

## Arms
- SPLIT8_1024 (.0374), LATE_DROP_TT_768 (.0087), EARLY_DROP_TT_768 (.0023) — instruments
- ALL18_DROP_TT_k: all 18 MLPs, tail×tail dropped, k = 768 / 512 / 384 (own frames blocks 0–7, bus U_8 blocks 8–17)
- ALL18_MLP_768: all 18 MLPs core×core only (the MLP-only width program at 768, for scale)
- LATE_MLP_512, LATE_DROP_TT_512: the late half's tt share at 512

## Predictions (bars literal; scored exactly as written)
- pred_a_instrument: baseline 3.0322401 ± 1e-4; SPLIT8_1024, LATE_DROP_TT_768, EARLY_DROP_TT_768 within .015 of prior.
- pred_b_all18_tail_linear_768_cheap: ALL18_DROP_TT_768 ≤ .020. NULL: ≥ .050.
- pred_c_all18_tail_linear_512_cheap: ALL18_DROP_TT_512 ≤ .050. NULL: ≥ .100.
- pred_d_late_tt_share_512_small: LATE_DROP_TT_512 / LATE_MLP_512 ≤ 0.15. NULL: ≥ 0.30.
- pred_e_all18_near_additive_768: ALL18_DROP_TT_768 ≤ 1.3 × (LATE_DROP_TT_768 + EARLY_DROP_TT_768). NULL: ≥ 2 × the parts.

## Price
1 fit pass (96 docs) + 64 eval docs × 10 forwards ≈ 736 GPU document-forwards; ≈ 25 s.
