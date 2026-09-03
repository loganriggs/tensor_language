# The pool's one-shot context map: how many input directions does it need, and does a quadratic term help? — preregistration

Registered 2026-09-03 21:25Z (box clock), before the script exists. Lane 1 (CUDA). SIGN CONVENTION (§2135): CE numbers are CE
ADDED ABOVE THE REAL MODEL on held-out docs 0–63 (FRESH split; fits on docs 96–191; baseline 3.0322401) — LOWER IS BETTER.
Descriptive; nothing installs into §312.

## Question
§2724: one 1152 × 1152 ridge map A of the block-11 normalised stream x̂₁₁ replaces mlp11–15 (ONESHOT_LIN .345 vs POOL_MEAN .724,
rec .524). §2723 showed the pool's VALUE is non-core (16-dim core-only .477). Is the map itself low-rank on the INPUT side
(a few context directions drive it), or does it read the stream broadly? And does a quadratic term in the top input directions —
the pool's blocks are bilinear — buy anything over the linear map?

## Method
Input-covariance-weighted truncation: with G = cov(x̂₁₁) (fit set) and A the full ridge map, M = G^{1/2} A = U S Vᵀ;
A_k = G^{-1/2} U_k S_k V_kᵀ (the rank-k map that best preserves A's output under the input distribution). Arms k ∈ {16, 32, 64,
128, 256, 512, 1152=full}. QUAD32: features x̂₁₁ ⊕ 528 quadratic monomials of the top-32 input PCA coordinates, one-shot ridge
(same λ rule). rec_k = 1 − CE(k)/CE(POOL_MEAN); eff rank of M = (Σs)²/Σs².

## Predictions (bars fixed now)
- **pred_a_instrument**: baseline within 1e-4; POOL_MEAN within .03 of .724; full map within .02 of .345.
- **pred_b_rank_128_keeps_most**: rec_128 ≥ .80 · rec_full. Null: ≤ .50 · rec_full.
- **pred_c_rank_16_is_not_the_map**: rec_16 ≤ .30 · rec_full. Null: ≥ .60 · rec_full.
- **pred_d_quadratic_helps_modestly**: CE(QUAD32) ≤ CE(full) − .03. Null: ≥ CE(full) − .005.
- **pred_e_map_is_broad**: eff rank of M ≥ 100. Null: ≤ 40.

## Price
96 × 2 fit passes + 64 × (2 + 8) ≈ 830 GPU document-forwards + ridge/SVD ≈ 25 s. Output late_pool_map_rank_curve_probe_results.json.
Frozen: this file, §2724 results (c3e1b9f3…), checkpoint, fit_natural.pt.
