# frame_principal_angle_spectrum_probe — preregistration (Registered 2026-09-03 23:17Z (box clock))

Lane 1 CUDA (Claude; one covariance pass + one instrument arm). Follows §2755 (the early frame drift is not a ≤ 128-direction
in-span/complement swap; construction-dependent). Question, construction-free: how many principal angles separate consecutive
early read frames, and every frame from the settled frame U_8 of §2754? Two 768-dim subspaces of R^1152 share at least 384
directions exactly, so at most 384 principal angles can be non-zero; "low-rank drift" means few of those 384 are large.

Sign convention (§2135) applies only to the instrument arm (CE ADDED above the real model, docs 0–63, LOWER IS BETTER).

Construction: the 36 own top-768 input cores U_s (fit docs 96–191, §2749 covariances) and the settled frame U_8 (top-768 of the
averaged covariance of blocks 8–17's 20 sites). For frames A, B (1152×768, orthonormal): cosines of principal angles = singular
values of AᵀB (768 of them; ≥ 384 equal 1 up to rounding). n_θ>t(A, B) = number of angles above t degrees, t ∈ {10, 30, 60}.
Sequence of the 22 early sites: attn0, mlp0, …, attn10, mlp10 (21 consecutive pairs). Reported for all 35 consecutive pairs of
the 36-site sequence and for all 36 sites against U_8.

Frozen: this file, §2755 results (early_frame_drift_rank_probe_results.json), checkpoint, fit_natural.pt.

- pred_a_instrument: baseline 3.0322401 within 1e-4; EARLY22_OWN_768 within .02 of .057.
- pred_b_drift_is_broad: median over the 21 early consecutive pairs of n_θ>30° ≥ 100 (of ≤ 384). Null: ≤ 30.
- pred_c_no_pair_is_nearly_shared: every early consecutive pair has n_θ>10° ≥ 200. Null: some early pair has n_θ>10° ≤ 80.
- pred_d_settled_sites_are_close_to_U8: median over the 20 sites of blocks 8–17 of n_θ>30°(U_s, U_8) ≤ 60. Null: ≥ 150.
- pred_e_drift_toward_U8_is_monotone: over the 22 early sites in sequence order, Spearman(sequence index, n_θ>30°(U_s, U_8)) ≤ −0.7.
  Null: ≥ −0.2.

Price: 1 fit pass (96 docs) + baseline + 1 arm × 64 docs = 224 GPU document-forwards (≈ 10 s) + CPU SVDs. Descriptive; nothing
installs into the §312 frontier.
