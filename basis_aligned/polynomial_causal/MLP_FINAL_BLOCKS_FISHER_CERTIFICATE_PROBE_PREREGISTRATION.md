# MLP16/17 write truncation: RMSNorm scale-gauge, pulled-back Fisher certificate, Fisher-whitened basis, shared interaction dictionary (Claude, CPU)

Registered 2026-09-03 16:31 UTC (system clock), BEFORE running. Script: `ops/mlp_final_blocks_fisher_certificate_probe.py`.
Results: `mlp_final_blocks_fisher_certificate_probe_results.json`. Price: CPU only, 0 GPU forwards; ~1,250 CPU document
forward-equivalents (96 fit docs: forward + 2 score backwards; 96 eval docs: forward + 1 true-token + 4 sampled-token
backwards; 288 patched eval forwards) — ~12-15 min at 16 threads. Source: MATHEMATICAL_REVIEW_2026-09-03_1630.md, Move 1 + 2.
SIGN CONVENTION (§2135): every CE number is CE ADDED ABOVE THE REAL MODEL on held-out natural docs 96-191 — LOWER = better.

## Objects and formulas (arms named)
- Bases: identical to §2694 (`fit_bases` on natural docs 0-95, all positions): mu_l, U_l (descending eigenvectors of the centred
  write covariance C_l), l in {16, 17}. Variance arm at rank k: w' = mu + U_k U_k^T (w - mu); delta_k = (I - U_k U_k^T)(w - mu).
- Scores (write space, per position t, exact autograd through the manual forward with model weights frozen):
  g_t = d[-log p_t(y_t)]/d w_t at the TRUE token; s_t^(i) = d[log p_t(y~)]/d w_t at y~ ~ p_t, i = 1..S sampled tokens
  (S = 4 on eval docs, S = 2 on fit docs; RNG pinned, torch.Generator seed 0). Both blocks' writes are leaves in one graph.
- CERTIFICATE (second-order, OBD lineage): pred_k = mean_t [ g_t . delta_k,t + 1/2 mean_i (s_t^(i) . delta_k,t)^2 ], eval docs.
  Measured_k = §2694's frozen `ce_added_ladder[l][k]` (same docs, same bases, same positions). ratio_k = measured_k / pred_k.
- FISHER METRIC: G_l = mean over fit-doc positions and samples of s s^T (1152 x 1152), regularised G_eps = G + eps I,
  eps = 1e-3 tr(G)/D. Whitened covariance M = G_eps^{1/2} C G_eps^{1/2}; V_k = its top-k eigenvectors;
  FISHER ARM at rank k: w' = mu + Pi_k (w - mu), Pi_k = G_eps^{-1/2} V_k V_k^T G_eps^{1/2} (oblique, G-orthogonal projector).
- RADIAL FRACTION: for block 17, x_18,t = x_t + w_t + Down_bias (residual entering the final rms-norm);
  rho_t = (w_t . x_18,t)^2 / (|w_t|^2 |x_18,t|^2). Mean over eval positions. Disclosed for every block l (w.r.t. its own
  post-write residual) and for the top variance direction u_1 of blocks 16/17: (u_1 . xhat_18)^2.
- SHARED DICTIONARY: Qs_j (j = 1..8) as in the 16:06 registration (`forms` of the quadratic-form probe, imported); B = eigenbasis
  of sum_j Qs_j^2; diagfrac_j = sum_i (b_i^T Qs_j b_i)^2 / |Qs_j|_F^2; mean over j, per block. (A random basis gives ~2/D.)

## Preregistered predictions (scored exactly as written)
- pred_a_instrument: (i) unpatched CE on eval docs reproduces §2694's frozen `baseline_ce.natural_h1` within 1e-4;
  (ii) the Fisher arm at k = D (full) adds |CE| <= 1e-4 on 4 eval docs (identity projector).
- pred_b_radial_gauge: mean rho (MLP17 write vs x_18) >= .5 — the dominant write is mostly radial, hence gauge under the final
  rms-norm. Null: <= .2.
- pred_c_certificate_mlp17: for ALL k in {4, 8, 16, 32, 64}, ratio_k in [.5, 2]. Null: ANY ratio_k outside [.25, 4].
  (k = 0, 1, 2 disclosed only: |delta| there is too large for second order.)
- pred_d_fisher_basis_mlp17_k8: the Fisher arm at k = 8 adds <= .05 (variance arm: .083). Null: >= .075.
  Closure note: §2118/§2125 closed metric bases on the §312 attention frontier; this is the final-block MLP write. If pred_d
  fails, the closure generalises and will be recorded as such.
- pred_e_no_shared_dictionary: mean diagfrac <= .2 in BOTH blocks (the eight forms do not share an eigenbasis).
  Null (a shared square-feature dictionary exists): >= .5 in either block.
Disclosed, not scored: MLP16 ratios (all k), Fisher arm MLP16 k = 8 and MLP17 k = 32, Fisher-metric effective rank of G_l
(entropy of its spectrum) for both blocks, per-block radial fractions (0-17), u_1 radial fraction, first- vs second-order
shares of pred_k, the tanh-curvature term is NOT modelled.

## Null model / what a failure means
pred_b false with null: the big write is not gauge; the variance/function gap needs another explanation. pred_c false: the
second-order certificate is not usable at this grain (higher-order or cross-position effects dominate) — no analytic pricing of
subspace surrogates. pred_d false with null: the loss-optimal metric does not rescue MLP17's rank-8 surrogate; the fat tail is
genuinely high-dimensional under the right metric too. pred_e null: the rank-8 write compiles to a shared dictionary — a
composable component to pursue.
Frozen: checkpoint 680d6c26…, fit_natural.pt 666a3201…, §2694 results 8a88b714…, quadratic-form probe script (hash frozen).
