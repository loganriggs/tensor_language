# Radial-constant + tangential-truncation map — preregistration

Registered 2026-09-03 20:08Z (box clock), before the script exists. Lane 1 (CUDA). SIGN CONVENTION (§2135): every number is CE
ADDED ABOVE THE REAL MODEL on held-out docs 96–159 — LOWER IS BETTER. Descriptive; nothing installs into the §312 frontier; the
bases are data covariances of the writes scored by CE only (§2118 stays closed).

## Motivation
§2705: at every one of the 36 write sites the radial component r x̂ (pre-write frame) can be replaced by a per-site constant r̄ x̂
(RADIAL_MEAN ≤ .036 everywhere; r̄ fitted on docs 0–95). §2696: plain in-situ PCA truncation of the whole write at k=32 costs
Σ_sites 2.3712 nat (mlp1 .883, mlp2 .220, mlp0 .165, mlp3 .130; 29 of 36 sites ≤ .05). §2702: at mlp1 keeping the radial part exact
and truncating the tangential part at k=64 cost .235 vs .357 for plain k=64. Question: with the radial axis taken out as a constant,
is the remaining tangential write cheaper to truncate — i.e. is "36 scalars + a rank-k tangential dictionary per site" a better
compact description than "a rank-k dictionary per site"?

## Design
Forward and split as §2704/§2705 (x̂ = pre-write unit residual, r = w·x̂, w_perp = w − r x̂). Fit on docs 0–95 in one collecting
pass per chunk: r̄_s (mean r) and the covariance of w_perp (mean μ_s, eigenvectors U_s). Arms, one site at a time, eval docs 96–159:
- **RM_TAN_k**: w' = r̄_s x̂ + μ_s + U_{s,k} U_{s,k}ᵀ (w_perp − μ_s), k ∈ {8, 32, 128}, all 36 sites.
- **RM_TAN_FULL** (k = 1152) at attn1 and mlp4 only — must equal §2705's RADIAL_MEAN (instrument).
Comparators (frozen priors): §2696 PLAIN_k32 per site; §2705 RADIAL_MEAN per site (the floor of RM_TAN_k as k → 1152).

## Predictions (bars fixed now)
- **pred_a_instrument**: baseline within 1e-4 of 3.1125031 (§2705, same device); RM_TAN_FULL at attn1 within 1e-3 of .0101 and at
  mlp4 within 1e-3 of .0360 (§2705 RADIAL_MEAN); RM_TAN_128 ≤ RM_TAN_32 ≤ RM_TAN_8 at ≥ 34 of 36 sites (monotone up to CE noise).
- **pred_b_radial_out_helps_per_site**: RM_TAN_32 < PLAIN_32 (§2696) at ≥ 28 of 36 sites. Null: ≤ 18 of 36 (no better than chance).
- **pred_c_attn1_compact**: attn1 RM_TAN_8 ≤ .05 (plain k=8 was .066; the attn1 write = constant gain + rank-8 tangential).
  Null: ≥ .10.
- **pred_d_total_price_drops**: Σ_36 RM_TAN_32 ≤ 0.8 × 2.3712 = 1.897 nat. Null: ≥ 0.95 × 2.3712 = 2.253.
- **pred_e_k128_ladder**: RM_TAN_128 ≤ .02 at ≥ 24 of 36 sites. Null: ≤ 14 of 36.

## Price
96 (fit) + 64·(1 + 3·36 + 2) = 7,200 GPU document-forwards + 36 eigendecompositions of 1152×1152 ≈ 70–90 s on lane 1. Output
radial_constant_tangential_truncation_map_probe_results.json. Frozen: this file, attention_radial_channel_probe_results.json
(db47a079d9969ee50e96901da03ca7e852e66c56205f0eb867ad7e749ffd8518), site_write_pca_truncation_ce_map_probe_results.json (48bd52ec…),
checkpoint, fit_natural.pt.
