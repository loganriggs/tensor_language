# Late joint installation k-ladder — preregistration

Registered 2026-09-03 20:19Z (box clock; an earlier draft line said 20:24Z — a mis-typed stamp, corrected before the script hash was frozen), before the script exists. Lane 1 (CUDA). SIGN CONVENTION (§2135): every number is CE
ADDED ABOVE THE REAL MODEL on held-out docs — LOWER IS BETTER. FRESH split as §2703: bases fitted on docs 96–191, CE on docs
0–63 (baseline 3.0322321 on CPU; ≤ 1e-4 expected on CUDA per §2704). Descriptive; nothing installs into the §312 frontier.

## Question
§2703: installing all 14 late writes (LATE14 = attn_l, mlp_l, l = 11…17) at rank 32 costs .902 nat = 2.76× the sum of the 14
singles (.327); the price is certified pairwise (second-order Fisher). If the joint price is a quadratic form in the per-site
truncation residuals, and each residual shrinks with k by a roughly common factor, the SUPERADDITIVITY FACTOR
F(k) = joint(k) / Σ singles(k) should be roughly k-independent while the absolute price falls. This ladder measures joint(k) and
all 14 singles(k) at k ∈ {32, 64, 128, 256, 512} in one GPU run, and gives the first honest "price of the whole late stack" curve.

## Arms
Plain in-situ write PCA (μ_s, U_s) per site from docs 96–191. SINGLE_s(k): only site s truncated, w' = μ + U_k U_kᵀ(w − μ).
JOINT(k): all 14 late sites truncated simultaneously with the same rule. F(k) = JOINT(k) / Σ_s SINGLE_s(k).

## Predictions (bars fixed now)
- **pred_a_instrument**: baseline within 1e-4 of 3.0322321; JOINT(32) within .01 of §2703's .9017 and Σ SINGLE(32) within .01 of
  .3268 (2bdb4ea7…); JOINT and all 14 single chains monotone non-increasing in k (≥ 14 of 15 chains).
- **pred_b_factor_k_independent**: F(k) ∈ [2.0, 3.6] at every k ∈ {64, 128, 256}. Null: F(256) ≤ 1.3 (additive at high rank) or
  F(256) ≥ 5.
- **pred_c_late_stack_price_128**: JOINT(128) ≤ .35 nat. Null: ≥ .60.
- **pred_d_late_stack_price_512**: JOINT(512) ≤ .05 nat (512 of 1152 dims per site). Null: ≥ .15.
- **pred_e_mlp_pairs_carry_it**: at k = 128, JOINT_MLP7(128) (the seven late MLPs only, §2703's A3 set) ≥ .6 × JOINT(128).
  Null: ≤ .3 × JOINT(128).

## Price
96 fit + 64 · (1 + 5·14 + 5 + 1) = 4,960 GPU document-forwards ≈ 1 min. Output late_joint_k_ladder_probe_results.json.
Frozen: this file, §2703 results (2bdb4ea7…), checkpoint, fit_natural.pt.
