# Early joint installation k-ladder — preregistration

Registered 2026-09-03 20:31Z (box clock), before the script exists. Lane 1 (CUDA). SIGN CONVENTION (§2135): every number is CE
ADDED ABOVE THE REAL MODEL on held-out docs 0–63 (FRESH split as §2703/§2709: bases from docs 96–191, baseline 3.0322401) —
LOWER IS BETTER. Descriptive; nothing installs into the §312 frontier.

## Question
§2696: mlp0–3 alone are 59 % of the 36-site single rank-32 price; §2709: the late stack's joint price is 2.8× its singles at
k=32 with a slowly decaying factor. Is the EARLY stack (EARLY8 = attn_l, mlp_l, l = 0…3) also superadditive, how fast does its
joint price fall with rank, and is it — as the single-site map says — the more expensive half of the model to compress?

## Arms
Plain in-situ write PCA per site from docs 96–191. SINGLE_s(k) and JOINT(k) over EARLY8 for k ∈ {32, 64, 128, 256, 512};
JOINT_MLP4(128) for mlp0–3 only. F(k) = JOINT(k) / Σ SINGLE(k).

## Predictions (bars fixed now)
- **pred_a_instrument**: baseline within 1e-4 of 3.0322401; identity patch ≤ 1e-4; all 9 chains monotone in k.
- **pred_b_early_superadditive**: F(32) ≥ 1.5. Null: ≤ 1.1.
- **pred_c_factor_decays**: F(512) ≤ .8 × F(32). Null: F(512) ≥ F(32).
- **pred_d_early_stack_price_512**: JOINT(512) ≤ .10. Null: ≥ .30.
- **pred_e_early_costs_more_than_late**: JOINT(128) ≥ .486 (= §2709's late JOINT(128)). Null: ≤ .30.

## Price
96 fit + 64 · (2 + 5·9 + 1) + 8 ≈ 3,180 GPU document-forwards ≈ 30 s. Output early_joint_k_ladder_probe_results.json.
Frozen: this file, §2709 results (6b2708a3…), checkpoint, fit_natural.pt.
