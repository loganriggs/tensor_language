# How much of the whole late MLP stack (mlp11–17, 1.885 nat) is a current-token lookup? — preregistration

Registered 2026-09-03 21:03Z (box clock), before the script exists. Lane 1 (CUDA). SIGN CONVENTION (§2135): CE numbers are CE
ADDED ABOVE THE REAL MODEL on held-out docs 0–63 (FRESH split; ridge fits and covariances on docs 96–191; baseline 3.0322401) —
LOWER IS BETTER; rec = 1 − CE(arm)/CE(MEAN of the same site set), higher = better. Descriptive; nothing installs into §312.

## Question
§2717 decomposed the mlp16+17 message (.848 nat above the mean) as 50% current-token lookup, 81% inside the 16-dim shared late core.
Does the same executable description cover the whole late stack (mlp11–17, MEAN value 1.885 nat, §2716), or is the lookup a
property of the last two blocks only? This fixes how much of the late stack could be compiled as "μ_l + A_l ê(t) (+ context)".

## Arms (ridge λ = 1e-2·tr/nf, centred, fitted per site on docs 96–191; ê = rms_norm(wte(t_cur)); core = §2710 16-dim CORE_TN)
Joint late7 patch: MEAN7, CUR7 (w → μ + A ê), CUR7_M (core-restricted output), ORACLE7_M (μ + P_M(w − μ)).
Per-site single patches for each of mlp11…mlp15: MEAN_s, CUR_s (mlp16/17 singles are in §2716/§2717: MEAN .141/.365).
Held-out full-write R² of every site's ridge fit reported.

## Predictions (bars fixed now)
- **pred_a_instrument**: baseline within 1e-4; MEAN7 within .03 of 1.885 (§2716).
- **pred_b_late_stack_half_lookup**: rec(CUR7) ≥ .35. Null: ≤ .15.
- **pred_c_core_carries_the_stack**: rec(ORACLE7_M) ≥ .60 (16 dims carry ≥ 60% of the whole late stack's value). Null: ≤ .35.
- **pred_d_lookup_is_generic_across_late_sites**: median over mlp11…15 of single-site rec(CUR_s) ≥ .35. Null: ≤ .15.
- **pred_e_core_restriction_is_cheap_for_the_stack**: rec(CUR7_M) ≥ .80 × rec(CUR7). Null: ≤ .50 ×.

## Price
96 fit docs × 2 passes + 64 × (1 + 4 + 10 + 1) ≈ 1200 GPU document-forwards + 7 ridge solves (1152²) ≈ 20 s.
Output late_stack_token_lookup_map_probe_results.json. Frozen: this file, §2717 results (84a3d8a9…), checkpoint, fit_natural.pt.
