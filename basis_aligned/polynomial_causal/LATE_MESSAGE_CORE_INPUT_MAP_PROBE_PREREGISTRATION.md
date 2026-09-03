# Is the context part of the late message a 16 → 16 map on the core? — preregistration

Registered 2026-09-03 20:59Z (box clock), before the script exists. Lane 1 (CUDA). SIGN CONVENTION (§2135): CE numbers are CE
ADDED ABOVE THE REAL MODEL on held-out docs 0–63 (FRESH split; fits on docs 96–191; baseline 3.0322401) — LOWER IS BETTER;
rec = 1 − CE(arm)/CE(MEAN(mlp16+17)), higher = better. Descriptive surrogate-sufficiency test; nothing installs into §312.

## Question
§2717: replacing mlp16+mlp17 by μ + P_M A ê(t_cur) recovers 50% of their .848-nat value; the oracle core component P_M(w − μ)
recovers 81%. The missing ~30% is context-dependent and lives in the same 16 directions. Is it a function of the block's INPUT
restricted to those 16 directions (then the last two MLPs are a token lookup plus a 16 → 16 bilinear map — a tiny program), or
does it need the full 1152-dim input?

## Arms (mlp16 + mlp17 replaced together; output always restricted to the core: w → μ + P_M ŷ; ŷ ridge-fitted on docs 96–191)
Let ê = rms_norm(wte(t_cur)), x̂ = rms_norm(x_in) the block's MLP input, c = U_Mᵀ x̂ (16 core coordinates).
- CUR_M: ŷ = A ê (§2717 reproduction, .427).
- COREIN_M: ŷ = A ê + B c + Σ_{i≤j} Q_ij c_i c_j (1152 + 16 + 136 features).
- FULLIN_M: ŷ = A ê + B′ x̂ (2304 linear features; the linear-in-full-input ceiling).
- ORACLE_M: w → μ + P_M(w − μ) (§2717 .158).
Ridge λ = 1e-2·tr(ΦᵀΦ)/n_features on centred features. Held-out R² of each fit on the core-projected write reported.

## Predictions (bars fixed now)
- **pred_a_instrument**: baseline within 1e-4; MEAN within .02 of .848; CUR_M within .02 of .427; ORACLE_M within .02 of .158.
- **pred_b_core_input_carries_the_context_part**: rec(COREIN_M) ≥ rec(CUR_M) + .15. Null: ≤ rec(CUR_M) + .03.
- **pred_c_full_linear_input_adds_little**: rec(FULLIN_M) ≤ rec(COREIN_M) + .05. Null: ≥ rec(COREIN_M) + .15.
- **pred_d_close_to_the_oracle**: rec(COREIN_M) ≥ .90 × rec(ORACLE_M). Null: ≤ .70 ×.
- **pred_e_core_write_well_explained**: held-out R² of COREIN on the core-projected write ≥ .70 for both sites. Null: ≤ .40 for both.

## Price
96 × 3 fit passes + 64 × (1 + 5 arms + 2 R²) ≈ 800 GPU document-forwards + three ridge solves (≤ 2304²) ≈ 20 s. Output
late_message_core_input_map_probe_results.json. Frozen: this file, §2717 results (84a3d8a9…), checkpoint, fit_natural.pt.
