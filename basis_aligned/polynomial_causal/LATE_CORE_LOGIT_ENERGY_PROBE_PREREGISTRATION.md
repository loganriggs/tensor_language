# How the readout reads the late core: logit-energy fraction and clean-ablation references — preregistration

Registered 2026-09-03 20:51Z (box clock), before the script exists. Lane 1 (CUDA). SIGN CONVENTION (§2135): CE numbers are CE
ADDED ABOVE THE REAL MODEL on held-out docs 0–63 (FRESH split; core from docs 96–191; baseline 3.0322401) — LOWER IS BETTER.
Fractions and ratios are labelled as such (HIGHER = more read). Descriptive; nothing installs into the §312 frontier.

## Question
§2714: dropping the CORE_16 component of the late MLP writes costs 6.15 nat although the core overlaps lm_head's top-128
right-singular subspace only at .18 (§2713). Which is it — does lm_head read the core with above-isotropic weight gain, or is the
core simply where the normalised final stream x̂ keeps most of its energy, read isotropically? And is §2714's oddity real: is
dropping ONLY the core of a write worse than replacing the whole write by its mean?

## Objects
M_16 = CORE_TW_16 (late-MLP pooled write PCA, docs 96–191); X_16 = final-stream PCA top-16; E_16 = early-MLP (mlp0–6) pooled
core; R_16 = seeded random orthonormal. W_c = lm_head weight with its per-column mean over the vocabulary removed (the
shift-invariant part of the logits). x̂ = rms_norm(x_final) on docs 0–63.
- Logit-energy fraction q(U) = Σ_t ‖W_c P_U x̂_t‖² / Σ_t ‖W_c x̂_t‖².  Activation fraction p(U) = Σ_t ‖P_U x̂_t‖² / Σ_t ‖x̂_t‖².
- Weight read-energy ratio ER_lm(U) = (‖W_c U‖²_F/16) / (‖W_c‖²_F/1152); 1.0 = isotropic.
- Clean ablations (CE added): MEAN(S) = every write in S replaced by its fit-set mean, for S = {mlp16}, {mlp17}, {mlp16, mlp17},
  {mlp11…17}; DROPCORE16(S) = the §2714 PLAIN_16 patch applied to S only, for S = {mlp16, mlp17} and {mlp11…17} (the latter is the
  §2714 reproduction).

## Predictions (bars fixed now)
- **pred_a_instrument**: baseline within 1e-4; DROPCORE16(mlp11–17) within .02 of 6.1496; ER_lm(R_16) ∈ [.9, 1.1]; q(R_16) ≤ .03.
- **pred_b_core_carries_logit_energy**: q(M_16) ≥ .50. Null: ≤ .20.
- **pred_c_lm_reads_core_above_isotropic**: ER_lm(M_16) ≥ 1.5. Null: ≤ 1.1. (If b TRUE and c null met, the core's readout
  relevance is activation amplitude, not weight gain.)
- **pred_d_dropping_only_the_core_is_worse_than_mean_ablation**: MEAN({mlp16, mlp17}) ≤ .80 × DROPCORE16({mlp16, mlp17}).
  Null: ≥ 1.0 ×.
- **pred_e_last_mlp_is_essential**: MEAN({mlp17}) ≥ 1.0 nat. Null: ≤ .30.

## Price
96 fit + 64 × (1 baseline + 1 logit pass + 6 ablation arms) = 608 GPU document-forwards ≈ 15 s. Output
late_core_logit_energy_probe_results.json. Frozen: this file, §2714 results (60e663ce…), checkpoint, fit_natural.pt.
