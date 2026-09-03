# Late shared write core vs the readout — preregistration

Registered 2026-09-03 20:39Z (box clock), before the script exists. Lane 1 (CUDA; only the 96-doc fit and a 64-doc baseline check
touch the GPU; the rest is weight algebra). SIGN CONVENTION (§2135): the only CE number is the baseline instrument on held-out
docs 0–63 (FRESH split, 3.0322401); all other numbers are SUBSPACE OVERLAP FRACTIONS in [0,1] — HIGHER = more aligned, chance
= k/1152 (.111 at k = 128). Descriptive; nothing installs into the §312 frontier.

## Question
§2710 found one 128-dim dictionary serving all seven late MLP writes at 1.23× the separate price, with a pooled spectrum of
effective rank 10 (trace-dominated by mlp16/17). WHAT is that core? Three candidates, all decidable from the weights plus the
fitted covariances: (i) it is the readout-facing subspace (the top right-singular directions of lm_head); (ii) it is merely the
residual stream's own high-variance geometry; (iii) it is a late-stack-specific object (the early stack's pooled core should not
face the readout the same way). And is it READ by the late MLPs' Left/Right inputs, i.e. is the shared write also a shared read?

## Objects (all fitted on docs 96–191)
- CORE_TW_k: §2710's pooled basis (equal-weight mean of the seven centred write covariances of mlp11–17; trace-weighted).
- CORE_TN_k: the same with each covariance divided by its trace (removes the mlp16/17 dominance). Reported alongside.
- EARLY_TN_k: trace-normalised pooled core of mlp0–6 (seven MLPs; the control stack).
- LM_k: top-k right-singular subspace of the lm_head weight (eigh of WᵀW).
- XPCA_k: top-k PCA directions of the final residual stream x (input to the last rms_norm) on docs 96–191.
- Overlap ov(U_j, V_k) = ‖U_jᵀ V_k‖_F² / j — the fraction of span(U_j) lying inside span(V_k); chance j·k/(j·1152) = k/1152.
- Read-energy ratio of block l on a subspace U_k: ER_l(U_k) = [(‖Left_l U_k‖_F² + ‖Right_l U_k‖_F²)/k] / [(‖Left_l‖_F² + ‖Right_l‖_F²)/1152];
  1.0 = isotropic.

## Predictions (bars fixed now)
- **pred_a_instrument**: baseline CE within 1e-4 of 3.0322401; CORE_TW spectrum eff rank within 0.5 of 10.004 (§2710); every
  basis orthonormal to 1e-4; a seeded random 128-dim subspace overlaps LM_128 in [.08, .15].
- **pred_b_core_faces_readout**: ov(CORE_TW_16, LM_128) ≥ .60 (chance .111). Null: ≤ .25.
- **pred_c_late_specific**: ov(CORE_TN_128, LM_128) ≥ 1.5 × ov(EARLY_TN_128, LM_128). Null: ≤ 1.1 ×.
- **pred_d_core_is_read**: ER_l(CORE_TN_128) ≥ 1.5 for at least 5 of the 6 late readers l = 12…17 (disclosed confound: the core
  includes block l's own write; the reader sees only earlier writes — a positive result must survive the leave-own-out check
  reported alongside, ER_l on the pooled core of mlp11…(l−1)). Null: ER_l ≤ 1.1 for at least 5 of 6.
- **pred_e_more_readout_facing_than_the_stream**: ov(CORE_TW_16, LM_128) ≥ ov(XPCA_16, LM_128) + .10 (the core is not just the
  stream's biggest directions). Null: ≤ ov(XPCA_16, LM_128) − .10.

## Price
96 + 64 GPU document-forwards + a dozen 1152² eigendecompositions ≈ 10 s. Output late_core_readout_alignment_probe_results.json.
Frozen: this file, §2710 results (b932d545…), checkpoint, fit_natural.pt.
