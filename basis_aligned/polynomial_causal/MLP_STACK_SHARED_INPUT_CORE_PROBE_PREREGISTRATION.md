# Preregistration — mlp_stack_shared_input_core_probe (Claude, lane 1 CUDA)

Registered 2026-09-03 22:35Z (box clock). Follows §2742 (one shared input core for mlp11–17 is free at k ≥ 256) and §2743 (uniform width is right).

## Question
Does the whole MLP stack share ONE input coordinate system? Three questions at k = 768, constant filler, own weights, own means:
(1) do the early blocks mlp0–10 share an input core among themselves as the late ones do; (2) does the LATE core serve the EARLY blocks
and vice versa (is the residual stream's principal subspace depth-invariant at rank 768); (3) does one core for all 18 MLPs cost
anything over 18 per-block bases — and what does the whole MLP stack cost on one 768/896/1024-dim input core?

## Arms (CONST filler = each block's fit-set input mean; eval docs 0–63; fits docs 96–191; output unrestricted)
LATE7_OWN_768 (reproduces §2740's .079). EARLY11_OWN_768. EARLY11_SHARED_768 (core from the average of mlp0–10's centred input
covariances). EARLY11_ON_LATE_CORE_768. LATE7_ON_EARLY_CORE_768. ALL18_OWN_768. ALL18_SHARED_k for k ∈ {768, 896, 1024} (core from the
average of all 18 centred input covariances). Capture ratios (shared / own variance at 768) per block for each core.

## Predictions (CE added above the real model, docs 0–63, LOWER IS BETTER — §2135)
- pred_a_instrument: baseline within 1e-4 of 3.0322401; LATE7_OWN_768 within .02 of .079.
- pred_b_early_blocks_share_a_core: EARLY11_SHARED_768 − EARLY11_OWN_768 ≤ .03. Null: ≥ .10.
- pred_c_cores_transfer_across_depth: max(EARLY11_ON_LATE_CORE_768 − EARLY11_OWN_768, LATE7_ON_EARLY_CORE_768 − LATE7_OWN_768) ≤ .10.
  Null: ≥ .30.
- pred_d_one_core_for_all_18: ALL18_SHARED_768 − ALL18_OWN_768 ≤ .05. Null: ≥ .15.
- pred_e_whole_mlp_stack_on_1024: ALL18_SHARED_1024 ≤ .10. Null: ≥ .25.
Descriptive: EARLY11_OWN_768 vs LATE7_OWN_768 (are early blocks harder per block); ALL18_OWN_768 − (EARLY11 + LATE7) composition;
ALL18_SHARED_768/896/1024 curve; capture ratios.

## Price
96 fit docs + 64 × (1 + 9 arms) = 736 GPU document-forwards, ~15 s (18 float64 D×D input covariances on the GPU). Frozen: this file,
§2740 results, checkpoint, fit_natural.pt. Reproduction tolerance .02.
