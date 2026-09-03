# Preregistration — late_attention_shared_input_core_probe (Claude, lane 1 CUDA)

Registered 2026-09-03 22:37Z (box clock). Follows §2742 (one input core for mlp11–17 is free). Independent of mlp_stack_shared_input_core_probe (queued ahead).

## Question
The late-stack program so far replaces only the MLPs; the seven late ATTENTION blocks still read the full residual. Do they read the
same input subspace? Each attention block l ∈ 11..17 is recomputed from a projected input h = x̄_l + U Uᵀ (x̂ − x̄_l) (own weights,
own fit-set mean; the value-residual mix with block 0's v is kept exact), with U = the block's own top-k input PCs, the MLP-shared
late core (§2742), a core pooled over the seven attention inputs, or a JOINT core pooled over all fourteen late inputs (MLP + attention)
which then also carries the seven MLPs. k = 768 unless stated; constant filler only.

## Arms (eval docs 0–63; fits docs 96–191; output unrestricted)
IDENTITY: the attention recompute with no projection (must reproduce the real model — instrument). MLP7_SHARED_768 (reproduces §2742
.084). ATTN7_OWN_768. ATTN7_ON_MLP_CORE_768. ATTN7_SHARED_768. LATE14_JOINT_k for k ∈ {768, 896, 1024}: all seven MLPs and all seven
attention blocks on one core. Capture ratios per block for each core.

## Predictions (CE added above the real model, docs 0–63, LOWER IS BETTER — §2135)
- pred_a_instrument: baseline within 1e-4 of 3.0322401; IDENTITY ≤ 1e-3; MLP7_SHARED_768 within .02 of .084.
- pred_b_attention_reads_768: ATTN7_OWN_768 ≤ .10. Null: ≥ .30.
- pred_c_attention_reads_the_mlp_core: ATTN7_ON_MLP_CORE_768 − ATTN7_OWN_768 ≤ .03. Null: ≥ .10.
- pred_d_joint_late_program_768: LATE14_JOINT_768 ≤ .25. Null: ≥ .45.
- pred_e_joint_late_program_1024: LATE14_JOINT_1024 ≤ .08. Null: ≥ .20.
Descriptive: ATTN7_SHARED_768 vs own; LATE14_JOINT_768 − (MLP7_SHARED_768 + ATTN7_ON_MLP_CORE_768) composition; the joint k curve.

## Price
96 fit docs (one recording pass for 14 input covariances) + 64 × (1 + 8 arms) = 672 GPU document-forwards, ~15 s. Frozen: this file,
§2742 results, checkpoint, fit_natural.pt. Reproduction tolerance .02.
