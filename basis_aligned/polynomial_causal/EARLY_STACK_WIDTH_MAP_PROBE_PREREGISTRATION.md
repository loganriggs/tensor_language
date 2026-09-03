# Preregistration — early_stack_width_map_probe (Claude, lane 1 CUDA)

Registered 2026-09-03 22:51Z (box clock). Follows §2744 (early MLPs on own 768-cores .0435; early sharing +.080), §2746 (all-18 attention on 256 dims .227; late seven .053)
and §2745 (late 14-sublayer program on one 768 core .109). The early stack (blocks 0–10, 22 sublayers) has no width map yet.

## Question
How wide does the EARLY stack read, sublayer by sublayer and as a stack, and what does the complete 36-sublayer width program cost?

## Arms (eval docs 0–63; fits 96–191; MLPs = OwnHead CONST on top-k input PCs; attention = AttnHead on top-k input PCs, block 0
without the self-mixing of its own value residual)
EARLY11_OWN_768 (MLPs 0–10, §2744 reproduction); LATE14_JOINT_768 (§2745 reproduction).
SINGLE_{site}_256 for each of the 22 early sublayers alone on its own 256-core.
EARLY22_OWN_k for k ∈ {512, 768, 1024}: all 22 early sublayers on their own k-cores.
EARLY22_SHARED_k for k ∈ {768, 1024}: all 22 on ONE core = top-k of the average of the 22 centred input covariances.
ALL36_768: EARLY22_OWN_768 + LATE14_JOINT_768 together — the complete width program of the model.

## Predictions (CE added above the real model, docs 0–63, LOWER IS BETTER — §2135)
- pred_a_instrument: baseline within 1e-4 of 3.0322401; EARLY11_OWN_768 within .02 of .0435; LATE14_JOINT_768 within .02 of .109.
- pred_b_early_stack_768: EARLY22_OWN_768 ≤ .08. Null: ≥ .20.
- pred_c_whole_model_768: ALL36_768 ≤ .25. Null: ≥ .45.
- pred_d_no_early_sublayer_is_narrow_critical: max over the 22 SINGLE_{site}_256 ≤ .10. Null: ≥ .30.
- pred_e_early_sharing_cost: EARLY22_SHARED_768 − EARLY22_OWN_768 ≤ .10. Null: ≥ .20.
Descriptive: the 22 single-site costs by depth and kind; the own/shared curves at 512/1024; the 36 input effective ranks;
composition penalty ALL36 − (EARLY22_OWN + LATE14_JOINT).

## Price
96 fit docs + 64 × (1 + 2 + 22 + 3 + 2 + 1) = 2080 GPU document-forwards, ~40 s. Frozen: this file, §2747 results, checkpoint, fit_natural.pt.
