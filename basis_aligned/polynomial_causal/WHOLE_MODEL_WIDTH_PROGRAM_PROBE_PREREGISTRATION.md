# Preregistration — whole_model_width_program_probe (Claude, lane 1 CUDA)

Registered 2026-09-03 22:56Z (box clock). Follows §2749 (all 36 sublayers on 768 input cores .197; early own 1024 .0084), §2745 (late joint 1024 .023) and §2748 (BUS routing).

## Question
What does the complete width program of bilin18 cost at 896 and 1024, is its composition penalty small at 1024, and does the BUS
form (late out-of-core writes straight to the readout) hold inside the whole-model program?

## Arms (eval docs 0–63; fits 96–191; early 22 sublayers on OWN k-cores, late 14 on ONE shared k-core; MLPs OwnHead CONST, attention AttnHead)
EARLY22_OWN_1024, LATE14_JOINT_1024, ALL36_768 (reproductions).
ALL36_k for k ∈ {896, 1024}.
ALL36_BUS_k for k ∈ {768, 1024}: ALL36_k with the late fourteen writes' out-of-core part (w.r.t. the late k-core) routed to the
final residual instead of the stream (§2748 construction).

## Predictions (CE added above the real model, docs 0–63, LOWER IS BETTER — §2135)
- pred_a_instrument: baseline within 1e-4 of 3.0322401; ALL36_768 within .02 of .197; LATE14_JOINT_1024 within .015 of .023;
  EARLY22_OWN_1024 within .015 of .0084.
- pred_b_whole_model_1024: ALL36_1024 ≤ .05. Null: ≥ .12.
- pred_c_whole_model_896: ALL36_896 ≤ .12. Null: ≥ .25.
- pred_d_composition_small_at_1024: ALL36_1024 − (EARLY22_OWN_1024 + LATE14_JOINT_1024) ≤ .015. Null: ≥ .05.
- pred_e_bus_holds_in_the_whole_program: |ALL36_BUS_768 − ALL36_768| ≤ .02. Null: ≥ .06.
Descriptive: ALL36_BUS_1024; the three-point whole-model curve 768/896/1024.

## Price
96 fit docs + 64 × (1 + 7 arms) = 608 GPU document-forwards, ~15 s. Frozen: this file, §2749 results, checkpoint, fit_natural.pt.
