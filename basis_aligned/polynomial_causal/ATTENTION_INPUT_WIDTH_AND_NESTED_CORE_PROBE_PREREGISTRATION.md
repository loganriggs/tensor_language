# Preregistration — attention_input_width_and_nested_core_probe (Claude, lane 1 CUDA)

Registered 2026-09-03 22:41Z (box clock). Follows §2745 (late attention on 768 input dims costs .0075; joint 14-sublayer late program .109 at 768).

## Question
Attention's own input-width knob is unpriced (§2745 limits). Where is attention's cliff, does it hold for all 18 attention blocks,
and — because the shared core is ONE ordered eigenbasis — can the late program be NESTED: attention reads the first k_a directions,
the MLPs the first 768, of the same U?

## Arms (own weights, own fit-set input means, constant filler; eval docs 0–63; fits docs 96–191; block-0 value residual exact)
ATTN7_OWN_768 (reproduces §2745 .0075). ATTN7_OWN_k for k ∈ {64, 128, 256, 512} (each late attention block on its own top-k input
PCs). ATTN18_OWN_256 (all 18 attention blocks, own top-256). ATTN18_OWN_128. LATE14_JOINT_768 (reproduces §2745 .109).
LATE14_NESTED_768_256 and LATE14_NESTED_768_128: MLPs 11–17 read the first 768 columns of the joint late core, attention 11–17 the
first 256 (128) columns of the same core.

## Predictions (CE added above the real model, docs 0–63, LOWER IS BETTER — §2135)
- pred_a_instrument: baseline within 1e-4 of 3.0322401; ATTN7_OWN_768 within .01 of .0075; LATE14_JOINT_768 within .02 of .109.
- pred_b_attention_256: ATTN7_OWN_256 ≤ .05. Null: ≥ .15.
- pred_c_attention_128: ATTN7_OWN_128 ≤ .15. Null: ≥ .40.
- pred_d_all_attention_256: ATTN18_OWN_256 ≤ .15. Null: ≥ .40.
- pred_e_nested_is_free: LATE14_NESTED_768_256 − LATE14_JOINT_768 ≤ .03. Null: ≥ .10.
Descriptive: the attention width curve; ATTN18_OWN_128; NESTED_768_128; attention-input effective ranks for all 18 blocks.

## Price
96 fit docs + 64 × (1 + 10 arms) = 800 GPU document-forwards, ~15 s. Frozen: this file, §2745 results, checkpoint, fit_natural.pt.
