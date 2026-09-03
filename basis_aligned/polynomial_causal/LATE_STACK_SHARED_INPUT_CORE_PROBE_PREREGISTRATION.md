# Preregistration — late_stack_shared_input_core_probe (Claude, lane 1 CUDA)

Registered 2026-09-03 22:31Z (box clock). Follows §2741 (a shared 16-dim INPUT core for mlp16/17 costs .01 over separate bases, where the shared WRITE core cost .066).

## Question
Compositional reuse across the whole late stack: can ONE k-dim input subspace, taken from the pooled centred input covariances of
mlp11–17 (each block keeps its own input mean and its own weights), replace the seven per-block bases at small cost? If so the late
program is "one shared k-dim input core + seven constants + the blocks' own weights restricted to the core".

## Arms (own weights; eval docs 0–63; fits docs 96–191; output unrestricted; CONST = fit-set mean filler, TOK = §2730 ridge read)
ALL7_CONST_256 (reproduces §2739 .397). SHARED7_CONST_k for k ∈ {64, 256, 512, 768}: U_k = top-k eigenvectors of the plain average of
the seven blocks' centred input covariances. SHARED7_TOK_256, SHARED7_TOK_768. Compared with §2739/§2740 per-block values
(ALL7_CONST_64/256/512/768 = .797/.397/.186/.079; ALL7_TOK_256/768 = .297/.065).

## Predictions (CE added above the real model, docs 0–63, LOWER IS BETTER — §2135)
- pred_a_instrument: baseline within 1e-4 of 3.0322401; ALL7_CONST_256 within .02 of .397.
- pred_b_shared_const_256: SHARED7_CONST_256 ≤ .45 (≤ .05 over per-block). Null: ≥ .55.
- pred_c_shared_const_768: SHARED7_CONST_768 ≤ .11. Null: ≥ .20.
- pred_d_shared_tok_256: SHARED7_TOK_256 ≤ .35. Null: ≥ .45.
- pred_e_shared_core_captures_variance: mean over the seven blocks of (variance captured by the shared 256-core / variance captured by
  the block's own top-256) ≥ .85. Null: ≤ .60.
Descriptive: the shared-vs-own gap as a function of k; per-block capture ratios; SHARED7_CONST_64 vs .797.

## Price
96 fit docs + 64 × (1 + 7 arms) = 608 GPU document-forwards, ~15 s. Frozen: this file, §2739 and §2740 results, checkpoint,
fit_natural.pt. Reproduction tolerance .02.
