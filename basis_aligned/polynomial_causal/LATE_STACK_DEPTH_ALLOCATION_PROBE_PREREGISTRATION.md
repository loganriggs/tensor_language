# Preregistration — late_stack_depth_allocation_probe (Claude, lane 1 CUDA)

Registered 2026-09-03 22:31Z (box clock). Follows §2741 (restore gains monotone in depth, mlp17 the bottleneck) and §2740 (iii) (pool-wide allocation beat uniform).

## Question
§2741 says the marginal value of a block's fidelity inside the late stack grows with depth; §2740's one allocation arm widened the
POOL and still beat uniform. Which way should a fixed budget of input directions be tilted? Fixed total 2240 directions across the
seven blocks (own weights, own centred input PCs, own means):
UNIFORM: 320 × 7. LATE_HEAVY: 128, 192, 256, 320, 384, 448, 512 (mlp11…17). EARLY_HEAVY: the reverse. LATE_STEEP: 64, 128, 192, 256,
384, 512, 704. Each with the TOK filler and with the CONST filler (8 arms) plus ALL7_TOK_256 (reproduction).

## Predictions (CE added above the real model, docs 0–63, LOWER IS BETTER — §2135)
- pred_a_instrument: baseline within 1e-4 of 3.0322401; ALL7_TOK_256 within .02 of .297.
- pred_b_late_heavy_beats_uniform_tok: LATE_HEAVY_TOK ≤ UNIFORM_TOK − .02. Null: LATE_HEAVY_TOK ≥ UNIFORM_TOK.
- pred_c_early_heavy_loses_tok: EARLY_HEAVY_TOK ≥ UNIFORM_TOK + .02. Null: EARLY_HEAVY_TOK ≤ UNIFORM_TOK.
- pred_d_same_order_const: LATE_HEAVY_CONST ≤ UNIFORM_CONST − .02 AND EARLY_HEAVY_CONST ≥ UNIFORM_CONST + .02. Null: either ordering
  reversed.
- pred_e_steeper_is_better_tok: LATE_STEEP_TOK ≤ LATE_HEAVY_TOK. Null: LATE_STEEP_TOK ≥ LATE_HEAVY_TOK + .02.
Descriptive: the four CONST values; the TOK − CONST gap per allocation.

## Price
96 fit docs + 64 × (1 + 9 arms) = 736 GPU document-forwards, ~15 s. Frozen: this file, §2739 results, checkpoint, fit_natural.pt.
Reproduction tolerance .02.
