# Do mlp16/17 error-correct the pool? — clean-write compensation, and the composition penalty vs the last two blocks' input width — preregistration

Registered 2026-09-03 21:59Z (box clock), before the script exists. Lane 1 (CUDA). SIGN CONVENTION (§2135): CE numbers are CE
ADDED ABOVE THE REAL MODEL on held-out docs 0–63 (FRESH split; heads/fillers on docs 96–191; baseline 3.0322401) — LOWER IS
BETTER. π(X, Y) = CE(X+Y) − CE(X) − CE(Y). Descriptive; nothing installs into §312.

## Question
§2733: the .180 composition penalty between the pool (OWN_32_TOK, .319) and any core-only mlp16/17 program is not truncation, not
core-borne, additive over pool blocks, and not fixable by co-adaptation. The remaining reading: the REAL mlp16/17 respond to the
pool's non-core error in a way that partly cancels it (error correction / redundancy, §2721), and a program that reads only 16
input directions cannot. Two direct tests. (1) CLEAN-WRITE: pool perturbed, but mlp16 and mlp17 write the outputs they produce on
the UNPERTURBED stream (a parallel clean forward per chunk) — this removes their response to the pool's error while introducing no
program error; the excess over POOL alone is the compensation the real blocks provide. (2) INPUT WIDTH: mlp16/17 with their OWN
weights on the top-k PCs of their own input + token filler (§2730's recipe, k = 16 [= the core-rank FULL of §2733, but with input
PCs rather than the write core], 32, 64, 128, 256), alone and composed with the pool; π(k). If the correction is carried by a few
hundred input directions, π falls toward the fitted stack's .066 by k ≈ 128–256.

## Arms (everything else real)
MEAN7 · POOL (ref .319) · CLEAN16_17 alone (pool intact; must be ≈ 0 — the clean writes ARE the real writes) · POOL+CLEAN16_17 ·
PROG (ref .246) · POOL+PROG (ref .745) · OWN_k (k = 16, 32, 64, 128, 256; mlp16/17 own weights on top-k input PCs of each block's
own input + token filler; output unrestricted) · POOL+OWN_k. Derived: compensation κ = CE(POOL+CLEAN16_17) − CE(POOL); π_k =
CE(POOL+OWN_k) − CE(POOL) − CE(OWN_k).

## Predictions (bars fixed now)
- **pred_a_instrument**: baseline within 1e-4; POOL within .02 of .319; PROG within .02 of .246; POOL+PROG within .02 of .745;
  CLEAN16_17 alone ≤ .002.
- **pred_b_real_blocks_compensate**: κ ≥ .10 (the real blocks remove at least .10 of the pool's damage). Null: κ ≤ .03.
- **pred_c_penalty_falls_with_input_width**: π_128 ≤ .09 (half of .180). Null: π_128 ≥ .15.
- **pred_d_wide_own_weights_beat_the_core_program_alone**: CE(OWN_64) ≤ .20 (PROG .246; FULL .233). Null: ≥ .26.
- **pred_e_extracted_stack_with_wide_last_blocks_beats_the_fitted_stack**: CE(POOL+OWN_256) ≤ .55 (§2725 fitted .614). Null: ≥ .65.
Descriptive: the π_k curve; OWN_k curve for mlp16/17 vs §2730's pool curve.

## Price
96 fit docs × 3 passes + 64 × (1 + 2 + 2×2 + 2×5) + clean forwards 64 × 2 ≈ 1,400 GPU document-forwards ≈ 30 s.
Output late_last_two_error_correction_probe_results.json. Frozen: this file, §2733 results
(late_stack_composition_penalty_anatomy_probe_results.json), checkpoint, fit_natural.pt.
