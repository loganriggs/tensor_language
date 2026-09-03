# The whole late MLP stack under the extraction recipe: pool blocks on 32 input PCs + token filler, mlp16/17 as the shared-square-space program — preregistration

Registered 2026-09-03 21:43Z (box clock), before the script exists. Lane 1 (CUDA). SIGN CONVENTION (§2135): CE numbers are CE
ADDED ABOVE THE REAL MODEL on held-out docs 0–63 (FRESH split; heads/fillers on docs 96–191; baseline 3.0322401) — LOWER IS
BETTER. Descriptive; nothing installs into §312.

## Question
§2725 priced the late stack as [fitted linear map] + [mlp16/17 own weights on the core] at .614 of MEAN7 1.885. §2729 found the
mlp16/17 program improves when its quadratic core is restricted to one shared 8-dim square space; §2730 found the pool is better
described by its own weights on 32 input PCs + token filler (.319) than by the fitted map (.345). What does the late stack cost
under the extraction recipe throughout, and does output-side truncation of the pool blocks (their writes' own PCs) come cheap?

## Arms (everything else real)
MEAN7 (ref 1.885) · POOL_OWN32_TOK (ref .319) · POOL_OWN32_TOK_OUT256 / _OUT64 (each pool block's write additionally restricted to
μ + Π_r(w − μ), Π_r = top-r PCs of that block's write on the fit set) · PROG16_17 = mlp16/17 exact compile with the quadratic core
restricted to the SHARED top-8 square space (Π₈ of S₁₆ + S₁₇, §2729), the token read at rank 8 (B truncated as in §2728), no offset
· COMBINED = POOL_OWN32_TOK + PROG16_17 · COMBINED_OUT256 = POOL_OWN32_TOK_OUT256 + PROG16_17. Composition penalty π = CE(COMBINED)
− CE(POOL_OWN32_TOK) − CE(PROG16_17).

## Predictions (bars fixed now)
- **pred_a_instrument**: baseline within 1e-4; MEAN7 within .03 of 1.885; POOL_OWN32_TOK within .02 of .319.
- **pred_b_shared_square_program_beats_minimal**: CE(PROG16_17) ≤ .26 (§2728 MINIMAL was .271; §2729 SHARED_8 pruning gains ~.01). Null: ≥ .30.
- **pred_c_extracted_late_stack_beats_the_fitted_one**: CE(COMBINED) ≤ .55 (§2725 COMBINED_SEQ_TOK .614). Null: ≥ .65.
- **pred_d_pool_output_truncation_to_256_is_cheap**: CE(POOL_OWN32_TOK_OUT256) − CE(POOL_OWN32_TOK) ≤ .03. Null: ≥ .10.
- **pred_e_composition_penalty_small**: π ≤ .10. Null: ≥ .25.

## Price
96 fit docs × 2 passes + 64 × (1 + 7) ≈ 700 GPU document-forwards ≈ 25 s. Output late_stack_extracted_program_probe_results.json.
Frozen: this file, §2730 results (late_pool_own_weights_input_head_probe_results.json), §2729 results
(late_core_square_features_probe_results.json), checkpoint, fit_natural.pt.
