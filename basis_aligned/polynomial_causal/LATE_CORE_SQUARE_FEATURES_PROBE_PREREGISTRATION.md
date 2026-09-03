# How many distinct squared features do mlp16 and mlp17 compute, and do the two blocks share them? — preregistration

Registered 2026-09-03 21:37Z (box clock), before the script exists. Lane 1 (CUDA). SIGN CONVENTION (§2135): CE numbers are CE
ADDED ABOVE THE REAL MODEL on held-out docs 0–63 (FRESH split; filler ridge on docs 96–191; baseline 3.0322401) — LOWER IS BETTER.
Descriptive; nothing installs into §312.

## Setting (from §2727/§2728, exact)
In the 16-dim core, each block's output k is y_k = cᵀA_k c + b_k(t)ᵀc + d_k(t); only sym(A_k) matters for the quadratic term and
two eigenpairs per output suffice (+.020). Across 16 outputs that is ≤ 32 squared linear features per block in a 16-dim space.
Question: how many DISTINCT directions do the squares use, and is that direction set shared between mlp16 and mlp17?

## Method
Per block, S_l = Σ_k Σ_j |λ_kj| q_kj q_kjᵀ over the eigenpairs of sym(A_k) (importance-weighted square directions); Π_r = top-r
eigvectors of S_l. Arm OWN_r replaces A_k by Π_r A_k Π_r (cross and offset terms intact, exact otherwise). SHARED_r uses Π_r from
S_16 + S_17 for both blocks. SWAP_r uses mlp17's Π_r inside mlp16 and mlp16's inside mlp17. r ∈ {2, 4, 6, 8, 10, 12, 16}. Also
reported: principal angles (cos²) between the two blocks' OWN_8 spans; spectra of S_l.

## Predictions (bars fixed now)
- **pred_a_instrument**: baseline within 1e-4; OWN_16 within .003 of COMPILED_TOK .2334 (Π_16 = identity).
- **pred_b_eight_directions_carry_the_squares**: CE(OWN_8) − CE(OWN_16) ≤ .05. Null: ≥ .15.
- **pred_c_but_not_four**: CE(OWN_4) − CE(OWN_16) ≥ .08. Null: ≤ .03.
- **pred_d_one_shared_square_space_serves_both**: CE(SHARED_8) − CE(OWN_8) ≤ .02. Null: ≥ .08.
- **pred_e_swapping_the_blocks_square_spaces_is_cheap**: CE(SWAP_8) − CE(OWN_8) ≤ .05. Null: ≥ .15.

## Price
96 fit docs × 2 passes + 64 × (1 + 1 + 7 + 6 + 6) ≈ 1540 GPU document-forwards ≈ 30 s. Output late_core_square_features_probe_results.json.
Frozen: this file, §2728 results (late_core_program_structure_probe_results.json), checkpoint, fit_natural.pt.
