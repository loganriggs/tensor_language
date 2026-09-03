# Inside the compiled 16-dim program of mlp16/17: how many squares per output, and how many token directions modulate the read? — preregistration

Registered 2026-09-03 21:33Z (box clock), before the script exists. Lane 1 (CUDA). SIGN CONVENTION (§2135): CE numbers are CE
ADDED ABOVE THE REAL MODEL on held-out docs 0–63 (FRESH split; filler ridge on docs 96–191; baseline 3.0322401) — LOWER IS BETTER.
Descriptive; nothing installs into §312.

## Setting (from §2727, exact)
mlp16/17's core program is y_k = cᵀA_k c + b_k(t)ᵀc + d_k(t) (k = 1..16), with A_k token-free 16 × 16, b(t) affine in ê_t —
b(t) = b₀ + B(ê_t − ē), B a 256 × 1152 matrix per block — and d(t) a per-token offset. §2727 measured: dropping the quadratic
+1.61, dropping the cross term +.236, dropping the offset +.014, token-free cross +.073, A_k at rank 4 +.006.

## Arms (mlp16 and 17 jointly; everything else real)
COMPILED_TOK (ref .2334) · CROSS_LINEAR_FULL (cross term rebuilt from B — must equal COMPILED_TOK) · CROSS_RANK_r, r ∈ {1, 2, 4, 8,
16, 32, 64} (B truncated by SVD weighted with the fit-set covariance of ê) · QUAD_SYMRANK_r, r ∈ {1, 2, 3} (sym(A_k) eigen-
truncated to its r largest-|λ| eigenpairs — r squared linear features per output) · MINIMAL = QUAD_SYMRANK_2 + CROSS_RANK_8 +
no offset. Per block MINIMAL holds 16 × 2 × 17 + 8 × (1152 + 256) ≈ 11.8 k numbers (plus the shared 1152 × 16 core basis).

## Predictions (bars fixed now)
- **pred_a_instrument**: baseline within 1e-4; COMPILED_TOK within .003 of .2334; |CE(CROSS_LINEAR_FULL) − CE(COMPILED_TOK)| ≤ .003.
- **pred_b_token_read_is_low_rank**: CE(CROSS_RANK_16) − CE(COMPILED_TOK) ≤ .03. Null: ≥ .10.
- **pred_c_but_not_one_dimensional**: CE(CROSS_RANK_2) − CE(COMPILED_TOK) ≥ .10. Null: ≤ .03.
- **pred_d_two_squares_per_output_suffice**: CE(QUAD_SYMRANK_2) − CE(COMPILED_TOK) ≤ .03. Null: ≥ .10.
- **pred_e_minimal_program_keeps_two_thirds**: CE(MINIMAL) ≤ .30 (rec ≥ .65 of MEAN_16_17 .848). Null: ≥ .45.

## Price
96 fit docs × 2 passes + 64 × (1 + 13) ≈ 1090 GPU document-forwards ≈ 20 s. Output late_core_program_structure_probe_results.json.
Frozen: this file, §2727 results (late_core_polynomial_compile_probe_results.json), checkpoint, fit_natural.pt.
