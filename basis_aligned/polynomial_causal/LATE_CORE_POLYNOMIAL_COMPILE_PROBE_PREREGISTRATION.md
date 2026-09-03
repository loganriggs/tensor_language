# Compiling mlp16/17's core program into an explicit 16-dimensional polynomial: 16 token-free 16 × 16 quadratic forms + a token-dependent linear term + a token offset — preregistration

Registered 2026-09-03 21:28Z (box clock), before the script exists. Lane 1 (CUDA). SIGN CONVENTION (§2135): CE numbers are CE
ADDED ABOVE THE REAL MODEL on held-out docs 0–63 (FRESH split; filler ridge on docs 96–191; baseline 3.0322401) — LOWER IS BETTER.
Descriptive; nothing installs into §312.

## Algebra (exact)
§2720's W_TOKFILL_M arm computes w = μ + P Pᵀ(Down[(Left x′)(Right x′)] − μ) with x′ = P c + f(t), c = Pᵀ x̂ (16 dims), f(t) =
x̄_⊥ + (ê_t − ē) A_fill the token filler. With L_c = Left·P, R_c = Right·P (4608 × 16), ℓ(t) = Left f(t), r(t) = Right f(t),
D_c = Pᵀ Down (16 × 4608), the 16 core outputs are EXACTLY
  y_k = cᵀ A_k c + b_k(t)ᵀ c + d_k(t),   A_k = L_cᵀ diag(D_c[k]) R_c (16 × 16, token-free),
  b_k(t) = L_cᵀ diag(D_c[k]) r(t) + R_cᵀ diag(D_c[k]) ℓ(t),   d_k(t) = Σ_h D_c[k,h] ℓ_h(t) r_h(t).
So the block's core program is 16 quadratic forms (4096 numbers), a per-token 16 × 16 linear read, and a per-token 16-vector
offset. No hidden width. This rung builds it explicitly and asks which term carries the value.

## Arms (each at mlp16 and mlp17 jointly; everything else real)
COMPILED_TOK (all three terms; must equal W_TOKFILL_M) · COMPILED_MEAN (f = x̄_⊥, i.e. ℓ0, r0; must equal W_MEANFILL_M) ·
NO_QUAD (drop cᵀA c) · NO_CROSS (drop bᵀc) · NO_OFFSET (drop d) · CROSS_TOKFREE (b built from ℓ0, r0; d keeps the token) ·
QUAD_RANK4 (each A_k SVD-truncated to rank 4) · also W_TOKFILL_M and W_MEANFILL_M computed the ordinary way.

## Predictions (bars fixed now)
- **pred_a_instrument**: baseline within 1e-4; |CE(COMPILED_TOK) − CE(W_TOKFILL_M)| ≤ .003 and |CE(COMPILED_MEAN) − CE(W_MEANFILL_M)| ≤ .003 (exact algebra, fp32 wobble); W_TOKFILL_M within .02 of .233.
- **pred_b_quadratic_term_matters**: CE(NO_QUAD) − CE(COMPILED_TOK) ≥ .15. Null: ≤ .03.
- **pred_c_linear_term_matters**: CE(NO_CROSS) − CE(COMPILED_TOK) ≥ .15. Null: ≤ .03.
- **pred_d_token_enters_through_the_offset**: CE(CROSS_TOKFREE) − CE(COMPILED_TOK) ≤ .04. Null: ≥ .10.
- **pred_e_quadratic_forms_are_simple**: CE(QUAD_RANK4) − CE(COMPILED_TOK) ≤ .03. Null: ≥ .10.

## Price
96 fit docs × 2 passes + 64 × (1 + 9) ≈ 830 GPU document-forwards ≈ 20 s. Output late_core_polynomial_compile_probe_results.json.
Frozen: this file, §2720 results (57772458…), checkpoint, fit_natural.pt.
