# Can the mlp11–15 pool be extracted the way mlp16/17 were — its own weights on a low-dimensional input head — and does that beat the fitted linear map at equal input rank? — preregistration

Registered 2026-09-03 21:39Z (box clock), before the script exists. Lane 1 (CUDA). SIGN CONVENTION (§2135): CE numbers are CE
ADDED ABOVE THE REAL MODEL on held-out docs 0–63 (FRESH split; heads and fillers on docs 96–191; baseline 3.0322401) — LOWER IS
BETTER. Descriptive; nothing installs into §312.

## Question
For mlp16/17 the model's OWN weights on 16 core input coordinates + a filler (§2720) beat any fitted surrogate and compiled into a
tiny exact polynomial (§2727/§2728). For the pool (mlp11–15) the fitted description so far is a linear context map (§2724), whose
input-weighted rank curve (§2726) gives CE .558 / .457 / .410 / .366 at rank 16 / 128 / 256 / 512 (POOL_MEAN .724). Does the same
extraction recipe work for the pool — each block's own weights on x′ = Π_k x̂_l + filler, Π_k = top-k input PCs of the block's own
normalised input — and how does it compare with the fitted map at equal input rank?

## Arms (all five pool blocks patched jointly; output unrestricted; everything else real)
POOL_MEAN (ref .724) · OWN_k_MEAN, k ∈ {16, 32, 64, 128, 256} (filler = per-block mean of x̂_⊥) · OWN_k_TOK, k ∈ {32, 128} (filler =
per-block ridge ê → x̂_⊥, same λ rule as §2720) · reported alongside the §2726 fitted-map values at the same k.

## Predictions (bars fixed now)
- **pred_a_instrument**: baseline within 1e-4; POOL_MEAN within .03 of .724.
- **pred_b_own_weights_on_64_input_pcs_recover_a_third**: CE(OWN_64_MEAN) ≤ .50. Null: ≥ .65.
- **pred_c_256_pcs_recover_most**: CE(OWN_256_MEAN) ≤ .40. Null: ≥ .60.
- **pred_d_token_filler_helps_the_pool_too**: CE(OWN_32_MEAN) − CE(OWN_32_TOK) ≥ .04. Null: ≤ .01.
- **pred_e_own_weights_beat_the_fitted_map_at_rank_128**: CE(OWN_128_MEAN) ≤ .457 − .03 = .427. Null: ≥ .457.

## Price
96 fit docs × 2 passes + 64 × (1 + 8) ≈ 770 GPU document-forwards + five 1152 × 1152 eigendecompositions ≈ 25 s. Output
late_pool_own_weights_input_head_probe_results.json. Frozen: this file, §2726 results (late_pool_map_rank_curve_probe_results.json),
checkpoint, fit_natural.pt.
