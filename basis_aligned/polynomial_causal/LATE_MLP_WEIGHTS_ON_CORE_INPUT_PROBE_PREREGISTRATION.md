# Do the last two MLPs' OWN WEIGHTS compute the message from (16 core coordinates + token)? — preregistration

Registered 2026-09-03 21:06Z (box clock), before the script exists. Lane 1 (CUDA). SIGN CONVENTION (§2135): CE numbers are CE
ADDED ABOVE THE REAL MODEL on held-out docs 0–63 (FRESH split; fits on docs 96–191; baseline 3.0322401) — LOWER IS BETTER;
rec = 1 − CE(arm)/CE(MEAN(mlp16+17)), higher = better. Extraction check for §2718; nothing installs into §312.

## Question
§2718 FITTED a surrogate μ + P_M[A ê + B c + Q(c,c)] that recovers 78% of the mlp16+17 value. Fitting can succeed for reasons
unrelated to what the block computes. This rung uses NO fitted map inside the block: it feeds the block's own Left/Right/Down a
RESTRICTED input — the real 16 core coordinates of its input plus a filler for the other 1136 directions — and asks how much of
the block's value its own algebra reproduces. If ≥ 60% survives with a token-only filler, the block genuinely computes its
message from (core input, token) and the §2718 program is an extraction, not a regression.

## Arms (mlp16 + mlp17 replaced together; x̂ = rms_norm(x) at the block's MLP input; P_M = §2710 16-dim CORE_TN; x̂_⊥ = (I − P_M) x̂)
- W_MEANFILL: block input x̂′ = P_M x̂ + x̄_⊥ (fit-set mean of x̂_⊥); write = Down[(L x̂′) ⊙ (R x̂′)] (no output projection).
- W_TOKFILL: x̂′ = P_M x̂ + f(ê) where f is a ridge fit ê → x̂_⊥ on docs 96–191 (the token's own filler; no block weights fitted).
- W_TOKFILL_M: W_TOKFILL with the write's non-core component replaced by the mean's: μ + P_M(write − μ).
- W_RANDFILL: x̂′ = P_M x̂ + x̂_⊥ of a RANDOM other position from the same chunk (a wrong but realistic filler; control).
- Reference reproductions: MEAN (.848), ORACLE_M (.158).

## Predictions (bars fixed now)
- **pred_a_instrument**: baseline within 1e-4; MEAN within .02 of .848; ORACLE_M within .02 of .158.
- **pred_b_own_weights_on_core_plus_token**: rec(W_TOKFILL) ≥ .60. Null: ≤ .30.
- **pred_c_token_filler_beats_mean_filler**: rec(W_TOKFILL) ≥ rec(W_MEANFILL) + .15. Null: ≤ rec(W_MEANFILL) + .03.
- **pred_d_output_lives_in_the_core_here_too**: rec(W_TOKFILL_M) ≥ .90 × rec(W_TOKFILL). Null: ≤ .60 ×.
- **pred_e_realistic_wrong_filler_hurts**: rec(W_RANDFILL) ≤ rec(W_TOKFILL) − .15. Null: ≥ rec(W_TOKFILL) − .03.

## Price
96 × 2 fit passes + 64 × (1 + 6) ≈ 640 GPU document-forwards + one 1152² ridge per site ≈ 15 s.
Output late_mlp_weights_on_core_input_probe_results.json. Frozen: this file, §2718 results (0993eb5d…), checkpoint, fit_natural.pt.
