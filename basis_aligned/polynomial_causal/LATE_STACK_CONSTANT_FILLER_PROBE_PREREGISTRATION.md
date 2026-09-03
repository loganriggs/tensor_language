# Preregistration — late_stack_constant_filler_probe

Registered 2026-09-03 22:18Z (box clock). Claude lane. Follows §2738 (mlp16/17: own top-16 input PCs + a CONSTANT filler cost .243, equal to the exact
core program with a token filler) and §2735 (POOL(k=32, token filler) + OWN_256(token filler) = .508, the best extracted late stack).
Nothing here installs into the §312 frontier.

SIGN CONVENTION (§2135): every CE number is CE ADDED ABOVE THE REAL MODEL on held-out docs 0-63 (FRESH split; fits docs 96-191)
— LOWER IS BETTER.

## Question

Is the token filler needed anywhere in the late stack? Each of mlp11–17 gets its own weights on its own top-k input PCs; the other
1152−k input dimensions are either the fit-set constant mean ("CONST") or the constant plus the rank-full ridge read of the current
token's embedding ("TOK", §2730 recipe). If CONST composes as well as TOK, the late stack is "seven blocks × (k directions + one
constant vector)" with no token lookup at all.

## Instrument (own weights everywhere; output unrestricted)

- POOL32_TOK (mlp11–15, k = 32, token filler; reference .319) and POOL32_CONST.
- LAST2_CONST_k (mlp16/17 only) for k ∈ {16, 64, 256} (anchors: §2738 KEEP_OWN_k .243 / .172 / .085).
- POOL32_TOK + LAST2_CONST_k for k ∈ {16, 64, 256} (compare §2735 POOL+OWN_k(TOK) .823 / .678 / .508).
- ALL7_CONST_k (all seven blocks, same k) for k ∈ {32, 64, 128, 256}; ALL7_TOK_256 for comparison.
- Derived: token-filler value per arm = CONST − TOK at matched k; composition penalties π as in §2733.

## Predictions (bars fixed before the run)

- pred_a_instrument: baseline within 1e-4 of 3.0322401; POOL32_TOK within .02 of .319; LAST2_CONST_64 within .02 of .172.
- pred_b_pool_needs_no_token_filler: CE(POOL32_CONST) ≤ .40. Null: ≥ .55.
- pred_c_constant_last_two_compose_as_well: CE(POOL32_TOK + LAST2_CONST_64) ≤ .70. Null: ≥ .80.
- pred_d_all_constant_stack_at_256: CE(ALL7_CONST_256) ≤ .55. Null: ≥ .70.
- pred_e_all_constant_stack_at_64: CE(ALL7_CONST_64) ≤ .75. Null: ≥ .90.

## Price

Fits (192 fit docs, one stats pass) + 14 arms × 64 eval docs ≈ 1,100 GPU doc-forwards ≈ 25 s on the 5090. Output:
late_stack_constant_filler_probe_results.json. Frozen: this file, late_last_two_error_correction_probe_results.json (§2735),
late_last_two_input_information_budget_probe_results.json (§2738), checkpoint, fit_natural.pt.

## What each outcome means

b–e TRUE: the late stack needs no token term — its per-token input is entirely in the stream, and the extracted program is own
weights on k directions plus one constant per block. b FALSE: the pool's token filler is real information (the pool's non-PC input
carries token-specific content the stream's top PCs do not). d/e FALSE with c TRUE: the constant filler is fine for the last two
blocks but not for the pool at matched k.
