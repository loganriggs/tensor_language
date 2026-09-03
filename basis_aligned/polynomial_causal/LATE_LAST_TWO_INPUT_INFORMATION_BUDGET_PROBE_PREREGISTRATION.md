# Preregistration — late_last_two_input_information_budget_probe

Registered 2026-09-03 22:14Z (box clock). Claude lane. Follows §2737 (the core's per-token variation is worth .176 to the real
mlp16/17; the exact-core program costs .246, so the non-core input must carry more). Nothing here installs into the §312 frontier.

SIGN CONVENTION (§2135): every CE number is CE ADDED ABOVE THE REAL MODEL on held-out docs 0-63 (FRESH split; fits docs 96-191)
— LOWER IS BETTER.

## Question

Where, in the rms-normed input of mlp16 and mlp17, is the per-token information the real blocks use? Mean-preserving budget: "pin"
a subspace = replace the input's coordinates in it by their fit-set means (constant kept, variation removed); "keep" a subspace =
pin its orthogonal complement. Subspaces: the block's OWN top-k input PCs (k = 16, 64, 256, 512; per-block centred input covariance
on the fit set) and the shared 16-dim late core P.

## Instrument (all arms patch mlp16 AND mlp17 together; each block uses its own PCs and its own means)

- PIN_OWN_k, KEEP_OWN_k for k ∈ {16, 64, 256, 512}; PIN_CORE16 (anchor, §2737 .176); KEEP_CORE16.
- Derived: budget curve KEEP_OWN_k vs k; complement check PIN_OWN_k + KEEP_OWN_k vs the full-pin cost.

## Predictions (bars fixed before the run)

- pred_a_instrument: baseline within 1e-4 of 3.0322401; PIN_CORE16 within .02 of .176.
- pred_b_top256_own_pcs_suffice: CE(KEEP_OWN_256) ≤ .15. Null: ≥ .30.
- pred_c_sixteen_own_pcs_do_not: CE(KEEP_OWN_16) ≥ .40. Null: ≤ .25.
- pred_d_own_pcs_beat_the_core_as_a_16_dim_channel: CE(KEEP_CORE16) − CE(KEEP_OWN_16) ≥ .05. Null: ≤ −.05.
- pred_e_top16_variation_is_not_critical: CE(PIN_OWN_16) ≤ .25. Null: ≥ .45.

## Price

Fits (192 fit docs) + 10 arms × 64 eval docs ≈ 900 GPU doc-forwards ≈ 20 s on the 5090. Output:
late_last_two_input_information_budget_probe_results.json. Frozen: this file, late_square_direction_mean_control_probe_results.json
(§2737), late_stack_extracted_program_probe_results.json (§2732), checkpoint, fit_natural.pt.

## What each outcome means

b,c TRUE: the last two blocks' per-token input lives in a ~64-256-dim variance band, not in a 16-dim head — the right shape for an
extracted program is "constant offsets + a few hundred input directions", which is what §2735's OWN_256 (.074 with a token filler)
already suggested. d TRUE: the block's own PCA is the better channel than the shared core; d FALSE: the shared core is the better
16-dim channel despite being mean-dominated — reuse beats per-block variance. e FALSE: the top-16 own PCs are themselves critical.
