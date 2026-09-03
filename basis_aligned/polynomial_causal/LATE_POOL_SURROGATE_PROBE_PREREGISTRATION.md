# One surrogate for the mlp11–15 pool: is it a chain or a one-shot linear map of the block-11 input? — preregistration

Registered 2026-09-03 21:17Z (box clock), before the script exists. Lane 1 (CUDA). SIGN CONVENTION (§2135): CE numbers are CE
ADDED ABOVE THE REAL MODEL on held-out docs 0–63 (FRESH split; ridge fits on docs 96–191; baseline 3.0322401) — LOWER IS BETTER;
rec = 1 − CE(arm)/CE(POOL_MEAN), higher = better. Descriptive surrogate-sufficiency test; nothing installs into §312.

## Question
§2721 found mlp11–15 to be a homogeneous, mutually-redundant pool (uniform +.04 pairwise interactions; joint mean-ablation .724 with
16/17 intact). §2719 found them not to be token lookups (held-out R² .04–.15 on ê). The natural compression target for a pool is
ONE surrogate. Two structural questions decide its form: (1) is a LINEAR map of each block's own rms-normed input enough (per
block, applied in sequence so later blocks read earlier surrogates' writes)? (2) does the chain matter — does a single one-shot
linear map of the block-11 input producing the SUM of the five writes (written at block 11's position) do as well?

## Arms (pool = mlp11..15; all ridge fits centred, λ = 1e-2·tr/nf; x̂_l = rms_norm of block l's MLP input)
- POOL_MEAN: pool → μ_l (reference .724).
- SEQ_LIN: each w_l → μ_l + A_l (x̂_l − x̄_l), A_l 1152 × 1152, applied in sequence on the patched stream.
- SEQ_LIN_TOK: features x̂_l ⊕ ê(t) (2304).
- ONESHOT_LIN: w_11 → Σ_l μ_l + A (x̂_11 − x̄_11) fitted to the SUM of the five real writes; mlp12..15 → 0 write.
- SEQ_R2 held out (full write) per block reported.

## Predictions (bars fixed now)
- **pred_a_instrument**: baseline within 1e-4; POOL_MEAN within .02 of .724.
- **pred_b_linear_in_own_input_recovers_half**: rec(SEQ_LIN) ≥ .50. Null: ≤ .20.
- **pred_c_chain_matters_little**: rec(ONESHOT_LIN) ≥ .80 × rec(SEQ_LIN). Null: ≤ .50 ×.
- **pred_d_token_adds_little**: rec(SEQ_LIN_TOK) ≤ rec(SEQ_LIN) + .05. Null: ≥ + .15.
- **pred_e_writes_are_linearly_predictable_from_input**: median held-out R² of SEQ_LIN over the five blocks ≥ .50. Null: ≤ .25.

## Price
96 × 3 fit passes + 64 × (1 + 4 + 1) ≈ 700 GPU document-forwards + 11 ridge solves (≤ 2304²) ≈ 25 s.
Output late_pool_surrogate_probe_results.json. Frozen: this file, §2721 results (884a0bba…), checkpoint, fit_natural.pt.
