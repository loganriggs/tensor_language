# Is the late shared write core a gain (norm) channel? — preregistration

Registered 2026-09-03 20:46Z (box clock), before the script exists. Lane 1 (CUDA). SIGN CONVENTION (§2135): every number is CE
ADDED ABOVE THE REAL MODEL on held-out docs 0–63 (FRESH split; bases from docs 96–191; baseline 3.0322401) — LOWER IS BETTER.
Descriptive/mechanistic; nothing installs into the §312 frontier.

## Hypothesis under test
§2713: the dictionary shared by the seven late MLP writes (§2710) is the final residual stream's dominant ~19-direction geometry
and faces lm_head weakly (ov .18, chance .11). Every reader downstream sees rms_norm(x) = √D · x/|x|, so a write along the dominant
directions acts mainly as a GAIN on everything else in the stream (it changes |x|, hence how large the informative remainder looks
to the next reader and to lm_head), not as a message in its own direction. If so, removing the late writes' core component while
restoring the token's residual norm should recover most of the loss; keeping the core's direction but discarding its norm effect
should not.

## Arms (patch applied to every late MLP write mlp11–17 at once; per token; core U_k = CORE_TW_k of §2710/§2713, k ∈ {16, 128})
Let w be the site's write, x the residual before it, P = U_kU_kᵀ, w′ = w − Pw, x_o = x + w, x_n = x + w′.
- PLAIN_k: residual becomes x_n (core component dropped: direction AND norm effect lost).
- NORMFIX_k: residual becomes x_n · |x_o| / |x_n| (direction dropped, norm effect kept).
- KEEPDIR_k: residual becomes x_o · |x_n| / |x_o| (direction kept, norm effect dropped).
Control stack: the same three arms on mlp0–6 with their own pooled core EARLY_TW_k (norms over the 1152 dims per token).

## Predictions (bars fixed now)
- **pred_a_instrument**: baseline within 1e-4 of 3.0322401; identity patch 0; CORE_TW eff rank within .5 of 10.004; PLAIN_16
  (late) ≥ .10 nat (otherwise there is nothing to repair and the test is void).
- **pred_b_norm_channel**: NORMFIX_16 ≤ .30 × PLAIN_16 (late). Null: ≥ .80 ×.
- **pred_c_direction_alone_is_not_the_message**: KEEPDIR_16 ≥ .70 × PLAIN_16 (late). Null: ≤ .30 ×.
- **pred_d_holds_at_128**: NORMFIX_128 ≤ .50 × PLAIN_128 (late). Null: ≥ .90 ×.
- **pred_e_early_control**: (NORMFIX_16/PLAIN_16)_early ≥ 2 × (NORMFIX_16/PLAIN_16)_late — the early stack's core is not a gain
  channel. Null: ≤ 1.0 ×.

## Price
96 fit + 64 × (2 + 3 arms × 2 k × 2 stacks) = 992 GPU document-forwards ≈ 20 s. Output late_core_norm_channel_probe_results.json.
Frozen: this file, §2713 results (20ff21d0…), checkpoint, fit_natural.pt.
