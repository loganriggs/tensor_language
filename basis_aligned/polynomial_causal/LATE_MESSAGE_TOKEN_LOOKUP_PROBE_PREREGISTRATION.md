# Is the late MLP message a current-token lookup? — preregistration

Registered 2026-09-03 20:56Z (box clock), before the script exists. Lane 1 (CUDA). SIGN CONVENTION (§2135): every CE number is CE
ADDED ABOVE THE REAL MODEL on held-out docs 0–63 (FRESH split; all fits on docs 96–191; baseline 3.0322401) — LOWER IS BETTER.
Recovery fractions rec = 1 − CE(arm)/CE(MEAN) are HIGHER = better. Descriptive surrogate test; nothing installs into the §312
frontier.

## Question
§2716: replacing mlp16 and mlp17's writes by their means costs .848 nat (mlp17 alone .365); what they add is low-rank (§2710:
eff rank 10–20) and token-varying. Program-structure question: is that message a function of the CURRENT token — a linear
lookup on the token embedding, μ + A·ê(t), i.e. 1152×1152 (or 1152×16 when restricted to the core) numbers per site instead of a
bilinear MLP — of the PREVIOUS token, or of the context?

## Arms (patches on mlp16 and mlp17 together = "last2"; the write w is replaced per token)
- MEAN: w → μ_s (§2716 reference, .848).
- CUR: w → μ_s + A_s ê(t_cur), A_s ridge-fitted on docs 96–191 from ê = rms_norm(wte(t)) to the centred write (λ = 1e-2·tr(ΦᵀΦ)/1152).
- CUR_M: w → μ_s + P_M A_s ê(t_cur) — the lookup restricted to the 16-dim late core (P_M = CORE_TW_16 projector).
- PREV: as CUR with ê(t_{cur−1}) (position 0 uses its own token).
- ORACLE_M: w → μ_s + P_M (w − μ_s) — the real core component, tail dropped: the ceiling for any 16-dim message.
- CUR_17: CUR on mlp17 alone (reference MEAN_17 = .365).
rec(arm) = 1 − CE(arm)/CE(MEAN) for last2 arms; rec_17 likewise against MEAN_17. Held-out R² of each ridge fit is reported.

## Predictions (bars fixed now)
- **pred_a_instrument**: baseline within 1e-4; MEAN within .02 of .848; MEAN_17 within .02 of .365; ridge fits have held-out R² > 0.
- **pred_b_current_token_lookup**: rec(CUR) ≥ .50. Null: ≤ .20.
- **pred_c_lookup_value_lives_in_the_core**: rec(CUR_M) ≥ .80 × rec(CUR). Null: ≤ .40 ×.
- **pred_d_previous_token_weak**: rec(PREV) ≤ .50 × rec(CUR). Null: ≥ 1.0 ×.
- **pred_e_core_ceiling**: rec(ORACLE_M) ≥ .70. Null: ≤ .40.

## Price
96 fit + 64 × (1 + 7 arms) = 608 GPU document-forwards + two 1152² ridge solves ≈ 15 s. Output
late_message_token_lookup_probe_results.json. Frozen: this file, §2716 results (e14e9d8f…), checkpoint, fit_natural.pt.
