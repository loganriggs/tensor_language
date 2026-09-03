# residual_energy_token_quantile_probe — preregistration (Registered Registered 2026-09-03 23:45Z (box clock))

Lane 1 CUDA (Claude). Follows §2762, whose energy budget (mlp_0 writes 2.5 × 10⁹ per token on average against ‖x₀‖² = 1152; the
blend's embedding term is < 1% of the carried residual from block 2 on) is a table of MEANS over tokens and was flagged there as
possibly carried by a few massive-norm tokens (the massive-activation / attention-sink regime: position 0 or delimiter tokens with
10³ × the typical norm). This rung measures the per-token distribution of the same squared norms so that §2762's ratios are
re-stated at the typical token and the massive-token share is known.

Sign convention (§2135): the one CE number (instrument) is CE ADDED ABOVE THE REAL MODEL on held-out docs 0–63 — LOWER IS BETTER.
Everything else is a squared norm per token (‖x₀‖² = 1152).

Construction: one collect pass over fit docs 96–191 (96 × 256 = 24 576 tokens) keeping, per token, ‖post(l)‖², ‖pre(l)‖²,
‖aw_l‖², ‖mw_l‖² for every block (post = attn_l's input after the blend; pre = after mlp_l's write; raw writes). For each quantity
and block: median, p99, max, share of the total carried by the top 1% of tokens (246 tokens), share carried by position-0 tokens
(96 tokens = 0.39%). Typical-token blend ratio r̃(l) = median over tokens of λ₁(l)²·1152 / (λ₀(l)²·‖pre(l−1)‖²(t)) for l = 1..7.
Instrument arm EARLY22_OWN_768 as §2753.

Arms: EARLY22_OWN_768 (instrument only).

Frozen: this file, §2762 results (early_residual_energy_budget_probe_results.json), checkpoint, fit_natural.pt.

- pred_a_instrument: baseline 3.0322401 within 1e-4; EARLY22_OWN_768 within .02 of .057.
- pred_b_massive_tokens_carry_mlp0: the top 1% of tokens carry ≥ 50% of the total ‖pre(0)‖² energy. Null: ≤ 5% (a uniform-ish
  distribution gives ≈ 1–3%).
- pred_c_position0_is_the_sink: position-0 tokens carry ≥ 30% of the total ‖pre(0)‖² energy. Null: ≤ 2%.
- pred_d_mlp0_is_large_at_the_typical_token: median over tokens of ‖mw_0‖² ≥ 100 × 1152 = 1.152 × 10⁵. Null: ≤ 10 × 1152.
- pred_e_blend_is_minor_at_the_typical_token: median over l = 2..7 of r̃(l) ≤ 0.25. Null: ≥ 1.

Price: 1 stats pass + 1 collect pass (96 docs each) + baseline + 1 arm × 64 docs = 320 GPU document-forwards (≈ 20 s). Descriptive;
nothing installs into the §312 frontier.
