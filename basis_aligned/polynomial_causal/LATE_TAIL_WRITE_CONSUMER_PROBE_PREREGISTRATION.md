# Preregistration — late_tail_write_consumer_probe (Claude, LANE 1 CUDA) — Registered 2026-09-04 02:10Z (box clock)

Sign convention (§2135): every CE number is CE ADDED ABOVE THE REAL MODEL on held-out docs 0–63 (FRESH split; fits on docs 96–191) — LOWER IS BETTER.

## Question
§2796 priced the late MLPs' tail writes at the WRITE SITE (all downstream consumers affected): .1459 for all three terms, .1244 for the core-only term alone — as much as confining every late MLP reader to the core (.1249). The writer-kind probe priced the same MLP-written tail at the later MLP READERS' inputs only: .0404. Who consumes the remaining ~.10 — later attention (whose total tail use is ≤ .0153, §2787) or the FINAL READOUT? This rung measures the decomposition directly instead of inferring it by subtraction.

## Instrument
Parent forward (late_tail_writer_identity_probe lineage). The centred tail writes of blocks 8–17's MLPs — perp(write) − μ_l, perp = projection onto the 384 dims outside the bus-768 core Uk of SET8, μ_l = fit-set mean of perp(write) — are accumulated in ONE accumulator A, multiplied by lambdas[0] at every later block entry (Tracker semantics, exact). A named consumer set computes from x − A: MLP l ∈ set normalises x − A; attention l ∈ set is recomputed exactly (own weights, block-0 value residual recomputed) from x − A; the final readout normalises x − A. The stream itself keeps every real write, so consumers NOT named see the tail exactly as the real model does. With all three consumers named, x − A is exactly §2796's write-site drop (all writes downstream are the modified ones and the accumulator holds the modified tail writes), so ALL must reproduce DROP_ALL_TAILOUT = .1459 within the CUDA reproduction tolerance.

Arms (all on docs 0–63): SPLIT8_1024 (repro .0374), NONE (no consumer; exactness), ALL, MLP_ONLY, ATTN_ONLY, FINAL_ONLY, NOT_FINAL (MLPs + attention), FINAL_W{j} for j = 8..17 (writer j alone, readout alone). 1 + 17 arms × 64 docs + 96 fit docs ≈ 1250 GPU document-forwards (~50 s).

## Predictions (scored exactly as written)
- pred_a_instrument: |baseline − 3.0322401| ≤ 1e-4; |SPLIT8_1024 − .0374| ≤ .015; |NONE| ≤ 1e-3; |ALL − .1459| ≤ .015.
- pred_b_readout_consumes_more_than_later_mlps: FINAL_ONLY / MLP_ONLY ≥ 1.5. Null: ≤ 0.8.
- pred_c_late_attention_consumes_little: ATTN_ONLY ≤ .03. Null: ≥ .06.
- pred_d_consumer_prices_additive: (MLP_ONLY + ATTN_ONLY + FINAL_ONLY) / ALL ∈ [0.7, 1.4]. Null: outside [0.5, 2.0].
- pred_e_readout_is_the_majority_consumer: (ALL − NOT_FINAL) / ALL ≥ 0.5. Null: ≤ 0.25.
- pred_f_readout_consumption_concentrated_in_blocks_15_17: Σ_{j≥15} FINAL_Wj / Σ_j FINAL_Wj ≥ 0.5. Null: ≤ 0.3.

Bars: {"ce_tol": 1e-4, "repro_tol": 0.015, "none_tol": 1e-3, "b_ratio": 1.5, "c_max": 0.03, "d_lo": 0.7, "d_hi": 1.4, "e_frac": 0.5, "f_frac": 0.5}. Null bars: {"b_ratio": 0.8, "c_min": 0.06, "d_lo": 0.5, "d_hi": 2.0, "e_frac": 0.25, "f_frac": 0.3}.

## What each outcome means
b and e TRUE: the "tail" outside the late MLPs' shared 768 frame is mainly a READOUT channel — written by the late MLPs from their core (§2796) and consumed by the unembedding; the late MLPs' own tail reads (.0404) are the minority use. b or e FALSE with the null met: the subtraction inference in §2796 was wrong and the write-site price comes from cascade effects among the late MLPs themselves (non-additivity across readers), which pred_d would then also show. f: whether the readout's consumption is dominated by the last three writers (the writers grow 5× with depth, §2793) or is spread across blocks 8–17.

Price: ≈ 1250 GPU document-forwards, ~50 s lane-1. Prior: §2796 late_tail_write_origin_probe_results.json (frozen). Nothing installs into the §312 frontier; the strict explained fraction is unchanged by any outcome.
