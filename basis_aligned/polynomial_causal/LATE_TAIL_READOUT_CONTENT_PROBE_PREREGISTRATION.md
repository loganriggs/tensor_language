# Preregistration — late_tail_readout_content_probe (Claude, LANE 1 CUDA) — Registered Registered 2026-09-04 02:29Z (box clock) (box clock)

Sign convention (§2135): every CE number is CE ADDED ABOVE THE REAL MODEL on held-out docs 0–63 (FRESH split; fits on docs 96–191) — LOWER IS BETTER. Per-token deltas are CE added per token under the patch.

## Question
§2797–§2799 closed the late-tail lineage as a description of WHO: the late MLPs (blocks 8–17) write a 384-dim channel outside their shared 768 read frame, the unembedding is its main consumer (.1130 when withheld from the readout alone), the channel is high-rank in every frame (§2798) and its gate is full-width (§2799). This rung asks WHAT the readout gets from it — which tokens pay when the channel is withheld — and whether the writers are one shared channel at the token level or several specialised ones. Compositional reuse: the target classes come from Codex's audited ops/target_token_classes.py (induction: an earlier position with the same current token and the same successor; repeat: target present in the available context but not induction; novel: target not in context).

## Instrument
Parent late_tail_write_consumer_probe (§2797), unchanged: fit means on docs 96–191; SPLIT8_1024 and NONE as scalar repro arms; FINAL_ONLY and FINAL_W{15,16,17} scored PER TOKEN (delta = per-token CE under the patch − per-token baseline CE; 64 × 256 tokens). Axes: class (above); fit-corpus unigram count of the target with bins [0 | 1 | 2–3 | 4–7 | 8–15 | 16–31 | 32–63 | 64–127 | 128–255 | ≥ 256] (rare = count ≤ 3; common = count ≥ 64); baseline-loss bins [<.5 | .5–1 | 1–2 | 2–4 | 4–8 | ≥ 8] nats (confident = baseline < .5); position in 32-token bands (early = 0–31; late = 128–255). Writer coherence: Pearson correlation across the 16,384 tokens between the per-token deltas of FINAL_W17 and FINAL_W15 / FINAL_W16. Recorded besides: top-10% share of the net damage, fraction of tokens helped by the withholding, correlation of the delta with the baseline loss, each writer's per-token damage by class. Price ≈ 96 + 64 × 7 ≈ 544 GPU document-forwards (~25 s).

## Predictions (scored exactly as written)
- pred_a_instrument: |baseline − 3.0322401| ≤ 1e-4; |SPLIT8_1024 − .0374| ≤ .015; |NONE| ≤ 1e-3; |FINAL_ONLY − .1130| ≤ .015.
- pred_b_damage_concentrated_on_few_tokens: the 10% of tokens with the largest per-token delta carry ≥ 0.6 of the net damage (Σ delta). Null: ≤ 0.35.
- pred_c_novel_targets_over_induction_targets: per-token damage on novel targets ≥ 1.2 × per-token damage on induction targets. Null: ≤ 0.8 ×.
- pred_d_confident_tokens_carry_a_large_share: tokens the real model already predicts at baseline loss < .5 nat carry ≥ 0.35 of the net damage. Null: ≤ 0.15.
- pred_e_rare_targets_over_common_targets: per-token damage on rare targets (fit count ≤ 3) ≥ 2.0 × per-token damage on common targets (fit count ≥ 64). Null: ≤ 1.2 ×.
- pred_f_writers_hurt_the_same_tokens: token-level Pearson r(FINAL_W17, FINAL_W15) ≥ 0.25 AND r(FINAL_W17, FINAL_W16) ≥ 0.25. Null: min of the two ≤ 0.1.
- pred_g_early_positions_carry_less: per-token damage at positions 0–31 ≤ 0.6 × per-token damage at positions 128–255. Null: ≥ 0.9 ×.

Bars: {"ce_tol": 1e-4, "repro_tol": 0.015, "none_tol": 1e-3, "b_top10": 0.6, "c_ratio": 1.2, "d_conf": 0.35, "e_ratio": 2.0, "f_r": 0.25, "g_ratio": 0.6}. Null bars: {"b_top10": 0.35, "c_ratio": 0.8, "d_conf": 0.15, "e_ratio": 1.2, "f_r": 0.1, "g_ratio": 0.9}.

## What each outcome means
b TRUE: the channel is a sparse correction path (a few tokens per document pay nearly all of .113) rather than a diffuse calibration. c TRUE: the readout's tail read serves out-of-context (knowledge / bigram-like) completions, not copying — consistent with attention owning induction and the late MLPs owning token-specific evidence; c FALSE with null met would make the tail an induction-sharpening channel. d TRUE: the channel sharpens predictions the model already gets right (a logit-boost channel) rather than rescuing hard tokens. e TRUE: a rare-token channel (the bilinear model's extra dims serve the long tail of the vocabulary). f TRUE: blocks 15–17 write ONE shared channel at the token level (the same tokens pay whichever writer is withheld) — compositional reuse of a channel across writers; f FALSE with null met: the writers are specialised, each serving its own tokens, and "the late tail" is several channels sharing a subspace. g TRUE: the channel needs context to be computed. Nothing installs into the §312 frontier; the strict explained fraction is unchanged by any outcome.
