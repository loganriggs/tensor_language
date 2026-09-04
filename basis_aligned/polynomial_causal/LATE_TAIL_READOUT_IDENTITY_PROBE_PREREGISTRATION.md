# Preregistration — late_tail_readout_identity_probe (Claude, LANE 1 CUDA) — Registered Registered 2026-09-04 02:35Z (box clock) (box clock)

Sign convention (§2135): every CE number is CE ADDED ABOVE THE REAL MODEL on held-out docs 0–63 (FRESH split; fits and unigram counts on docs 96–191) — LOWER IS BETTER. recovered(P) = 1 − KEEP_P / FINAL_ONLY.

## Question
§2800: the unembedding's read of the late MLPs' 384-dim tail write (.1130 when withheld) is a novel-token evidence channel — its damage is monotone in target frequency, from +.285/token on targets unseen in the fit corpus to −.063/token on the most frequent targets. Two accounts fit that: (i) a CALIBRATION signal — one logit-space direction (roughly "rare vs frequent") whose gain the late MLPs set from context; (ii) TOKEN IDENTITY — evidence for WHICH out-of-context token, which is high-rank (§2798, eff rank 261) and could not be a few directions. This rung prices account (i) exactly and asks how much of the channel is left once it is removed.

## Instrument
Parent late_tail_readout_rank_probe (§2798). u = centred log(1 + fit-corpus count) over the vocabulary, unit norm; WU = W_U · Ut (V × 384); d_f = WUᵀu / ‖·‖ — the single tail direction that drives the logits furthest along u per unit of tail write. Arms (readout sees x − A + P Pᵀ A in tail coordinates): FINAL_ONLY (repro .1130); KEEP_ALL (P = I, exactness of the keep machinery: must be 0); KEEP_FREQ1 (P = d_f); DROP_FREQ1 (P = the 383-dim orthonormal complement of d_f); KEEP_WU1 (top right singular vector of WU); KEEP_PCA1 (top activation PC of A's tail); KEEP_RAND1_{0,1,2} (three seeded random unit tail directions; mean reported); KEEP_FREQ1_WU32 (d_f ⊕ WU's top 32, orthonormalised; compared with §2798's recovered(WU, 32) = .193). All arms scored per token (64 × 256) with the target classes of ops/target_token_classes.py and the fit-corpus counts; the mean predictive entropy (nats) is recorded for the baseline and every arm. Also recorded: |cos| of d_f with the top WU and PCA directions, the tail-covariance energy along each, the share of WU's logit energy along d_f. Price ≈ 2 × 96 + 64 × 12 ≈ 960 GPU document-forwards (~30 s).

## Predictions (scored exactly as written)
- pred_a_instrument: |baseline − 3.0322401| ≤ 1e-4; |SPLIT8_1024 − .0374| ≤ .015; |FINAL_ONLY − .1130| ≤ .015; |KEEP_ALL| ≤ 1e-3.
- pred_b_frequency_direction_alone_recovers_a_real_share: recovered(FREQ1) ≥ 0.15. Null: ≤ 0.05.
- pred_c_frequency_direction_beats_the_top_unembed_direction: recovered(FREQ1) ≥ 2 × recovered(WU1). Null: recovered(FREQ1) ≤ recovered(WU1).
- pred_d_channel_is_mostly_identity_not_frequency: DROP_FREQ1 / FINAL_ONLY ≥ 0.85 (removing the frequency direction alone leaves ≥ 85% of the damage in place). Null: ≤ 0.6.
- pred_e_withholding_the_channel_raises_entropy: mean predictive entropy under FINAL_ONLY − baseline ≥ +0.05 nat. Null: ≤ 0.
- pred_f_frequency_direction_value_sits_on_novel_targets: of the per-token CE recovered by KEEP_FREQ1 (delta_FINAL_ONLY − delta_KEEP_FREQ1, summed), the novel-class share ≥ 0.7. Null: ≤ 0.4.

Bars: {"ce_tol": 1e-4, "repro_tol": 0.015, "none_tol": 1e-3, "b_rec": 0.15, "c_mult": 2.0, "d_drop": 0.85, "e_ent": 0.05, "f_share": 0.7}. Null bars: {"b_rec": 0.05, "c_mult": 1.0, "d_drop": 0.6, "e_ent": 0.0, "f_share": 0.4}.

## What each outcome means
b TRUE and d TRUE together: the channel has a real but minor calibration component (one direction worth ~15%) and is otherwise identity — the §2798 high rank is the vocabulary tail, not a frame artefact. d FALSE with null met: most of the .1130 IS one direction and the "high-rank channel" of §2798 was mostly the rare-vs-frequent gain spread across a poorly aligned frame — that would reopen a one-parameter-per-token program item (a context-set rare-token gain). c decides whether the frequency direction is what the unembedding's own top direction already is. e says which way the readout moves without the channel (flatter or over-confident on frequent tokens). f says whether the frequency direction's value is on the novel targets it was meant to serve. Nothing installs into the §312 frontier; the strict explained fraction is unchanged by any outcome.
