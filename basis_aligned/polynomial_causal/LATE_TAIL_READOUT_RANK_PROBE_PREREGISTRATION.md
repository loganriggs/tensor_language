# Preregistration — late_tail_readout_rank_probe (Claude, LANE 1 CUDA) — Registered 2026-09-04 02:14Z (box clock)

Sign convention (§2135): every CE number is CE ADDED ABOVE THE REAL MODEL on held-out docs 0–63 (FRESH split; fits on docs 96–191) — LOWER IS BETTER.

## Question
§2797: the late MLPs' tail writes (blocks 8–17, the 384 dims outside the bus-768 core, centred, λ-propagated in one accumulator A) cost .1130 when withheld from the FINAL READOUT alone — the unembedding is the tail's main consumer. How many tail dimensions does the unembedding actually use, and are they the tail's own principal directions or the unembedding's? §2786 found the tail's dominant activation directions only moderately aligned with the unembedding's top-128 (cosines .37/.35), so the two frames may differ.

## Instrument
Parent late_tail_write_consumer_probe (§2797). Fit pass (docs 96–191): the covariance of A's tail coordinates at the final site → activation frame PCA (top-k eigenvectors). Weight frame WU: right singular vectors of W_U · Ut (V × 384), i.e. the tail directions ordered by the logit energy they produce. Arms: SPLIT8_1024 (repro .0374); FINAL_ONLY (repro §2797's .1130); KEEP_{PCA,WU}{8,32,128}: the readout sees x − A + P Pᵀ A (in tail coordinates), i.e. it is denied the late-MLP tail write except its component in the k-dim frame. recovered(frame, k) = 1 − KEEP/FINAL_ONLY. Also recorded: the tail-covariance energy each frame captures, the effective ranks of A's covariance and of W_U's tail restriction, the top-32 frame overlap. Price ≈ 2 × 96 + 64 × 9 ≈ 770 GPU document-forwards (~25 s).

## Predictions (scored exactly as written)
- pred_a_instrument: |baseline − 3.0322401| ≤ 1e-4; |SPLIT8_1024 − .0374| ≤ .015; |FINAL_ONLY − .1130| ≤ .015.
- pred_b_unembed_frame_k32_recovers_most: recovered(WU, 32) ≥ 0.7. Null: ≤ 0.4.
- pred_c_unembed_frame_beats_activation_pca_at_k32: recovered(WU, 32) ≥ recovered(PCA, 32) + 0.1. Null: recovered(WU, 32) ≤ recovered(PCA, 32).
- pred_d_unembed_frame_k8_recovers_a_large_share: recovered(WU, 8) ≥ 0.4. Null: ≤ 0.15.
- pred_e_k128_recovers_nearly_all_in_either_frame: max(recovered(WU, 128), recovered(PCA, 128)) ≥ 0.9. Null: ≤ 0.6.

Bars: {"ce_tol": 1e-4, "repro_tol": 0.015, "b_rec": 0.7, "c_margin": 0.1, "d_rec": 0.4, "e_rec": 0.9}. Null bars: {"b_rec": 0.4, "d_rec": 0.15, "e_rec": 0.6}.

## What each outcome means
b TRUE: the readout channel is low-rank on the unembedding's side (≤ 32 of 384 directions carry ≥ 70% of .113) — a small, weight-defined interface between the late MLPs and the vocabulary. c TRUE: the right frame is the unembedding's, not the activations' (the tail's big activation directions are not what the readout uses — consistent with §2786's weak alignment). b FALSE with null met: the readout uses the tail broadly (rank ≳ 128) and there is no small interface; e then decides whether even 128 suffice. Nothing installs into the §312 frontier; the strict explained fraction is unchanged by any outcome.
