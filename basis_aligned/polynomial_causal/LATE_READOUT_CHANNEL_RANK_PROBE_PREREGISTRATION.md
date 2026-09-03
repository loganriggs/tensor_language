# Preregistration — late_readout_channel_rank_probe (Claude, lane 1 CUDA)

Registered 2026-09-03 23:01Z (box clock). Follows §2748 (the quarter of each late write outside the 768 read core is readout-bound: routed straight to the final residual
it costs .050; the BUS program costs .105) and §2751 (the BUS form holds in the whole-model program).

## Question
How wide is the readout side-channel? The routed content is the sum over the 14 late sites of (I − P)(w − μ_s); by construction it
lives in the 384-dim complement of the read core. If it is low-rank — a few dozen directions — the late program is "a 768-dim bus
plus a narrow side-channel to the logits", and the readout's use of the late writes is a small object.

## Arms (eval docs 0–63; fits 96–191; P = 768-core of §2745; the side-channel basis Q_r = top-r PCs of the fit-set covariance of the
routed sum, collected in one extra fit pass)
TO_READOUT_768 (§2748 reproduction), BUS_768 (§2748 reproduction).
TO_READOUT_768_R{r} for r ∈ {32, 64, 128, 256}: the routed sum projected onto Q_r (centred on its fit mean) before it is added.
BUS_768_R128: the BUS program with the side-channel truncated to 128 dims.

## Predictions (CE added above the real model, docs 0–63, LOWER IS BETTER — §2135)
- pred_a_instrument: baseline within 1e-4 of 3.0322401; TO_READOUT_768 within .02 of .050; BUS_768 within .02 of .105.
- pred_b_side_channel_128: TO_READOUT_768_R128 − TO_READOUT_768 ≤ .03. Null: ≥ .10.
- pred_c_side_channel_64: TO_READOUT_768_R64 − TO_READOUT_768 ≤ .06. Null: ≥ .15.
- pred_d_bus_with_128_side_channel: BUS_768_R128 − BUS_768 ≤ .03. Null: ≥ .10.
- pred_e_side_channel_is_low_rank: effective rank of the routed sum's centred covariance ≤ 150 (of 384 possible). Null: ≥ 300.
Descriptive: R32 and R256; the routed sum's spectrum (rank_90); its mean energy vs variance.

## Price
96 fit docs × 2 passes + 64 × (1 + 7 arms) = 704 GPU document-forwards, ~17 s. Frozen: this file, §2751 results, checkpoint, fit_natural.pt.
