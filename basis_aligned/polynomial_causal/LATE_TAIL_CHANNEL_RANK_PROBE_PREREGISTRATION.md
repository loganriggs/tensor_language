# late_tail_channel_rank_probe — preregistration (Registered 2026-09-04 00:38Z (box clock))

Lane 1 CUDA (Claude). Follows §2777/§2778 (the late MLPs' tail read is a cross-block channel: 27% early-origin, 18% late
attention-written, 32% late MLP-written). Is that channel LOW-RANK? The bus U_8 orders directions by the total covariance of
the late inputs, in which the early-origin residual dominates the low-variance energy at blocks 8–11 (§2777). Here the
768-complement is re-ordered by the pooled covariance of the LATE-ORIGIN component (z_a + z_m, λ-propagated writes of the
settled region, scaled to the normalised input, projected off the top-768 core; plain average over the ten late MLP inputs) —
a data covariance of a physically defined part of the residual, scored by CE only. Arms read the core (top-768 of U_8) exactly
plus the top-r channel directions, remainder constant; the fair comparison is the bus's own order at k = 768 + r.

Sign convention (§2135): CE numbers are CE ADDED ABOVE THE REAL MODEL on held-out docs 0–63 — LOWER IS BETTER. Priors (late MLP
reads only, all else exact): LATE_MLP_768 = .1249, LATE_MLP_896 = .0662 (§2773).

Arms: SPLIT8_1024 (instrument), LATE_MLP_768 (repro), BUS_832, BUS_896 (repro), BUS_1024 (bus order, late MLP reads only),
CH_64, CH_128, CH_256 (core 768 + top-r late-origin channel directions). Also measured: the channel covariance's effective
rank and 90%-energy rank.

Frozen: this file, §2778 results (late_tail_writer_kind_probe_results.json), checkpoint, fit_natural.pt.

- pred_a_instrument: baseline 3.0322401 within 1e-4; SPLIT8_1024 within .015 of .0374; LATE_MLP_768 within .015 of .1249;
  BUS_896 within .015 of .0662.
- pred_b_channel_order_beats_bus_order_at_128: CH_128 ≤ BUS_896 − .015. Null: CH_128 ≥ BUS_896 − .003 (no better than the bus order).
- pred_c_channel_order_beats_bus_order_at_64: CH_64 ≤ BUS_832 − .010. Null: CH_64 ≥ BUS_832 − .003.
- pred_d_256_channel_dims_close_the_gap: CH_256 ≤ .010 (core 768 + 256 channel dims recover the k = 1024 late-MLP read within .01).
  Null: ≥ .030.
- pred_e_channel_is_low_rank: effective rank of the pooled late-origin tail covariance ≤ 150 (of the 384-dim complement). Null: ≥ 300.

Price: 2 fit passes (96 docs each) + baseline + 8 arms × 64 docs = 768 GPU document-forwards (≈ 30 s). Descriptive; nothing installs
into the §312 frontier (§2125); the channel basis is a data covariance scored by CE (§2118 stays closed).
