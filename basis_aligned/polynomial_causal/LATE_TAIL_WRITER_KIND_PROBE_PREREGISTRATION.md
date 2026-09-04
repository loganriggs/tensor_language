# late_tail_writer_kind_probe — preregistration (Registered 2026-09-04 00:35Z (box clock))

Lane 1 CUDA (Claude). Follows §2777 (the tail the late MLPs read off the top-768 bus directions is 57% late-origin — written by
earlier late blocks, not the block's own attention — and 27% early-origin). Which late WRITER kind builds it? The late-origin
residual splits EXACTLY (scalar λ mixing) into z_a (all attention writes of blocks 8…l, λ-propagated) and z_m (all MLP writes +
Down biases of blocks 8…l−1, λ-propagated): x_l = y_l + z_a + z_m. Arms keep the tail of all but one late writer kind (the dropped
kind's tail replaced by its fit-set mean); the core (top-768) is exact throughout; all reads except the late MLPs' are untouched.

Sign convention (§2135): CE numbers are CE ADDED ABOVE THE REAL MODEL on held-out docs 0–63 — LOWER IS BETTER. Priors: LATE_MLP_768
= .1249 (§2773); EARLY_TAIL_ONLY (both late kinds dropped) = .0711 (§2777).

Arms: SPLIT8_1024 (instrument), LATE_MLP_768 (repro), EARLY_TAIL_ONLY (repro), DROP_LATE_ATTN_TAIL (keep early + MLP-written),
DROP_LATE_MLP_TAIL (keep early + attention-written). Also measured: fit-set tail energy shares of z_a and z_m per block.

Frozen: this file, §2777 results (late_tail_origin_probe_results.json), checkpoint, fit_natural.pt.

- pred_a_instrument: baseline 3.0322401 within 1e-4; SPLIT8_1024 within .015 of .0374; LATE_MLP_768 within .015 of .1249;
  EARLY_TAIL_ONLY within .015 of .0711.
- pred_b_mlp_written_tail_dominates: DROP_LATE_MLP_TAIL ≥ 2 × DROP_LATE_ATTN_TAIL (late attention writes are low-rank, §2692; the
  high-rank MLP writes fill the low-variance directions). Null: DROP_LATE_ATTN_TAIL ≥ DROP_LATE_MLP_TAIL.
- pred_c_mlp_written_tail_is_a_third_of_the_cost: DROP_LATE_MLP_TAIL ≥ 0.35 × LATE_MLP_768. Null: ≤ 0.15 ×.
- pred_d_attention_written_tail_is_minor: DROP_LATE_ATTN_TAIL ≤ 0.25 × LATE_MLP_768. Null: ≥ 0.50 ×.
- pred_e_kinds_complementary_within_late_origin: (DROP_LATE_ATTN_TAIL + DROP_LATE_MLP_TAIL) / EARLY_TAIL_ONLY ∈ [0.7, 1.5]. Null: ≥ 2.0.

Price: 2 fit passes (96 docs each) + baseline + 5 arms × 64 docs = 576 GPU document-forwards (≈ 25 s). Descriptive; nothing installs
into the §312 frontier; bases are data covariances scored by CE only (§2118 stays closed).
