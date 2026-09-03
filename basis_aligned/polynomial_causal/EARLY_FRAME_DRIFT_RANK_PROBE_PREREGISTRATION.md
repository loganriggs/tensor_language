# early_frame_drift_rank_probe — preregistration (Registered 2026-09-03 23:11Z (box clock))

Lane 1 CUDA (Claude). Follows §2753: the early read frames are per-site (leave-one-out +.070, neighbours-only +.081 over own
.057 at k = 768). Question: is the CHANGE of frame from one sublayer to the next LOW-DIMENSIONAL? If site s's 768-frame can be
written as its predecessor's frame with only m directions swapped, the whole-model width program needs one frame plus 21 small
"drift patches" instead of 22 full frames.

Sign convention (§2135): every number is CE ADDED ABOVE THE REAL MODEL on held-out docs 0–63 (FRESH split; fits docs 96–191) —
LOWER IS BETTER. Gaps are "arm − EARLY22_OWN_768" = extra damage.

Construction (§2753 vocabulary, k = 768 throughout). Sequence of the 22 early sites: attn0, mlp0, attn1, mlp1, …, attn10, mlp10;
each site's predecessor is the previous site in the sequence; attn0 (no predecessor) keeps its own core in every arm. For site s
with own centred input covariance C_s and predecessor own core U_p (1152×768):
  PREV:      U_s := U_p (pure carry-over).
  PATCH_m:   energies e_j = u_jᵀ C_s u_j over U_p's columns; keep the 768−m highest-energy columns; take the top-m eigenvectors of
             C_s restricted to span(U_p)^⊥ (the 384-dim complement, via the projector); U_s := [kept | complement top-m] — 768 columns,
             orthonormal by construction.
OwnHead CONST / AttnHead as in §2749–§2753.

Arms: EARLY22_OWN_768 (instrument, prior .057), EARLY22_PREV_768, EARLY22_PATCH32_768, EARLY22_PATCH64_768, EARLY22_PATCH128_768.

Frozen: this file, §2753 results (early_frame_smoothness_probe_results.json), checkpoint, fit_natural.pt.

- pred_a_instrument: baseline 3.0322401 within 1e-4; EARLY22_OWN_768 within .02 of .057.
- pred_b_carry_over_costs: PREV − OWN ≥ .03 (the frames genuinely differ; §2753 NBR +.081). Null: ≤ .01.
- pred_c_drift_is_low_rank_64: PATCH64 − OWN ≤ .010. Null: ≥ .040.
- pred_d_drift_is_low_rank_128: PATCH128 − OWN ≤ .005. Null: ≥ .020.
- pred_e_drift_32: PATCH32 − OWN ≤ .020. Null: ≥ .050.

Price: 1 fit pass (96 docs) + baseline + 5 arms × 64 docs = 480 GPU document-forwards (≈ 55 s). Descriptive; nothing installs
into the §312 frontier.
