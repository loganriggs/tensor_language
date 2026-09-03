# embedding_frame_origin_probe — preregistration (Registered Registered 2026-09-03 23:33Z (box clock))

Lane 1 CUDA (Claude). Follows §2757/§2758 and the queued block_boundary_blend_rotation_probe. Every block begins with the blend
x ← λ₀x + λ₁x₀, x₀ = rms-normed token embedding. Block 0's attention input is therefore x₀ itself (scale-free), so attn_0's read
frame IS the embedding frame. Questions: (1) is that frame derivable from the weights alone — the top-768 eigenframe of the
UNWEIGHTED covariance of the rms-normed wte rows over the whole vocabulary — or does it need the token-frequency-weighted data
covariance? (2) does each block's blend pull the read frame BACK toward the embedding frame (distance to U_emb drops across
mlp_l → attn_{l+1})? (3) how far into the network can sites read through the embedding frame at k = 768 before it costs?

Sign convention (§2135): CE numbers are CE ADDED ABOVE THE REAL MODEL on held-out docs 0–63 (FRESH split; fits docs 96–191) —
LOWER IS BETTER; gaps "arm − EARLY22_OWN_768" = extra damage.

Construction: §2753's own top-768 input cores (rms-normed inputs, fit docs 96–191). U_emb = attn_0's own core (the data embedding
frame). U_w = top-768 eigenframe of the centred covariance of rms_norm(wte.weight) over all V rows, equal weights (weights only, no
data). Principal angles as in §2757 (≤ 384 of 768 free); n30 = number of angles > 30°; capture(C, U) = tr(UᵀCU)/tr C.
d_emb(s) = n30(U_emb, U_s) for the 22 early sites. CE arms at k = 768 over the 22 early sites: EARLY22_OWN_768 (all own);
EMB_BLOCKS012_768 (the 6 sites of blocks 0–2 read through U_emb, others own); EMBW_BLOCKS012_768 (same 6 sites through U_w);
EARLY22_EMB_768 (all 22 early sites through U_emb).

Arms: EARLY22_OWN_768, EMB_BLOCKS012_768, EMBW_BLOCKS012_768, EARLY22_EMB_768.

Frozen: this file, §2757 results (frame_principal_angle_spectrum_probe_results.json), checkpoint, fit_natural.pt.

- pred_a_instrument: baseline 3.0322401 within 1e-4; EARLY22_OWN_768 within .02 of .057.
- pred_b_weights_only_frame_suffices: capture(Cx_attn0, U_w) ≥ 0.93 AND EMBW_BLOCKS012_768 − EMB_BLOCKS012_768 ≤ .010.
  Null: capture ≤ 0.80 OR gap ≥ .040.
- pred_c_blend_pulls_toward_embedding: number of boundaries l = 0..7 with d_emb(attn_{l+1}) < d_emb(mlp_l) ≥ 6 of 8. Null: ≤ 3 of 8.
- pred_d_blocks_012_read_the_embedding_frame: EMB_BLOCKS012_768 − EARLY22_OWN_768 ≤ .030. Null: ≥ .080.
- pred_e_embedding_frame_does_not_reach_block_10: EARLY22_EMB_768 − EARLY22_OWN_768 ≥ .060. Null: ≤ .020.

Price: 1 fit pass (96 docs) + baseline + 4 arms × 64 docs = 416 GPU document-forwards (≈ 20 s) + one 1152² eigh on CPU/GPU.
Descriptive; nothing installs into the §312 frontier; U_w is a weight covariance frame scored by CE only (§2118 stays closed).
