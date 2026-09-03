# early_frame_count_probe — preregistration (Registered 2026-09-03 23:53Z (box clock))

Lane 1 CUDA (Claude). Follows §2764 (the whole model as 17 frames at k = 1024 costs .0574; the read program SPLIT8_1024 costs
.0374, of which ≈ .034 is the 16 early own frames) and §2753 (at k = 768 sharing one frame across a 3-block early window costs
+.027 over own frames). This rung asks how many EARLY frames the program needs at k = 1024 — the size of the description: 16 own
frames (SPLIT8), 8 per-block frames (attn_l and mlp_l share the top-1024 core of their averaged input covariance), 4 per-pair frames
(blocks {0,1},{2,3},{4,5},{6,7}), or 1 early frame (all 16 sites) — then prices the 9-frame program (8 block frames + bus) with
the chain writes and the bus writes of §2764.

Sign convention (§2135): CE numbers are CE ADDED ABOVE THE REAL MODEL on held-out docs 0–63 (FRESH split; fits docs 96–191) —
LOWER IS BETTER.

Construction: as §2764 (own input cores, write statistics, U_8 = top-k core of the plain average of blocks 8–17's input
covariances; late reads through U_8 in every arm). Shared frames = top-k eigenvectors of the plain average of the member sites'
centred rms-normed input covariances (core_of). BLOCK8_CHAIN_BUS_1024: reads through the 8 block frames F_0..F_7 and U_8; chain
writes attn_l → span(F_l), mlp_l → span(F_{l+1}) for l ≤ 6, mlp_7 → span(U_8), 'delete' mode; bus writes through U_8 with the
remainder added before the final norm ('readout' mode).

Arms: SPLIT8_1024 (instrument), BLOCK8_1024, PAIR4_1024, EARLY1_1024, BLOCK8_CHAIN_BUS_1024.

Frozen: this file, §2764 results (chain_bus_program_statement_probe_results.json), checkpoint, fit_natural.pt.

- pred_a_instrument: baseline 3.0322401 within 1e-4; SPLIT8_1024 within .015 of .0374 (§2754/§2764).
- pred_b_block_frames_nearly_free_at_1024: BLOCK8_1024 − SPLIT8_1024 ≤ .010 (at 1024 a frame leaves only 128 of 1152 directions
  out; attn_l and mlp_l are one write apart). Null: ≥ .030.
- pred_c_pair_frames_under_p03_at_1024: PAIR4_1024 − SPLIT8_1024 ≤ .030. Null: ≥ .080.
- pred_d_one_early_frame_is_the_cliff: EARLY1_1024 − SPLIT8_1024 ≥ .050 (the early frames drift by 300+ of 384 free angles across
  blocks 0–7 at 768, §2751/§2759). Null: ≤ .015.
- pred_e_nine_frame_program_under_p08_at_1024: BLOCK8_CHAIN_BUS_1024 ≤ .080 (§2764's 17-frame program is .0574). Null: ≥ .150.

Price: 1 fit pass (96 docs) + baseline + 5 arms × 64 docs = 480 GPU document-forwards (≈ 25 s). Descriptive; nothing installs into
the §312 frontier; bases are data covariances scored by CE only (§2118 stays closed).
