# frame_count_by_width_probe — preregistration (Registered 2026-09-03 23:59Z (box clock))

Lane 1 CUDA (Claude). Follows §2765 (at k = 1024 the early frames can be shared: per block +.00005, per pair +.0028, ONE early
frame +.0133 over sixteen own frames — no cliff) and §2753 (at k = 768 sharing one frame across a 3-block early window costs
+.027). The description size of the frame program is (number of frames) × k; this rung measures the sharing penalty at k = 896
and k = 768 with the same four early-frame counts (16 own / 8 per block / 4 per pair / 1), late reads through U_8 (top-k of the
plain average of blocks 8–17's input covariances) in every arm. Reads only (no write constraints).

Sign convention (§2135): CE numbers are CE ADDED ABOVE THE REAL MODEL on held-out docs 0–63 (FRESH split; fits docs 96–191) —
LOWER IS BETTER. Penalty P_k(n) = arm with n early frames at width k − OWN16_k (SPLIT8_k).

Arms: SPLIT8_1024 (instrument), and for k ∈ {896, 768}: OWN16_k, BLOCK8_k, PAIR4_k, EARLY1_k. Shared frames = top-k eigenvectors
of the plain average of the member sites' centred rms-normed input covariances (core_of), as §2765.

Frozen: this file, §2765 results (early_frame_count_probe_results.json), checkpoint, fit_natural.pt.

- pred_a_instrument: baseline 3.0322401 within 1e-4; SPLIT8_1024 within .015 of .0374.
- pred_b_block_frames_cheap_at_896: P_896(8) ≤ .010. Null: ≥ .030.
- pred_c_block_frames_under_p03_at_768: P_768(8) ≤ .030 (§2753's 3-block windows cost +.027). Null: ≥ .060.
- pred_d_one_early_frame_is_a_cliff_at_768: P_768(1) ≥ .100. Null: ≤ .030.
- pred_e_sharing_penalty_grows_as_k_shrinks: P_896(1) ≥ .027 (2× the §2765 value .0133) and P_768(1) ≥ P_896(1). Null: P_896(1) ≤ .015.

Price: 1 fit pass (96 docs) + baseline + 9 arms × 64 docs = 736 GPU document-forwards (≈ 45 s). Descriptive; nothing installs
into the §312 frontier; bases are data covariances scored by CE only (§2118 stays closed).
