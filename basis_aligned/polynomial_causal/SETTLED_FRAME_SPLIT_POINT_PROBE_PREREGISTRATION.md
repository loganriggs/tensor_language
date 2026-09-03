# settled_frame_split_point_probe — preregistration (Registered 2026-09-03 23:09Z (box clock))

Lane 1 CUDA (Claude). Follows §2753 (one shared read frame for blocks 8–17 costs +.022 over §2751's ALL36 program at k = 768)
and §2751 (ALL36 at 896 / 1024 = .096 / .034). Question: at the program widths that matter (896, 1024), how much does the
"16 own frames for blocks 0–7 + ONE frame for blocks 8–17" program cost over ALL36 (22 own + one late-7 frame), and where does the
settled region causally begin (split at 6, 8, 10)?

Sign convention (§2135): every number is CE ADDED ABOVE THE REAL MODEL on held-out docs 0–63 (FRESH split; fits docs 96–191) —
LOWER IS BETTER. Gaps are "arm − reference" = extra damage.

Construction: exactly §2753's cores/heads (top-k eigenvectors of the plain average of the named sites' centred rms-normed input
covariances; OwnHead CONST filler / AttnHead recompute; block 0 skips value self-mixing). ALL36_k = blocks 0–10 own + one frame
for the 14 late sites (§2751's program). SPLITp_k = blocks 0..p−1 own + one frame fitted on the 2(18−p) sites of blocks p..17.

Arms: ALL36_1024, SPLIT6_1024, SPLIT8_1024, SPLIT10_1024, ALL36_896, SPLIT8_896.

Frozen: this file, §2751 results (whole_model_width_program_probe_results.json), checkpoint, fit_natural.pt.

- pred_a_instrument: baseline 3.0322401 within 1e-4; ALL36_1024 within .015 of .0337; ALL36_896 within .02 of .096.
- pred_b_split8_cheap_at_1024: SPLIT8_1024 − ALL36_1024 ≤ .010. Null: ≥ .030.
- pred_c_split8_cheap_at_896: SPLIT8_896 − ALL36_896 ≤ .015. Null: ≥ .040.
- pred_d_earlier_split_costs: SPLIT6_1024 − SPLIT8_1024 ≥ .003 (blocks 6–7 are not yet settled). Null: ≤ 0.
- pred_e_settled_by_8: SPLIT10_1024 − SPLIT8_1024 ≥ −.003 (moving the split to 10 buys ≤ .003). Null: ≤ −.010.

Price: 1 fit pass (96 docs) + baseline + 6 arms × 64 docs = 544 GPU document-forwards (≈ 60 s). Descriptive; nothing installs
into the §312 frontier.
