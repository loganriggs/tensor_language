# early_block_read_cost_map_probe — preregistration (Registered 2026-09-04 00:07Z (box clock))

Lane 1 CUDA (Claude). Follows §2769 (program v2 — 8 per-block early frames + bus, union write rule — costs .2419 at k = 768 of
which the writes are +.017: the residual cost is the early READ cost, BLOCK8_768 = .2253) and §2767 (the block is the sharing
unit). This rung maps the k = 768 read cost per early block: for each block l = 0..7 alone, attention and MLP of block l read
through the block frame F_l at k = 768 with every other site of the model untouched (c_l = arm − 0); plus the bus reads alone
(blocks 8–17 through U_8 at 768, early sites untouched).

Sign convention (§2135): CE numbers are CE ADDED ABOVE THE REAL MODEL on held-out docs 0–63 (FRESH split; fits docs 96–191) —
LOWER IS BETTER.

Arms: SPLIT8_1024 (instrument), BLOCK8_768 (reads only, all 8 block frames + bus at 768; §2767 .2253), ONE_B<l>_768 for
l = 0..7, BUS_768 (late reads only).

Frozen: this file, §2769 results (nine_frame_union_program_probe_results.json), checkpoint, fit_natural.pt.

- pred_a_instrument: baseline 3.0322401 within 1e-4; SPLIT8_1024 within .015 of .0374; BLOCK8_768 within .015 of .2253.
- pred_b_block_read_costs_are_additive: (Σ_l c_l + BUS_768) / BLOCK8_768 in [0.5, 1.5]. Null: ≥ 3 or ≤ 0.25.
- pred_c_read_cost_is_concentrated: the two largest c_l carry ≥ 50% of Σ_l c_l. Null: ≤ 30% (flat over 8 blocks).
- pred_d_bus_reads_cheap_at_768: BUS_768 ≤ .050 (the late frame costs ≈ .004 at 1024, §2754). Null: ≥ .100.
- pred_e_no_single_block_is_the_cliff: max_l c_l ≤ .100. Null: ≥ .150.

Price: 1 fit pass (96 docs) + baseline + 11 arms × 64 docs = 864 GPU document-forwards (≈ 35 s). Descriptive; nothing installs
into the §312 frontier; bases are data covariances scored by CE only (§2118 stays closed).
