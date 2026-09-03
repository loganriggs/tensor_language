# early_chain_write_cost_map_probe — preregistration (Registered 2026-09-03 23:56Z (box clock))

Lane 1 CUDA (Claude). Follows §2764 (confining all 16 early writes to the next site's 1024-frame — the chain — costs +.0199 on
top of the read program SPLIT8_1024 = .0374; the bus writes are free). This rung maps that .020 per site: for each of the 16 early
sites (attn_l, mlp_l, l = 0..7) alone, SPLIT8_1024 reads + that ONE site's write confined to the next site's frame ('delete'
mode, §2764's next_frame: attn_l → U_mlp_l, mlp_l → U_attn_{l+1}, mlp_7 → U_8), and asks whether the per-site costs add up to the
joint cost, how concentrated they are, whether any single write carries the chain, and whether block 0's writes are free.

Sign convention (§2135): CE numbers are CE ADDED ABOVE THE REAL MODEL on held-out docs 0–63 (FRESH split; fits docs 96–191) —
LOWER IS BETTER. Per-site cost c_s = ONE_s − SPLIT8_1024.

Arms: SPLIT8_1024 (instrument), ONE_<site>_1024 for the 16 early sites, CHAIN_ONLY_1024 (all 16 jointly; §2764 .0572).

Frozen: this file, §2764 results (chain_bus_program_statement_probe_results.json), checkpoint, fit_natural.pt.

- pred_a_instrument: baseline 3.0322401 within 1e-4; SPLIT8_1024 within .015 of .0374 and CHAIN_ONLY_1024 within .015 of .0572.
- pred_b_per_site_costs_are_additive: Σ_s c_s / (CHAIN_ONLY_1024 − SPLIT8_1024) in [0.5, 2.0]. Null: ratio ≥ 4 or ≤ 0.25.
- pred_c_cost_is_concentrated: the four largest c_s carry ≥ 60% of Σ_s c_s. Null: ≤ 35% (flat).
- pred_d_no_single_write_carries_the_chain: max_s c_s ≤ .010. Null: ≥ .015.
- pred_e_block0_writes_are_free: c_attn0 + c_mlp0 ≤ .002 (mlp0's write IS the residual that defines attn1's frame, §2762). Null: ≥ .006.

Price: 1 fit pass (96 docs) + baseline + 18 arms × 64 docs = 1248 GPU document-forwards (≈ 60 s). Descriptive; nothing installs
into the §312 frontier; bases are data covariances scored by CE only (§2118 stays closed).
