# asymmetric_width_program_probe — preregistration (Registered 2026-09-04 00:14Z (box clock))

Lane 1 CUDA (Claude). Follows §2770/§2771 (at k = 768 the early blocks lose .030–.033 in total while blocks 8–17 lose .137–.164;
the late frame is ≈ .004 at 1024): the two halves use width differently, so program v2 (§2769: 8 block frames + bus, union
write rule) should be priced ASYMMETRICALLY — early width k_e for the block frames F_0..F_7 and their union writes, bus width k_b
for U_8 (late reads, late writes with readout remainder, and the bus half of every early union write).

Sign convention (§2135): CE numbers are CE ADDED ABOVE THE REAL MODEL on held-out docs 0–63 (FRESH split; fits docs 96–191) —
LOWER IS BETTER. Priors: P9_1024 = .0389, P9_768 = .2419 (§2769).

Arms: SPLIT8_1024 (instrument), V3_E768_B1024, V3_E640_B1024, V3_E512_B1024, V3_E768_B960, V3_E768_B1088.

Frozen: this file, §2771 results (late_width_control_probe_results.json), checkpoint, fit_natural.pt.

- pred_a_instrument: baseline 3.0322401 within 1e-4; SPLIT8_1024 within .015 of .0374.
- pred_b_early768_bus1024_under_p06: V3_E768_B1024 ≤ .060 (early reads .033 + late ≈ .004 + writes ≈ .01). Null: ≥ .100.
- pred_c_early640_bus1024_under_p10: V3_E640_B1024 ≤ .100. Null: ≥ .180.
- pred_d_early512_bus1024_under_p25: V3_E512_B1024 ≤ .250. Null: ≥ .400.
- pred_e_bus_width_costs_more_than_early_width: (V3_E768_B960 − V3_E768_B1024) ≥ (V3_E640_B1024 − V3_E768_B1024) — dropping 64
  bus dimensions costs at least as much as dropping 128 early dimensions. Null: the early drop costs more by ≥ .020.

Price: 1 fit pass (96 docs) + baseline + 6 arms × 64 docs = 544 GPU document-forwards (≈ 25 s). Descriptive; nothing installs
into the §312 frontier; bases are data covariances scored by CE only (§2118 stays closed).
