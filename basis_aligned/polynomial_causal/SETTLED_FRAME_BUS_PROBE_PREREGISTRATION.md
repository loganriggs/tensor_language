# settled_frame_bus_probe — preregistration (Registered 2026-09-03 23:15Z (box clock))

Lane 1 CUDA (Claude). Follows §2754 (16 own early frames + ONE 1024-frame for blocks 8–17 reads: .0374) and §2748 (late writes
outside the read core are readout-bound). Question: can the WRITES of blocks 8–17 also be confined to that one frame, with the
remainder routed to the readout — i.e. is the settled half of the model a single 1024-dim bus plus a readout side-channel?

Sign convention (§2135): every number is CE ADDED ABOVE THE REAL MODEL on held-out docs 0–63 (FRESH split; fits docs 96–191) —
LOWER IS BETTER. Gaps are "arm − SPLIT8_1024" = extra damage.

Construction: §2754's SPLIT8 read program at k = 1024 (blocks 0–7 own cores; blocks 8–17 the single core U_8 of their 20 sites'
averaged input covariance). Route (§2748) applied to every write of blocks 8–17 (20 sites) with U_8[:, :1024]: the centred write
w − μ_s is split into its in-frame part and its remainder; modes: readout (remainder summed into a buffer added just before the
final norm), delete (remainder dropped), hidden (remainder kept in the stream but subtracted before the final norm). Out-of-frame
write energy per site = 1 − tr(UᵀC_wU)/tr C_w on the fit-set write covariance.

Arms: ALL36_1024 (inst, prior .0337), SPLIT8_1024 (inst, prior .0374), SPLIT8_BUS_1024 (reads + writes readout-routed),
SPLIT8_DEL_1024 (remainder deleted), SPLIT8_HID_1024 (remainder hidden from the readout).

Frozen: this file, §2754 results (settled_frame_split_point_probe_results.json), checkpoint, fit_natural.pt.

- pred_a_instrument: baseline 3.0322401 within 1e-4; ALL36_1024 within .015 of .0337; SPLIT8_1024 within .015 of .0374.
- pred_b_bus_is_free: SPLIT8_BUS_1024 − SPLIT8_1024 ≤ .005. Null: ≥ .020.
- pred_c_remainder_matters: SPLIT8_DEL_1024 − SPLIT8_1024 ≥ .010. Null: ≤ .003.
- pred_d_readout_is_the_main_consumer: (SPLIT8_HID_1024 − SPLIT8_1024) / (SPLIT8_DEL_1024 − SPLIT8_1024) ≥ 0.5 (ratio computed only
  if the denominator ≥ .003; otherwise pred_d is scored FALSE and noted). Null: ratio ≤ 0.2.
- pred_e_out_of_frame_energy_small: median over the 20 sites of blocks 8–17 of out-of-frame write energy at 1024 ≤ .10. Null: ≥ .25.

Price: 1 fit pass (96 docs, read + write covariances) + baseline + 5 arms × 64 docs = 480 GPU document-forwards (≈ 20 s).
Descriptive; nothing installs into the §312 frontier.
