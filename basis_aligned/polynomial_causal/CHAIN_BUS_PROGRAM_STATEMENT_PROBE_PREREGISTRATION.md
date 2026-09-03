# chain_bus_program_statement_probe — preregistration (Registered Registered 2026-09-03 23:47Z (box clock))

Lane 1 CUDA (Claude). Follows §2756 (blocks 8–17 read and write through one bus frame, remainder routed to the readout: .0362 at
k = 1024) and §2761 (early MLP writes land in the NEXT site's read frame; deleting the next-frame remainder costs .032 for eight
MLP writes and .030 for eight attention writes at k = 768; routing it to the readout is worse than deleting it). This rung states
the whole model as ONE frame program and prices it: every site of blocks 0–7 reads through its own frame and writes INTO the next
site's frame (remainder deleted; mlp_7 writes into the bus frame); blocks 8–17 read and write through the bus frame U_8 with the
remainder routed to the readout. No parameter beyond the 16 early frames and the bus frame is introduced — the write frames ARE the
next reads' frames.

Sign convention (§2135): CE numbers are CE ADDED ABOVE THE REAL MODEL on held-out docs 0–63 (FRESH split; fits docs 96–191) —
LOWER IS BETTER.

Construction: own top-k input cores per site and write statistics as §2756/§2761; U_8 = top-k core of the plain average of blocks
8–17's input covariances. Chain writes: attn_l → span(U_mlp_l), mlp_l → span(U_attn_{l+1}) for l = 0..6, mlp_7 → span(U_8);
'delete' mode. Bus: reads through U_8, writes split with the remainder added before the final norm (§2756 'readout' mode).
Arms: SPLIT8_1024 (§2754's read program: own reads 0–7, U_8 reads 8–17; instrument .0374); CHAIN_ONLY_1024 (SPLIT8 reads + early
chain writes, late writes free); CHAIN_BUS_1024 (chain writes + bus reads/writes with readout remainder); CHAIN_BUS_768 (the same
program at k = 768).

Arms: SPLIT8_1024, CHAIN_ONLY_1024, CHAIN_BUS_1024, CHAIN_BUS_768.

Frozen: this file, §2761 results (early_write_frame_chain_probe_results.json), checkpoint, fit_natural.pt.

- pred_a_instrument: baseline 3.0322401 within 1e-4; SPLIT8_1024 within .015 of .0374 (§2754/§2756/§2758).
- pred_b_early_chain_is_cheaper_at_1024: CHAIN_ONLY_1024 − SPLIT8_1024 ≤ .030 (at 768 the two halves cost .032 + .030). Null: ≥ .080.
- pred_c_whole_program_under_p08_at_1024: CHAIN_BUS_1024 ≤ .080. Null: ≥ .150.
- pred_d_whole_program_under_p35_at_768: CHAIN_BUS_768 ≤ .350 (SPLIT8_768 reads alone are .218, §2753). Null: ≥ .600.
- pred_e_bus_writes_add_nothing_on_top_of_the_chain: CHAIN_BUS_1024 − CHAIN_ONLY_1024 ≤ .005 (§2756: the bus write constraint with
  readout routing cost −.0012 on top of the reads). Null: ≥ .020.

Price: 1 fit pass (96 docs) + baseline + 4 arms × 64 docs = 416 GPU document-forwards (≈ 25 s). Descriptive; nothing installs into
the §312 frontier; bases are data covariances scored by CE only (§2118 stays closed).
