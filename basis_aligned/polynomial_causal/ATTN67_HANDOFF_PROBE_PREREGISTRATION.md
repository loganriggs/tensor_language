# attn67_handoff_probe — preregistration (Registered 2026-09-04 00:01Z (box clock))

Lane 1 CUDA (Claude). Follows §2766 (the .020 chain-write cost at k = 1024 is attn6 .0094 + attn7 .0070: the last two attention
writes before the bus begins at block 8 write directions their own block's MLP does not read) and §2765 (a frame that includes
more of the following inputs captures more of them). Hypothesis: the part of attn6/attn7's write that lies outside the next
read's frame is a HAND-OFF into the bus — it lies inside the bus frame U_8 and is read by blocks 8–17. Test: SPLIT8_1024 reads
plus attn6 and attn7's writes constrained jointly in four ways.

Sign convention (§2135): CE numbers are CE ADDED ABOVE THE REAL MODEL on held-out docs 0–63 (FRESH split; fits docs 96–191) —
LOWER IS BETTER. Cost = arm − SPLIT8_1024.

Arms (attn6 and attn7 both; r = centred write's remainder outside span(U_mlp_l), k = 1024): SPLIT8_1024 (instrument);
DEL67 — r deleted (§2766's joint); BUS67 — r replaced by its projection onto span(U_8) (write confined to next frame ∪ bus frame);
BUSONLY67 — the whole centred write projected onto span(U_8) alone; RO67 — r routed to the readout (added before the final norm).

Frozen: this file, §2766 results (early_chain_write_cost_map_probe_results.json), checkpoint, fit_natural.pt.

- pred_a_instrument: baseline 3.0322401 within 1e-4; SPLIT8_1024 within .015 of .0374.
- pred_b_handoff_is_bus_bound: BUS67 − SPLIT8_1024 ≤ .003. Null: ≥ .010.
- pred_c_delete_reproduces_the_map: DEL67 − SPLIT8_1024 in [.010, .025] (§2766: .0094 + .0070 = .0164, ratio 1.25 sub-additive). Null: ≤ .005.
- pred_d_bus_frame_alone_carries_the_writes: BUSONLY67 − SPLIT8_1024 ≤ .005. Null: ≥ .015.
- pred_e_readout_worse_than_delete: RO67 ≥ DEL67 (§2761 pattern: the remainder is consumed by later blocks, not the readout). Null: RO67 ≤ DEL67 − .003.

Price: 1 fit pass (96 docs) + baseline + 5 arms × 64 docs = 480 GPU document-forwards (≈ 25 s). Descriptive; nothing installs
into the §312 frontier; bases are data covariances scored by CE only (§2118 stays closed).
