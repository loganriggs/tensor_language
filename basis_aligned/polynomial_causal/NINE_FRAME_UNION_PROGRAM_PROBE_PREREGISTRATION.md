# nine_frame_union_program_probe — preregistration (Registered 2026-09-04 00:04Z (box clock))

Lane 1 CUDA (Claude). Follows §2765 (8 per-block early frames are free at 1024; the 9-frame chain+bus program costs .0501), §2767
(block frames stay cheap at 896/768: +.003/+.007), §2766 (the chain-write cost is attn6/attn7) and §2768 (their out-of-frame
write lies in the bus frame: keeping its U_8 part costs +.0009 instead of +.017). This rung states the program v2 and prices it
across k: NINE frames — 8 block frames F_0..F_7 (top-k core of the block's averaged attention+MLP input covariance) and the bus
U_8 — with the UNION write rule: every early write is confined to span(F_next) ∪ span(U_8) (its remainder outside the next
read's frame is replaced by that remainder's projection onto U_8; next frame = F_l for attn_l, F_{l+1} for mlp_l, U_8 for mlp_7);
blocks 8–17 read and write through U_8 with the remainder routed to the readout (§2756).

Sign convention (§2135): CE numbers are CE ADDED ABOVE THE REAL MODEL on held-out docs 0–63 (FRESH split; fits docs 96–191) —
LOWER IS BETTER. Write cost W_k = P9_k − BLOCK8_k (reads only; §2765/§2767 priors .03744 / .1099 / .2253 at 1024/896/768;
BLOCK8_1088 measured here).

Arms: SPLIT8_1024 (instrument), P9_1024, P9_896, P9_768, P9_1088, BLOCK8_1088.

Frozen: this file, §2768 results (attn67_handoff_probe_results.json), checkpoint, fit_natural.pt.

- pred_a_instrument: baseline 3.0322401 within 1e-4; SPLIT8_1024 within .015 of .0374.
- pred_b_nine_frame_program_under_p045_at_1024: P9_1024 ≤ .045 (reads .0374; union writes should cost ≈ .004 by §2766+§2768). Null: ≥ .060.
- pred_c_union_writes_under_p045_at_768: W_768 = P9_768 − .2253 ≤ .045 (the delete chain cost .062 at 768 with own frames, §2761). Null: ≥ .080.
- pred_d_nine_frame_program_under_p13_at_896: P9_896 ≤ .130. Null: ≥ .180.
- pred_e_nine_frame_program_under_p025_at_1088: P9_1088 ≤ .025 (the read cost halves per 128-dim widening: .197/.096/.034). Null: ≥ .045.

Price: 1 fit pass (96 docs) + baseline + 6 arms × 64 docs = 544 GPU document-forwards (≈ 30 s). Descriptive; nothing installs
into the §312 frontier; bases are data covariances scored by CE only (§2118 stays closed).
