# late_tail_origin_probe — preregistration (Registered 2026-09-04 00:30Z (box clock))

Lane 1 CUDA (Claude). Follows §2776 (82% of the late MLPs' tail read is contextual). WHERE does the contextual tail come from? The
residual entering late block l decomposes EXACTLY (the block mixing x ← λ0·x + λ1·x0 is scalar-linear) into an EARLY-ORIGIN part
y_l (the residual at block 8's entry, propagated through the λ mixing with the embedding x0, no late writes) and a LATE-ORIGIN
part x_l − y_l (every write of blocks 8…l−1 plus block l's attention). Arms keep ONE origin's tail (off the top-768 bus
directions) in each late MLP's input and replace the other's tail by its fit-set mean; the core (top-768) is exact throughout.
A third split isolates block l's OWN attention write.

Sign convention (§2135): CE numbers are CE ADDED ABOVE THE REAL MODEL on held-out docs 0–63 — LOWER IS BETTER. Prior: LATE_MLP_768
(both tails dropped, constant fill) = .1249 (§2773).

Arms (all blocks 8–17 MLP reads, k = 768): SPLIT8_1024 (instrument), LATE_MLP_768 (repro, drop both tails), EARLY_TAIL_ONLY (drop the
late-origin tail), LATE_TAIL_ONLY (drop the early-origin tail), DROP_OWN_ATTN_TAIL (drop only block l's own attention write's tail),
OWN_ATTN_TAIL_ONLY (keep only block l's own attention write's tail). Also measured: the fit-set energy shares of the tail by origin.

Frozen: this file, §2776 results (late_tail_token_fill_probe_results.json), checkpoint, fit_natural.pt.

- pred_a_instrument: baseline 3.0322401 within 1e-4; SPLIT8_1024 within .015 of .0374; LATE_MLP_768 within .015 of .1249.
- pred_b_late_origin_tail_carries_the_cost: EARLY_TAIL_ONLY ≥ 0.60 × LATE_MLP_768 (the tail the MLPs need is written inside the
  settled region — the compounding of §2771/§2774). Null: ≤ 0.30 ×.
- pred_c_early_origin_tail_is_minor: LATE_TAIL_ONLY ≤ 0.50 × LATE_MLP_768. Null: ≥ 0.90 ×.
- pred_d_own_attention_tail_is_minor: DROP_OWN_ATTN_TAIL ≤ 0.30 × LATE_MLP_768. Null: ≥ 0.60 ×.
- pred_e_origins_complementary: (EARLY_TAIL_ONLY + LATE_TAIL_ONLY) / LATE_MLP_768 ∈ [0.7, 1.5]. Null: ≥ 2.0.

Price: 2 fit passes (96 docs each) + baseline + 6 arms × 64 docs = 640 GPU document-forwards (≈ 30 s). Descriptive; nothing installs
into the §312 frontier; bases are data covariances scored by CE only (§2118 stays closed).
