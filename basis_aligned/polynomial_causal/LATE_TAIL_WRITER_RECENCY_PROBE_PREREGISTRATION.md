# late_tail_writer_recency_probe — preregistration (Registered 2026-09-04 01:28Z (box clock))

Lane 1 (Claude). Parent: ops/late_tail_writer_kind_probe.py (§2777/§2778). Prior results frozen: early_attn_tail_read_probe_results.json (§2789).
Script: bilinear_quotient/ops/late_tail_writer_recency_probe.py. Results: bilinear_quotient/late_tail_writer_recency_probe_results.json.

SIGN CONVENTION (§2135): every number below is CE ADDED ABOVE THE REAL MODEL on held-out docs 0-63 (FRESH split) — LOWER IS BETTER.

## Question
§2777/§2778 showed the ~300-dim isotropic tail the late MLPs read is mostly LATE-origin (dropping all late-written tail costs .0711 =
EARLY_TAIL_ONLY; dropping only the block's own attention write costs .0023), and §2780–§2788 showed each late MLP reads that tail through a
core-gated linear read. Open: is that communication BANDED (each MLP reads what the last block or two wrote) or SPREAD (an accumulated
signal from many blocks back)? The answer decides whether the program's tail channel can be modelled as short-range block-to-block wires
or must be a bus.

## Method (exact decomposition, no fitting beyond means)
At every late block the residual is x = y + Σ_j c_j exactly, where y is the early-origin part and c_j the λ-propagated write of late
block j (attention write added at ('attn', j); MLP write + Down_bias at ('mlp', j); every c_j multiplied by λ0 at each later block
entry). Reader l (blocks 8–17) sees its top-768 core exactly and a tail from which the writers in the window are removed:
xp = xh − perp(Σ_{j∈S(l,win)} (c_j·scale − mean_fit(c_j·scale))), scale = ‖xh‖/‖x‖, perp = complement of the 768 core.
Window S by distance d = l − j: D_LE0 (own attention write only), D_LE1, D_LE2, D_LE4, D_GT2, D_GT4, DROP_ALL (every late writer ≤ l;
identical to the parent's 'both' = EARLY_TAIL_ONLY). Fit pass (docs 96–191) records the per-(reader, writer) means and the pooled
tail energy of each window's composite and of each distance. Instruments: SPLIT8_1024, LATE_MLP_768.

## Arms
SPLIT8_1024, LATE_MLP_768, DROP_ALL, D_LE0, D_LE1, D_LE2, D_LE4, D_GT2, D_GT4 (+ EXACT_CHECK in smoke only).
cost_share(W) = CE(W)/CE(DROP_ALL); energy_share(W) = pooled tail energy of W's composite / pooled tail energy of DROP_ALL's composite.

## Priors
PRIOR_BASE 3.0322401; SPLIT8_1024 .0374; LATE_MLP_768 .1249; EARLY_TAIL_ONLY .0711 (§2777, = DROP_ALL here); DROP_OWN_ATTN_TAIL .0023 (§2777, = D_LE0 here).

## Predictions (scored exactly as written)
- pred_a_instrument: baseline within 1e-4 of 3.0322401; SPLIT8_1024, LATE_MLP_768, DROP_ALL, D_LE0 within 0.015 of their priors.
- pred_b_recent_two_blocks_are_a_minority (SPREAD hypothesis): cost_share(D_LE2) ≤ 0.50. Null: cost_share(D_LE2) ≥ 0.75 (banded).
- pred_c_writers_five_or_more_blocks_back_carry_a_quarter: cost_share(D_GT4) ≥ 0.25. Null: ≤ 0.10.
- pred_d_cost_share_tracks_energy_share: |cost_share(D_LE2) − energy_share(D_LE2)| ≤ 0.20 (the read is not selective by recency beyond
  what the amplitudes say; §2788 quadratic gain law). Null: gap ≥ 0.40 (recency-selective read).
- pred_e_windows_complementary: 0.7 ≤ (CE(D_LE2)+CE(D_GT2))/CE(DROP_ALL) ≤ 1.5. Null: ≥ 2.0 (strong cross-window interaction).

BARS = {"ce_tol": 1e-4, "repro_tol": 0.015, "b_frac": 0.50, "c_frac": 0.25, "d_gap": 0.20, "e_lo": 0.7, "e_hi": 1.5}
NULLS = {"b_frac": 0.75, "c_frac": 0.10, "d_gap": 0.40, "e_hi": 2.0}

## Price
GPU: 1 fit pass (96 docs) + 1 baseline + 9 arms × 64 docs ≈ 830 doc-forwards; ~60 s on the 5090. Nothing installs into the §312 frontier.

## What would change my mind
b false with null met → banded, short-range wires: the program's tail channel is a block-to-block pipe, not a bus. d null met → the read
selects by writer recency, which the linear-read picture (§2780–§2788) does not predict.
