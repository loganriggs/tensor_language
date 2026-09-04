# EARLY_ATTN_TAIL_READ_PROBE — preregistration (Registered 2026-09-04 01:16Z (box clock))

Claude, LANE 1 (CUDA). Script ops/early_attn_tail_read_probe.py, derived from ops/late_attn_tail_read_probe.py (§2787) with the
PRIOR = §2788's results (sha 40e9a5a9… frozen). FRESH split: fits docs 96–191, eval docs 0–63. Sign convention (§2135): CE ADDED ABOVE
THE REAL MODEL — LOWER IS BETTER.

## Question
§2787: late attention's tail use is the PATTERN's (72%), query side > key side (1.66×), additive. Compositionality check at early depth
(blocks 0–7, each attention block reading its OWN fitted frame, §2783's setting for the MLPs): same read-role structure? And a first
measurement of the early attention own-frame cost itself (no prior exists; the early MLP core-only cost at 512 is .098, §2783).

## Arms (blocks 0–7 attention patched, own frame top-k, everything else exact)
- SPLIT8_1024 (.0374), LATE_ATTN_768 (.0153) — instruments.
- EARLY_ATTN_k (q, k, v all read the own-frame projection), EARLY_ATTN_PAT_k (q, k only), EARLY_ATTN_VAL_k (v only), k = 512, 768.
- EARLY_ATTN_Q_512, EARLY_ATTN_K_512.

## Predictions (bars literal; scored exactly as written)
- pred_a_instrument: baseline 3.0322401 ± 1e-4; SPLIT8_1024, LATE_ATTN_768 within .015 of prior.
- pred_b_pattern_dominates_early_too: EARLY_ATTN_PAT_512 ≥ EARLY_ATTN_VAL_512. NULL: EARLY_ATTN_VAL_512 ≥ 1.5 × EARLY_ATTN_PAT_512.
- pred_c_query_side_dominates_early_too: EARLY_ATTN_Q_512 ≥ 1.3 × EARLY_ATTN_K_512. NULL: EARLY_ATTN_K_512 ≥ EARLY_ATTN_Q_512.
- pred_d_sides_additive: |PAT_512 + VAL_512 − EARLY_ATTN_512| ≤ 0.25 × EARLY_ATTN_512. NULL: sum ≤ 0.5× or ≥ 1.5×.
- pred_e_early_attention_is_a_minor_width_consumer: EARLY_ATTN_512 ≤ .040 (vs the early MLP core-only .098 at 512). NULL: ≥ .098.

## Price
1 fit pass (96 docs) + 64 eval docs × 11 forwards ≈ 800 GPU document-forwards; ≈ 40 s.
