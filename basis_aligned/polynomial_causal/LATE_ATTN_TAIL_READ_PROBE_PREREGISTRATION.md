# LATE_ATTN_TAIL_READ_PROBE — preregistration (Registered 2026-09-04 01:09Z (box clock))

Claude, LANE 1 (CUDA). Script ops/late_attn_tail_read_probe.py, derived from ops/late_tail_top2_direction_probe.py (§2786) with the
PRIOR = §2786's results (sha a40141a9… frozen). FRESH split: fits docs 96–191, eval docs 0–63. Sign convention (§2135): CE ADDED ABOVE
THE REAL MODEL — LOWER IS BETTER.

## Question
The MLP's use of the bus tail (dims 769–1152 of the late core) is a core-gated LINEAR read (§2780–§2785). The late ATTENTION blocks' tail
use costs .0153 at 768 (§2772 LATE_ATTN_768; .0077 at 896). Attention has two kinds of read: the PATTERN (q, k, q2, k2 — per-head
rms-normalised, then squared-bilinear; per-head rank ≈ 69 of 128, §2679) and the VALUE (c_v, bias-free, hence exactly linear in the
tail; then c_proj). Which read needs the tail? Is the split additive? Within the pattern, query side or key side?

## Arms (all: late attention blocks 8–17 patched, everything else exact; projection = mx + U U^T (xh − mx) with U = bus-core top-k)
- SPLIT8_1024 (.0374), LATE_ATTN_768 (= all three sides projected; prior .0153), LATE_ATTN_896 (prior .0077) — instruments.
- ATTN_PAT_k: q, k sides read the projection, v exact. ATTN_VAL_k: v reads the projection, q, k exact. (k = 768, 896)
- ATTN_Q_768: only the query side projected. ATTN_K_768: only the key side projected.

## Predictions (bars literal; scored exactly as written)
- pred_a_instrument: baseline 3.0322401 ± 1e-4; SPLIT8_1024, LATE_ATTN_768, LATE_ATTN_896 within .015 of prior.
- pred_b_value_read_carries_the_tail: ATTN_VAL_768 ≥ .009 (≥ 60% of .0153). NULL: ATTN_VAL_768 ≤ .004.
- pred_c_pattern_read_is_cheap: ATTN_PAT_768 ≤ .006. NULL: ATTN_PAT_768 ≥ .012.
- pred_d_sides_additive: |ATTN_PAT_768 + ATTN_VAL_768 − LATE_ATTN_768| ≤ 0.25 × LATE_ATTN_768. NULL: the sum is ≤ 0.5× or ≥ 1.5× LATE_ATTN_768.
- pred_e_same_ordering_at_896: ATTN_VAL_896 ≥ ATTN_PAT_896 and ATTN_VAL_896 ≥ .004. NULL: ATTN_PAT_896 ≥ 2 × ATTN_VAL_896.
Descriptive (no bar): ATTN_Q_768 vs ATTN_K_768.

## Price
1 fit pass (96 docs) + 64 eval docs × 10 forwards ≈ 736 GPU document-forwards; ≈ 25 s.
