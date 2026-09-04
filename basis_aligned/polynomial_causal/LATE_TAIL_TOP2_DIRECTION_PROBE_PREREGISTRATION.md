# LATE_TAIL_TOP2_DIRECTION_PROBE — preregistration (Registered 2026-09-04 01:03Z (box clock))

Claude, LANE 1 (CUDA). Script ops/late_tail_top2_direction_probe.py, derived from ops/late_tail_channel_rank_probe.py (§2779) with the
PRIOR = §2785's results (sha 4cd9256f… frozen). FRESH split: fits docs 96–191, eval docs 0–63. Sign convention (§2135): CE ADDED ABOVE
THE REAL MODEL — LOWER IS BETTER.

## Question
§2779 found the pooled late-origin tail covariance (λ-propagated writes of blocks 8..l, scaled, off the 768 core) has two large eigenvalues
(1.46, .67) above a flat floor (~.065) and is otherwise isotropic. What are those two directions — a position-0 (sink-like) spike, a
readout-aligned direction, a shared cross-block direction — and do they matter for CE?

## Measurements
- u1, u2 = top-2 eigenvectors of the pooled covariance (fit docs). Per-position energy of the tail on u1, u2 (share at position 0; first 8).
- Subspace cosines of u1, u2 with the unembed row space (top-16 / top-128 right singular vectors of lm_head), |cos| with the mean wte.
- Per late block l: top eigenvector of that block's own late-origin covariance; |cos| with u1; count of blocks with |cos| ≥ 0.9.
- Arms: SPLIT8_1024 (.0374), BUS_896 (.0662) instruments; DROP_TOP2_MLP = all late MLPs read the exact input with u1, u2 replaced by their
  fit-set mean; DROP_RAND2_MLP = same for 2 seeded random unit vectors in the complement's directions 3–386.

## Predictions (bars literal; scored exactly as written)
- pred_a_instrument: baseline 3.0322401 ± 1e-4; SPLIT8_1024, BUS_896 within .015 of prior; top eigenvalue within 0.05 of 1.46.
- pred_b_top_direction_is_a_position0_spike: u1's position-0 energy share ≥ 0.50. NULL: ≤ 0.10.
- pred_c_top_directions_are_readout_aligned: max over u1, u2 of the cosine with the unembed top-128 subspace ≥ 0.60. NULL: ≤ 0.30.
- pred_d_dropping_top2_is_cheap: DROP_TOP2_MLP ≤ .003. NULL: ≥ .010.
- pred_e_top_direction_shared_across_late_blocks: ≥ 8 of 10 late blocks have |cos(own top-1, u1)| ≥ 0.9. NULL: ≤ 3 blocks.

## Price
3 fit passes (96 docs each) + 64 eval docs × 5 forwards ≈ 608 GPU document-forwards; ≈ 25 s.
