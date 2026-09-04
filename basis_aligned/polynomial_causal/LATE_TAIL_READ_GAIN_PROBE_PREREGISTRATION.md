# LATE_TAIL_READ_GAIN_PROBE — preregistration (Registered 2026-09-04 01:12Z (box clock))

Claude, LANE 1 (CUDA). Script ops/late_tail_read_gain_probe.py, derived from ops/late_tail_product_term_probe.py (§2780) with the
PRIOR = §2786's results (sha a40141a9… frozen). FRESH split: fits docs 96–191, eval docs 0–63. Sign convention (§2135): CE ADDED ABOVE
THE REAL MODEL — LOWER IS BETTER.

## Question
§2780–§2785: each late MLP's use of the bus tail t is the core-gated LINEAR read Down(Lc∘Rt + Lt∘Rc) (cross term; .0087 residual when
the tail's self term is dropped). A manipulation test of "calibrated linear read": put a GAIN α on the cross term (tail-linear program,
self term dropped) and on the whole tail input (exact model with t → αt). If the read is a calibrated linear correction, CE(α) is
quadratic around α = 1: half the read recovers ~3/4 of the cost, and doubling it costs about as much as removing it. If the read were a
saturating / on-off signal, over-gain would be cheap.

## Arms (late MLPs 8–17 at k = 768, everything else exact)
- TL_A{0,05,1,15,2}: prod = Lc∘Rc + α (Lc∘Rt + Lt∘Rc), α ∈ {0, .5, 1, 1.5, 2}. TL_A0 = LATE_MLP_768 (.1249); TL_A1 = DROP_TT_768 (.0087).
- SC_A{05,15,2}: the exact MLP on the input c + α t (cross scales by α, self term by α²). SC at α = 1 is the real model (0).
- SPLIT8_1024 (.0374) instrument.

## Predictions (bars literal; scored exactly as written)
- pred_a_instrument: baseline 3.0322401 ± 1e-4; SPLIT8_1024, TL_A0 (vs .1249), TL_A1 (vs .0087) within .015 of prior.
- pred_b_half_gain_recovers_three_quarters: TL_A05 ≤ 0.35 × TL_A0. NULL: TL_A05 ≥ 0.60 × TL_A0.
- pred_c_double_gain_costs_like_removal: 0.6 × TL_A0 ≤ TL_A2 ≤ 1.6 × TL_A0. NULL: TL_A2 ≤ 0.30 × TL_A0 (over-gain is cheap) or TL_A2 ≥ 3 × TL_A0.
- pred_d_unit_gain_is_the_minimum: TL_A1 ≤ min(TL_A05, TL_A15) − 0.000 (i.e. TL_A1 is the smallest of the TL grid). NULL: TL_A15 ≤ TL_A1 − .003 (over-reading the tail HELPS).
- pred_e_self_term_does_not_rescue_over_gain: SC_A2 ≥ TL_A2. NULL: SC_A2 ≤ 0.7 × TL_A2.
Descriptive (no bar): SC_A05 vs TL_A05; TL_A15 / TL_A05 (quadratic → 1).

## Price
1 fit pass (96 docs) + 64 eval docs × 10 forwards ≈ 736 GPU document-forwards; ≈ 25 s.
