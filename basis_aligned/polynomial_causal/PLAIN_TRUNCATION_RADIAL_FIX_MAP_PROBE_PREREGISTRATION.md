# Plain truncation + radial-fix map — preregistration

Registered 2026-09-03 20:14Z (box clock), before the script exists. Lane 1 (CUDA). SIGN CONVENTION (§2135): every number is CE
ADDED ABOVE THE REAL MODEL on held-out docs 96–159 — LOWER IS BETTER. Descriptive; nothing installs into the §312 frontier; bases
are data covariances of the writes scored by CE only (§2118 stays closed).

## Motivation
§2705: the radial component r = w·x̂ of every write is replaceable by a per-site constant r̄ (≤ .036 everywhere). §2706: fitting
the tangential remainder in the PRE-write frame is the wrong frame at low-rank sites (w_perp is higher-rank than w: attn1 111 vs
22) and the right one at the fat early MLPs (mlp1 .572 vs .883 at k=32). Rule extracted in §2706: truncate the write in its OWN
frame first, then set the radial scalar of the reconstruction to r̄. This probe tests that rule and, as a by-product, fills in
the plain k=8 / k=128 columns §2696 lacked.

## Design
Forward/split as §2704–§2706. Fit on docs 0–95 in one collecting pass: r̄_s and the plain write covariance (μ_s, U_s). Arms, one
site at a time, eval docs 96–159, k ∈ {8, 32, 128}, all 36 sites:
- **PLAIN_k**: w' = μ + U_k U_kᵀ (w − μ) (the §2696 construction).
- **PLAINFIX_k**: w' = P_k(w) + (r̄ − P_k(w)·x̂) x̂ — the plain reconstruction with its radial scalar reset to the constant.
Frozen comparators: §2696 PLAIN_32 (48bd52ec…), §2706 RM_TAN_k (828d86b3…), §2705 RADIAL_MEAN floor (db47a079…).

## Predictions (bars fixed now)
- **pred_a_instrument**: baseline within 1e-4 of 3.1125031; PLAIN_32 within .01 of §2696's per-site value at ≥ 34 of 36 sites
  (bases are re-fitted here in float64 on the GPU; near-degenerate eigenpairs may differ); monotone in k at ≥ 34 of 36 for both arms.
- **pred_b_fix_helps_where_the_frame_failed**: PLAINFIX_8 < PLAIN_8 at all five sites where §2706 was worse than plain by > .02
  (attn1, attn5, attn6, mlp16, mlp17). Null: ≤ 2 of 5.
- **pred_c_fix_never_hurts**: PLAINFIX_32 ≤ PLAIN_32 + .005 at ≥ 33 of 36 sites. Null: ≤ 25 of 36.
- **pred_d_attn1_compact_corrected**: attn1 PLAINFIX_8 ≤ .03 (plain k=8 .066; radial floor .010). Null: ≥ .06.
- **pred_e_best_of_three_total**: Σ_36 min(PLAIN_32, PLAINFIX_32, RM_TAN_32[§2706]) ≤ 1.60 nat (best-of-two was 1.6715; plain
  2.3712). Null: ≥ 1.65.

## Price
96 + 64·(1 + 6·36) = 13,984 GPU document-forwards ≈ 2.5 min on lane 1. Output plain_truncation_radial_fix_map_probe_results.json.
Frozen: this file, the three priors above, checkpoint, fit_natural.pt.
