# Preregistration — radial_gauge_map_probe (Claude lane, CPU-only, LANE 2)

Registered 2026-09-03 19:28Z (box clock), BEFORE the script was written. Frozen (sha256 in the script) together with
`early_mlp_radial_tangential_probe_results.json` (§2702) and `site_write_pca_truncation_ce_map_probe_results.json` (§2696).

Source: §2702 (the pre-write radial component of the early MLP writes is inert: DROP_RADIAL adds ≤ .025 at mlp0–3) and
MATHEMATICAL_REVIEW_2026-09-03_1930.md Analysis 3 / Move B: the rms_norm scale-gauge is soft, not exact — x_post = (|x| + r)·x̂ +
w_perp changes direction with r whenever w_perp ≠ 0 — so WHERE the quotient can be taken must be mapped, and the final block
should be the counter-example (its write dominates the final residual and is read by one rms_norm then lm_head).

SIGN CONVENTION (§2135): every CE number is CE ADDED ABOVE THE REAL MODEL on held-out docs 96–159 (baseline 3.1124951, the
§2696/§2700/§2702 set) — LOWER IS BETTER. Descriptive map; nothing installs into the §312 frontier; no bases are fitted.

## Definitions (arms named)
At each of the 36 write sites s = (attn_l | mlp_l), l = 0…17, the write w is split per position in the PRE-write frame:
x = the residual the write is added to (after the block's λ-mix for attention; after the attention write for the MLP),
x̂ = x/|x|, r = w·x̂, w_perp = w − r·x̂. Arms: IDENTITY (w' = r·x̂ + w_perp, reassembled — instrument only);
DROP_RADIAL (w' = w_perp); SCALE_RADIAL_2 (w' = 2r·x̂ + w_perp). One site patched at a time; 64 docs per arm.

## Preregistered predictions (scored exactly as written)
- pred_a_instrument: (i) unpatched CE reproduces the frozen 3.1124951 within 1e-4; (ii) IDENTITY at mlp17 and at attn0 changes CE
  by ≤ 1e-4 on the first 4 docs; (iii) DROP_RADIAL at mlp1 reproduces §2702's frozen .0079 within .003.
- pred_b_mid_mlp_radial_inert: DROP_RADIAL ≤ .03 at EVERY MLP site of blocks 2–15 (14 sites). Null: ≥ 3 of the 14 have
  DROP_RADIAL ≥ .10.
- pred_c_final_mlp_radial_functional: DROP_RADIAL at mlp17 ≥ .30. Null: ≤ .05 (the radial part is inert even at the last write,
  which would mean the final rms_norm + lm_head is insensitive to the mix of old stream vs MLP17 content — surprising).
- pred_d_early_radial_soft_both_ways: SCALE_RADIAL_2 ≤ .05 at each of mlp1, mlp2, mlp3. Null: ≥ .20 at any of the three
  (the gauge is one-sided: deletable but not inflatable).
Disclosed, not scored: all 72 arm numbers (both arms × 36 sites), in particular the 18 attention sites, mlp0 and mlp16; the
per-site radial energy fraction of the write on the eval docs; the DROP_RADIAL ranking as a "where the scale-gauge is legal" map.

## Null model / what a failure means
pred_b null: the inertness of §2702 is an early-block property (write ≫ stream), not a network-wide gauge — the quotient can
only be taken at blocks 0–3. pred_c null: the radial part is inert everywhere, including the last write; then the map is
trivial and the §2699 radial share .50 at MLP17 is a gauge coordinate too. pred_d null: the gauge is one-sided; the write's radial
component can be removed but not amplified, i.e. the network sits at a saturation, not on a flat direction.

## Price (literal)
64 baseline + 8 instrument + 36 × 2 × 64 = 4,680 CPU document forwards; 0 GPU. Lane 2 at ~1.0 s per forward with lane 1
GPU-bound: ~80 min (up to ~3 h under CPU contention). Bars frozen here; the script reads this file's sha256.
