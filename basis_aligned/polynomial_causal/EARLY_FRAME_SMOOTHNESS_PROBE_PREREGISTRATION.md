# Preregistration — early_frame_smoothness_probe (Claude, lane 1 CUDA)

Registered 2026-09-03 23:03Z (box clock). Follows §2750 (the early read frame rotates continuously; adjacent-block capture ratio .84 → .98; the late core captures blocks
8–17 at ≥ .90) and §2751 (whole-model width program .197 at 768 with 22 own early frames + 1 late frame).

## Question
Is the rotating early frame SMOOTH — predictable from its neighbours — so that the 22 early frames could be described by fewer than
22 covariances? And does the "settled" region start at block 8: can blocks 8–17 share one frame?

## Arms (eval docs 0–63; fits 96–191; k = 768; MLPs OwnHead CONST, attention AttnHead; a core is the top-768 of the plain average of
the named sites' centred input covariances)
EARLY22_OWN_768, ALL36_768 (reproductions of §2749/§2751).
EARLY22_WIN3_768: site (kind, l) reads the core of blocks l−1, l, l+1 (both kinds; clipped at the ends).
EARLY22_LOO_768: as WIN3 but with the site's OWN covariance left out (its same-block partner and the neighbour blocks remain).
EARLY22_NBR_768: neighbour blocks l−1 and l+1 only (own block excluded entirely; block 0 uses block 1, block 10 uses blocks 9 and 11).
ALL36_SPLIT8_768: own cores for blocks 0–7 (16 sites) + ONE shared core for blocks 8–17 (20 sites).

## Predictions (CE added above the real model, docs 0–63, LOWER IS BETTER — §2135)
- pred_a_instrument: baseline within 1e-4 of 3.0322401; EARLY22_OWN_768 within .02 of .057; ALL36_768 within .02 of .197.
- pred_b_sliding_window_is_free: EARLY22_WIN3_768 − EARLY22_OWN_768 ≤ .02. Null: ≥ .06.
- pred_c_frame_predictable_from_neighbours: EARLY22_NBR_768 − EARLY22_OWN_768 ≤ .04. Null: ≥ .10.
- pred_d_leave_one_out: EARLY22_LOO_768 − EARLY22_OWN_768 ≤ .03. Null: ≥ .08.
- pred_e_settled_from_block_8: ALL36_SPLIT8_768 − ALL36_768 ≤ .03. Null: ≥ .10.
Descriptive: per-arm differences by depth are not measured (stack arms only); this rung is the smoothness test, not a per-site map.

## Price
96 fit docs + 64 × (1 + 6 arms) = 544 GPU document-forwards, ~18 s. Frozen: this file, §2751 results, checkpoint, fit_natural.pt.
