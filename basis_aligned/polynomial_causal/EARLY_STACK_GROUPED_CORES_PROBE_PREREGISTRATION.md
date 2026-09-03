# Preregistration — early_stack_grouped_cores_probe (Claude, lane 1 CUDA)

Registered 2026-09-03 22:54Z (box clock). Follows §2749 (the early 22 sublayers on own 768-cores .057; on ONE shared core .180; the late 14 share one core at ≤ .006).

## Question
The early stack does not share one input coordinate system. Does it share a FEW — by depth (consecutive blocks) or by kind
(attention vs MLP)? And how fast does the read subspace drift: how well does block l's 768-core capture block l±1's input?

## Arms (eval docs 0–63; fits 96–191; MLPs OwnHead CONST, attention AttnHead; k = 768 everywhere; group core = top-768 of the plain
average of the member sites' centred input covariances)
EARLY22_OWN_768, EARLY22_SHARED_768 (reproductions of §2749).
G2_768: two depth groups, blocks 0–5 and 6–10 (both sublayer kinds in each group).
G3_768: three depth groups, 0–3 / 4–7 / 8–10.
G4_768: four depth groups, 0–2 / 3–5 / 6–8 / 9–10.
KIND2_768: two cores by kind — one for the 11 early attention blocks, one for the 11 early MLPs.
Descriptive: the 36 × 36 capture matrix cap(i, j) = tr(U_jᵀ C_i U_j) / tr C_i (site i's input covariance under site j's 768-core);
the adjacent-block same-kind ratio r_i = cap(i, i±1) / cap(i, i), min over the 22 early sites.

## Predictions (CE added above the real model, docs 0–63, LOWER IS BETTER — §2135)
- pred_a_instrument: baseline within 1e-4 of 3.0322401; EARLY22_OWN_768 within .02 of .057; EARLY22_SHARED_768 within .02 of .180.
- pred_b_three_depth_groups_nearly_free: G3_768 − EARLY22_OWN_768 ≤ .04. Null: ≥ .10.
- pred_c_two_depth_groups: G2_768 − EARLY22_OWN_768 ≤ .06. Null: ≥ .12.
- pred_d_depth_not_kind: KIND2_768 − G2_768 ≥ .02 (sharing by kind is worse than sharing by depth with the same number of cores).
  Null: KIND2_768 ≤ G2_768.
- pred_e_adjacent_blocks_read_alike: min over the 22 early sites of the adjacent same-kind capture ratio r_i ≥ .95. Null: < .85.
Descriptive: G4; the full capture matrix; the ratio profile by depth.

## Price
96 fit docs + 64 × (1 + 6 arms) = 544 GPU document-forwards, ~15 s. Frozen: this file, §2749 results, checkpoint, fit_natural.pt.
