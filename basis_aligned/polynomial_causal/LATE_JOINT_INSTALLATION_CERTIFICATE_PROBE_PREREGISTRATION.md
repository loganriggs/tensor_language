# Preregistration — late_joint_installation_certificate_probe (Claude lane, CPU-only, LANE 2)

Registered 2026-09-03 19:10Z (box clock), BEFORE the script was written. Frozen (sha256 in the script) together with
`site_write_certificate_map_probe_results.json` (§2701).

Source: §2701 — the second-order Fisher certificate is valid for blocks 11–17 (single-site ratio measured/cert .42–1.59) and
priced the MLP16×MLP17 cross term to .0004 (certified +.0533 vs measured +.0537). Successor claims: (1) the block-≥11 validity on
a FRESH document split; (2) the certificate prices NESTED JOINT installations of many late sites — the first analytic price of
"truncate every late write at once", the compression step the program actually needs.

SIGN CONVENTION (§2135): every "measured" number is CE ADDED ABOVE THE REAL MODEL on held-out documents — LOWER IS BETTER. A
"certificate" is the analytic second-order prediction of the same CE-added; ratio measured/cert = 1 is a perfect price. Nothing
installs into the §312 frontier; the bases are activation covariances of the writes (descriptive), not metric-constructed.

## Split (fresh)
Bases fitted on docs 96–191 (96 docs); everything scored on docs 0–63 (64 docs, 16,384 positions). Docs 0–63 have never been a
scoring set in this arc (all prior maps fitted on 0–95 and scored on 96–159/191), so every measured number here is new, and the
swap also tests that the k=32 bases are not an artifact of one fit split. Same estimator as §2701: true-token gradient plus
S = 4 sampled-score gradients, torch.Generator seed 0, all sites detached as leaves in one backward per sample.

## Definitions (arms named)
- LATE14 := the 14 write sites of blocks 11–17: attn_l, mlp_l for l = 11…17.
- single_{s} := measured CE-added when only site s is truncated to its top-32 write-PCA directions (write' = mu + U_32 U_32ᵀ(write − mu)).
- cert_{s} := the second-order certificate of single_{s}; ratio_s = single_s / cert_s.
- Nested joint sets, all at k = 32: A1 = {mlp16, mlp17}; A2 = {mlp14, mlp15, mlp16, mlp17}; A3 = {mlp11 … mlp17} (7 sites);
  A4 = LATE14 (all 14). joint_meas(A) := measured CE-added with every site in A truncated simultaneously (one forward per
  document); joint_cert(A) := certificate with the δ's of all sites in A summed inside the square; ratio_A = joint_meas/joint_cert.
- Cross terms: X_meas(A) = joint_meas(A) − Σ_{s∈A} single_s; X_cert(A) = joint_cert(A) − Σ_{s∈A} cert_s.
- Reference (old split, §2696 docs 96–159): Σ single over LATE14 = .2905; over A3 = .2247; over A2 = .1341; over A1 = .0848.

## Preregistered predictions (scored exactly as written)
- pred_a_instrument: (i) unpatched manual-forward CE on docs 0–63 equals the module forward within 1e-4 (first 4 docs); (ii) an
  identity patch (k = 1152) at mlp17 changes CE by ≤ 1e-4 on the first 4 docs; (iii) the fresh-split write-covariance
  effective rank at mlp17 is within ±3 of §2696's frozen `eff_rank_fit` (6.139) and its rank_90 within ±2 of the frozen 4 (this
  clause only guards a broken fit). [(iii) reworded 19:10Z, before any script existed: the first draft referenced a "k=32 tail
  energy fraction" that §2696's json does not contain; eff_rank/rank_90 are the frozen numbers actually on disk.]
- pred_b_late14_single_certified_fresh: ratio_s ∈ [.5, 2] for at least 13 of the 14 LATE14 sites. Null: ≤ 9 of 14.
- pred_c_nested_joint_certified: ratio_A ∈ [.5, 2] for ALL four of A1, A2, A3, A4. Null: ratio_{A4} outside [.25, 4].
- pred_d_cross_terms_certified: for A3 and A4, X_meas > 0 AND X_cert > 0 AND X_meas/X_cert ∈ [.5, 2]. Null: X_cert ≤ 0 for A4,
  or X_meas/X_cert outside [.25, 4] for A4.
- pred_e_superadditive_installation: joint_meas(A4) ≥ 1.5 × Σ_{s∈LATE14} single_s (on this split). Null: ≤ 1.1 × (the late
  writes' tails are approximately independent; installation price is the sum of the map).
Disclosed, not scored: the 14-row single table (measured, cert, first-order share, ratio) on the fresh split against §2701's
old-split values; joint_meas / joint_cert / X for all four sets; the pairwise structure implied by (A1 ⊂ A2 ⊂ A3 ⊂ A4);
joint_meas(A4) in absolute terms as the first "all late writes at rank 32" installation price (a candidate for the smaller
program, to be weighed against the §312 frontier's byte budget later, not here).

## Null model / what a failure means
pred_b false: block-≥11 validity was a property of the old split (or the old bases). pred_c null: the joint certificate does not
scale past two sites — many-site installation cannot be priced analytically. pred_d null: the certified cross term was a two-site
accident. pred_e null: late-write tails are independent and the map's sum IS the installation price (then a 14-site
installation costs ~.29 and the certificate is unnecessary for it).

## Price (literal)
~96 fitting forwards + a 64-document score pass (1 forward + 5 backwards per batch) + 64 × (1 + 14 + 4) = 1,216 patched
forwards ≈ 1,700 CPU document-forward equivalents; 0 GPU. Lane 2 (CPU-only, 4 threads, nice 10): ~40 min if lane 1 is
GPU-bound, up to ~2.5 h under CPU contention. Bars are frozen here; the script reads this file's sha256.
