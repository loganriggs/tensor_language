# Preregistration — pairwise_fisher_subset_price_probe (Claude lane, CPU-only, LANE 2)

Registered 2026-09-03 19:40Z (box clock), BEFORE the script was written. Frozen (sha256 in the script) together with
`late_joint_installation_certificate_probe_results.json` (§2703).

Source: §2703 (the second-order Fisher certificate prices late singles, nested joints and their cross terms on a fresh split) +
MATHEMATICAL_REVIEW_2026-09-03_1930.md Analysis 2 / Move A: the joint certificate is EXACTLY PAIRWISE —
cert(A) = Σ_{s∈A} c_s + Σ_{s<t∈A} X_st with X_st = mean_t E_i (s_t^(i)·δ_s)(s_t^(i)·δ_t) — so 14 + 91 numbers from ONE score pass
predict the price of every one of the 2^14 late installations. This probe tests that subset-price model on random subsets.

SIGN CONVENTION (§2135): every "measured" number is CE ADDED ABOVE THE REAL MODEL on held-out documents — LOWER IS BETTER; J(A)
is the analytic second-order prediction of the same quantity; ratio = measured/J. Nothing installs into the §312 frontier.

## Split (fresh again)
Bases fitted on docs 96–191 (same fit as §2703). Scored on docs 64–95 (32 docs, 8,192 positions) — never used as a fit or
scoring set anywhere in this arc. Same estimator (true-token gradient + S = 4 sampled-score gradients, torch.Generator seed 0).

## Definitions (arms named)
- LATE14 and single truncation at k = 32 as in §2703. c_s := single-site certificate; single_s := measured single price.
- X_st := pairwise certified cross term for the 91 unordered pairs of LATE14 (from the same score pass, each pair as a joint set).
- J(A) := Σ_{s∈A} c_s + Σ_{s<t∈A} X_st (the pairwise model). By algebra J(A) equals the direct joint certificate of A.
- RANDOM12 := 12 subsets of LATE14 drawn by torch.Generator seed 1: four each of sizes 3, 5, 8 (drawn without replacement,
  listed in the results json). NESTED4 := A1 ⊂ A2 ⊂ A3 ⊂ A4 of §2703. Every set's joint price is MEASURED by one patched
  forward per document (all its sites truncated at k = 32 simultaneously).

## Preregistered predictions (scored exactly as written)
- pred_a_instrument: (i) manual-forward CE on the first 4 docs equals the module forward within 1e-4; (ii) for every one of the 16
  sets, the direct joint certificate and J(A) agree within 1e-6 (relative) — the pairwise identity holds in code; (iii) c_s on
  docs 64–95 is within a factor [.5, 2] of §2703's c_s (docs 0–63) for ≥ 12 of the 14 sites (document-set stability of the
  certificate itself).
- pred_b_pairwise_model_prices_random_subsets: ratio measured/J(A) ∈ [.7, 1.4] for ≥ 10 of the 12 RANDOM12 sets. Null: ≤ 6 of 12.
- pred_c_cross_terms_positive: X_st > 0 for ≥ 80 of the 91 pairs. Null: ≥ 30 pairs with X_st < 0 (tails often anti-aligned in
  the Fisher metric — truncating two sites together would then be CHEAPER than separately).
- pred_d_ordering: Spearman(J(A), measured) over the 16 sets (RANDOM12 ∪ NESTED4) ≥ .9. Null: ≤ .5.
- pred_e_design_gain: let BEST7 := the size-7 subset of LATE14 that contains at least four MLP sites and minimises J(A) over all
  such subsets (enumerated; C(14,7) = 3,432 candidates), and WORST7 := the analogous maximiser. Measured price of BEST7 ≤ .5 ×
  measured price of WORST7. Null: ≥ .8 × (the model cannot separate cheap from expensive designs of equal size).
  (BEST7/WORST7 are chosen from the certificate before any measurement and then measured — 2 extra forwards per document.)
Disclosed, not scored: the full 14 × 14 X matrix; MLP–MLP vs attn–attn vs mixed mean X; the 16-set table (J, direct joint
cert, measured, ratio); BEST7/WORST7 membership and prices; the absolute price of A4 on this split vs .9017 on docs 0–63.

## Null model / what a failure means
pred_b null: pairwise second order does not price generic subsets (third-order content is generic, and §2703's nested sets were
favourable). pred_c null: interactions have both signs and cheap joint installations exist that no site map would find — a
different but also useful outcome. pred_e null: the price differences among equal-size designs are inside the certificate's
error, so the certificate cannot design, only verify.

## Price (literal)
96 fitting forwards + a 32-document score pass (1 forward + 5 backwards per batch; 107 joint sets per batch are cheap sums)
+ 32 × (1 baseline + 14 singles + 12 random + 4 nested + 2 designs) = 1,056 patched forwards ≈ 1,350 CPU document-forward
equivalents; 0 GPU. Lane 2: ~12–25 min. Bars frozen here; the script reads this file's sha256.
