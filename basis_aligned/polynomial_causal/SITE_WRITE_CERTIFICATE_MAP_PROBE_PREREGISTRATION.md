# Second-order Fisher certificate for ALL 36 write sites at k = 32, and the joint MLP16+MLP17 certificate (Claude, CPU)

Registered 2026-09-03 18:46 UTC (box `date -u`), BEFORE running. Script: `ops/site_write_certificate_map_probe.py`.
Results: `site_write_certificate_map_probe_results.json`. Price: CPU only, 0 GPU forwards; ~1,900 CPU document
forward-equivalents (96 fit-doc forwards for the bases; docs 96-159: forward + 1 true-token + 4 sampled-token backwards with all
36 writes as leaves; docs 96-191: the same with the two final MLP writes as leaves) — est. 10-14 min alone at 16 threads.
Source: §2699 (the certificate holds for MLP17, ratios .79-.89) + §2696 (the measured 36-site k = 32 price map) — Move 3 of
MATHEMATICAL_REVIEW_2026-09-03_1630.md (price the whole map analytically; price JOINT installations from one score pass).
SIGN CONVENTION (§2135): every CE number is CE ADDED ABOVE THE REAL MODEL on held-out natural docs — LOWER = better.
Stage A docs 96-159 (baseline 3.11250, §2696); Stage B docs 96-191 (baseline 3.08238, §2694). No installation into §312.

## Objects and formulas (arms named)
- Bases: `site_write_pca_truncation_ce_map_probe.fit_bases` on natural docs 0-95, all positions — identical to §2696 (mu_s, U_s
  for every site s in {attn_l, mlp_l}, l = 0..17). Rank-k residual at position t: delta_{s,k,t} = (I - U_k U_k^T)(w_{s,t} - mu_s).
- Scores: g_{s,t} = d[-log p_t(y_t)]/d w_{s,t} (true token) and s_{s,t}^(i) = d[log p_t(y~_i)]/d w_{s,t}, y~_i ~ p_t, i = 1..4,
  RNG torch.Generator seed 0; ALL sites are leaves of one autograd graph per batch (CH = 8 docs), so one backward per
  (true / sample) yields the scores at all 36 sites at once. Model weights frozen; exact autograd through the manual forward.
- SINGLE-SITE CERTIFICATE (§2699's formula, unchanged): cert_{s,k} = mean_t [ g_{s,t}.delta_{s,k,t} + 1/2 mean_i (s_{s,t}^(i).delta_{s,k,t})^2 ].
- JOINT CERTIFICATE for a set A of sites: cert_{A,k} = mean_t [ sum_{s in A} g_{s,t}.delta_{s,k,t} + 1/2 mean_i (sum_{s in A} s_{s,t}^(i).delta_{s,k,t})^2 ];
  cross term X_{A,k} = cert_{A,k} - sum_{s in A} cert_{s,k} (second-order interaction between the sites' tails).
- ratio_{s,k} = measured_{s,k} / cert_{s,k}, measured from the frozen prior jsons (§2696 `sites[].ce_added_k32` for Stage A;
  §2694 `ce_added_ladder` incl. `both` for Stage B). LATE13 := the 13 sites in blocks 7-17 whose §2696 k = 32 price is >= .02:
  attn7 mlp7 attn8 mlp8 mlp9 mlp10 mlp11 mlp12 mlp13 mlp14 mlp15 mlp16 mlp17 (read from the frozen json at run time, not typed).

## Preregistered predictions (scored exactly as written)
- pred_a_instrument: (i) unpatched manual-forward CE on docs 96-159 reproduces §2696's frozen `baseline_ce_eval` (3.1124951)
  within 1e-4; (ii) Stage B's single-site cert_{mlp17,32} on docs 96-191 reproduces §2699's frozen `certificate_pred["17"]["32"]`
  (.070165) within .01 (same estimator, same seed, different sampling order — MC-sample sensitivity, §2135 tolerance note).
- pred_b_late_sites_certified: for ALL 13 LATE13 sites, ratio_{s,32} in [.5, 2]. Null: >= 5 of the 13 outside [.25, 4].
- pred_c_early_breakdown: mlp1 (measured .8834, the largest tail) has ratio_{mlp1,32} OUTSIDE [.5, 2] — the second-order
  expansion breaks where the tail is large. Null: all four of mlp0, mlp1, mlp2, mlp3 have ratios inside [.5, 2] (the certificate
  is valid even at .9 nat — a stronger tool than expected).
- pred_d_joint_mlp16_17_k8: on docs 96-191, ratio measured `both.8` (.17248) / cert_{{mlp16,mlp17},8} in [.5, 2] AND the cross term
  X_{{16,17},8} >= +.02 (measured superadditivity is .17248 - .03554 - .08326 = +.054). Null: X <= 0 (the certificate is blind
  to the measured interaction).
- pred_e_ordering: Spearman(cert_{s,32}, measured_{s,32}) over all 36 sites >= .8. Null: <= .4.
Disclosed, not scored: the full 36-row table (measured, cert, first-order share, ratio); ratios for all early sites; joint k = 32
for {16,17} vs measured `both.32` (.12333) and its cross term; cert_{mlp16,32}, cert_{mlp16,8}, cert_{mlp17,8} on docs 96-191 vs
§2699's frozen values; the sum over all 36 single-site k = 32 certificates (an analytic "everything truncated at once, no
interactions" price) — disclosed for a later joint-installation registration, not scored here.

## Null model / what a failure means
pred_b false: the certificate does not transfer beyond MLP16/17 — pricing mid/late sites still needs forwards. pred_c null:
second order is valid even for the early dense writes (then the whole 36-site map is analytically recoverable). pred_d null: the
observed MLP16×MLP17 interaction is not a second-order tail interaction (higher-order or via attention pattern changes) — joint
installation cannot be priced from single-site scores. pred_e null: the certificate does not even order the sites.
Frozen: checkpoint 680d6c26…, fit_natural.pt 666a3201…, §2696 results 48bd52ec…, §2694 results 8a88b714…, §2699 results 1ef01351…
(all hashes frozen in the script). No GPU. Expected: b TRUE, c TRUE, d TRUE, e TRUE (priors from §2699's .79-.89 ratios).
