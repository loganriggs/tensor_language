# MLP16/17 rank-8 write as 8 quadratic forms — exact rank and in-situ interaction-rank price (Claude, CPU)

Registered 2026-09-03 16:06 UTC (system clock), BEFORE running. Script: `ops/mlp_final_blocks_quadratic_form_rank_probe.py`.
Results: `mlp_final_blocks_quadratic_form_rank_probe_results.json`. Price: CPU only, 0 GPU forwards, ~960 CPU document
forwards (~10-12 min at 16 threads; ~2x if contended). Follows §2694 (rank-8 output truncation: MLP16 adds .036, MLP17 .083).

## Question (Logan's directive: "specific interaction terms that are important … smaller than an MLP block")
With the write of block l restricted to its fitted rank-8 basis U (§2694, fit docs 0-95), the surrogate is
write' = mu + U (c(x) - U^T mu) with c_j(x) = U_j^T Down (Left xhat * Right xhat) = xhat^T Q_j xhat, an EXACT quadratic form
Q_j = Left^T diag(M_j) Right, M = U^T Down (8 x 4608), symmetrised Qs_j = (Q_j + Q_j^T)/2 (Left/Right/Down have no bias;
Down_bias is added outside the write). So the rank-8 surrogate IS 8 quadratic forms in the rms-normed block input. This probe
asks (i) the exact weight rank of those 8 forms and (ii) how many eigen-directions of each form the model actually USES in situ:
truncate each Qs_j to its top-r eigenpairs by |lambda| and score CE ADDED (§2135: above the real model, LOWER = better) on
held-out natural docs 96-191, all 256 positions. Blocks 16 and 17, k = 8 fixed, r in {16, 64, 256, 1152 (exact)}.

## Preregistered predictions (scored exactly as written; SIGN: CE ADDED, LOWER = better)
- pred_a_instrument: (i) the quadratic-form forward with no patch reproduces the §2694 module CE within 1e-4 on 4 EVAL docs;
  (ii) the EXACT forms (r = 1152) at k = 8 reproduce §2694's frozen k = 8 CE ADDED for BOTH blocks (.03554 MLP16, .08326 MLP17,
  read from the frozen results json) within .002 each.
- pred_b_forms_high_rank: mean over the 8 forms per block of the lambda^2-energy effective rank (exp of the entropy of
  lambda_i^2 / sum lambda^2) >= 200 in BOTH blocks — the exact interaction operators are high-rank even where the output
  collapsed (§2673-§2676 logic). Null: <= 64 in EITHER block.
- pred_c_mlp16_r64_cheap: MLP16 (k = 8, r = 64) adds <= .06 (i.e. <= .025 above its exact rank-8 cost .036). Null: >= .15.
- pred_d_mlp16_r256_near_exact: MLP16 (k = 8, r = 256) adds <= .045. Null: >= .10.
- pred_e_mlp17_r64: MLP17 (k = 8, r = 64) adds <= .12 (i.e. <= .04 above its exact rank-8 cost .083). Null: >= .25.
Disclosed, not scored: r = 16 rows for both blocks; per-form spectra (top-|lambda| share, positive-mass fraction
sum(lambda+)/sum|lambda|, energy eff rank); the r = 64 excess over exact for both blocks; the coefficient RMS per form on the
EVAL half (which of the 8 forms carry the variance).

## Arms / formulas
Bases: identical fit to §2694 (`fit_bases` on natural docs 0-95, all positions; U = top-8 eigenvectors of the centred write
covariance, descending). Forms: Qs_j as above from the float32 weights; eigh; top-r by |lambda|. Patch: mw' = mu + U (c_r(x) - U^T mu),
c_r,j(x) = sum_{i<=r} lambda_ji (v_ji . xhat)^2, xhat = rms_norm(x) exactly as in the manual forward. CE over docs 96-191,
inputs [:, :256], targets [:, 1:257]. Baseline = unpatched manual forward on the same docs.

## Null model / what a failure means
If pred_b fails with the null (forms low-rank) the 8 forms are themselves a small tensor program (8 x <=64 rank-1 interaction
terms). If pred_c/d fail (in-situ price high at r = 64/256) the block's causally relevant interactions are spread over hundreds
of input directions even for a rank-8 output — "specific interaction terms" do not exist at this grain for MLP16.
Frozen inputs: checkpoint 680d6c26…, fit_natural.pt 666a3201…, §2694 results json (hash frozen in the script). No GPU.
