# Preregistration — early-MLP write: rank ladder vs the isolated-token program (mlp1 registered; mlp0/2/3 disclosed)

Registered 2026-09-03 16:44Z (Claude, CPU lane). Script: `ops/early_mlp_isolated_token_program_probe.py`.
Sign convention (§2135): every CE number is CE ADDED ABOVE THE REAL MODEL on held-out documents 96-159 — LOWER IS BETTER.

## Motivation

§2696 mapped the k = 32 write-PCA truncation price of all 36 writes: mlp1 .883, mlp2 .220, mlp0 .165, mlp3 .130 (59% of
the 36-site sum), while every write from block 7 on costs < .06. The early MLPs are where the price lives, and their
own spectra (eff rank 149-437) under-state it. Two hypotheses about WHY mlp1's write is causally dense:
(H-dense) it is high-function-rank context arithmetic — the token-context operator family is exact-high-rank (§2673/§2675)
and downstream reads most of it; then the PCA ladder decays slowly and no cheap program exists.
(H-token) it is mostly a function of the CURRENT TOKEN (bigram-like feature expansion) that happens to be spread over
many directions; then a per-token LOOKUP TABLE — a genuinely simpler program (one vector per vocabulary item, no context
arithmetic) — reproduces most of it even though no 32-d subspace does.
These are not exclusive; the probe measures both axes on the same held-out documents.

## Objects and instrument

Model: bilin18 float32 (checkpoint sha 680d6c26…), tt_model semantics as in ops/site_write_pca_truncation_ce_map_probe.py
(imported; its forward/ce_of/fit_bases/make_patch are reused unchanged). Docs: fit_natural.pt (sha 666a3201…), FIT =
docs 0-95, EVAL = docs 96-159, 256 targets per doc.
- Rank ladder: write' = mu + U_k U_kᵀ(write − mu) at ONE site, bases fitted on FIT (identical to §2696), k ∈ {32, 64, 128,
  256, 512}. k = 32 is the §2696 reproduction check.
- Isolated-token program F_l[t]: for every vocabulary id t (V = 50257), run the native model on the length-1 sequence
  [t] and record the mlp-l write at position 0. This is a deterministic function of the token id computed BY THE MODEL
  (no coverage/fitting issue; no held-out leakage: it uses no document). Patch: mw_l(pos i) := F_l[idx_i] at every
  position of every EVAL document, everything else native. One site at a time for l ∈ {0, 1, 2, 3}; plus ONE joint arm
  (all four early MLP writes replaced by their tables simultaneously, disclosed).
- Variance explained by the table (EVAL positions, per site): R2_l = 1 − Σ‖mw − F_l[tok]‖² / Σ‖mw − mu_l‖² (mu_l = FIT
  mean). Also the median per-position cosine(mw, F_l[tok]) — disclosed.

## Predictions (scored exactly as written; mlp1 only unless stated)

- pred_a_instrument: unpatched EVAL CE equals §2696's baseline_ce_eval 3.11250 within 1e-4; AND the mlp1 k = 32 arm
  reproduces §2696's .8834 within .015 (CUDA-atomics-free CPU path; tolerance kept for safety).
- pred_b_mlp1_dense (H-dense): mlp1's ladder is slow — CE added at k = 256 ≥ .20. Null: CE added at k = 256 ≤ .05.
- pred_c_isolated_token_program_mlp1 (H-token): the isolated-token table for mlp1 adds ≤ .40 — less than half the
  k = 32 subspace price, i.e. a token lookup is a better simple program for mlp1 than any 32-d subspace. Null: ≥ .883
  (no better than the rank-32 truncation).
- pred_d_token_R2_mlp1: R2_1 ≥ .5. Null: R2_1 ≤ .2.

## Reading rules

- b TRUE and c TRUE together: mlp1 is token-dominated but the token program is spread over hundreds of directions — the
  compilable object is the table, not a subspace. b TRUE and c FALSE (null): context arithmetic, dense — keep mlp1 whole.
  b FALSE (null) : §2696's k = 32 price was a fat-head effect and a k ≈ 256 subspace suffices (a cheap program after all).
- mlp0/mlp2/mlp3 rows and the joint early-MLP table arm are DISCLOSED only, not registered.
- Nothing installs into the §312 frontier; no metric-constructed bases (§2118 closed); natural text only.
- Failures recorded as scored; corrections separately.

## Price

CPU only. Table build: 50257 length-1 forwards (≈ 196 doc-equivalents, batched). Bases: 96 FIT docs. EVAL: 64 docs ×
(1 baseline + 5 ladder + 4 tables + 1 joint) = 704 document forwards. Estimated 10-14 minutes at 16 threads.
