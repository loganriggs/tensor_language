# Preregistration — MLP0 hybrid-target IN-SITU separability, CROSS-FITTED (honest out-of-sample) — Claude CPU lane

Registered 2026-09-03 15:12 UTC (system clock), BEFORE running. Script: `ops/mlp0_hybrid_target_in_situ_crossfit_probe.py`.
Results: `mlp0_hybrid_target_in_situ_crossfit_probe_results.json`. Price: CPU only, 0 GPU forwards, ~10 min (same 2 x 24,576
in-situ samples as §2690, same seed/draw order; 16 eigendecompositions of 4608-dim covariances).

## Why
§2691 showed the sample general Wiener map (4608 x 4608) overfits at 12-25k samples: within-corpus split-half penalties
(.08-.10 token, .20 context) exceed the in-sample residuals of §2690, so §2690's any-rank residuals are in-sample lower
bounds and §2691's cross-corpus contrast was floor-dominated. This probe replaces the in-sample instrument with a
cross-fitted one and re-asks both questions (separability bar; corpus specificity) honestly.

## Instrument
The natural (192 docs) and code (192 docs) corpora are split doc-major into quarters q0..q3 (48 docs x 128 positions =
6,144 samples each); h0 = q0+q1, h1 = q2+q3. Ridge Wiener map P_lam = C_to (C_oo + lam * tr(C_oo)/4608 * I)^-1, lam on the
grid {1e-8, 1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1, 3}. Nested cross-fit, direction 0: fit on q0, choose lam on q1 (minimum
residual), refit on h0 at that lam, evaluate on h1. Direction 1 mirrors (q2 -> q3 -> h1 -> h0). Reported residual = mean
of the two out-of-sample residuals (residual_under as in §2691, output metric W_D; LOWER = more faithful). Out-of-sample
rank-k ladder: SVD of W_D P (C_oo,train + lam I)^{1/2}, truncated to k in {3, 8, 32, 128, 512}, mapped back, evaluated
on the held half; capture_k = 1 - residual_k. Cross-corpus: the whole-source-corpus map at the source's direction-0 lam,
evaluated on each destination half; penalty = transfer residual - destination's own cross-fitted residual on that half,
averaged over the two halves.

## Predictions (scored exactly as written)
- pred_a_instrument: pooled quarters at lam = 1e-8 reproduce §2690's four any-rank residuals to <= 1e-6; every quarter >= 5,000 samples.
- pred_b_token_oos_within_bar: TOKEN target, natural corpus, cross-fitted residual <= .15 (the §2690 bar, now out of sample).
  Null: >= .25.
- pred_c_context_oos_outside_bar: CONTEXT target, natural corpus, cross-fitted residual > .15 (i.e. §2690's in-sample pass
  does not survive out of sample). Null: <= .15.
- pred_d_token_rank32_oos_capture: TOKEN natural, out-of-sample rank-32 capture >= .60 (in-sample §2690: .76). Null: <= .40.
- pred_e_token_cross_corpus_penalty: TOKEN cross-corpus penalty (honest) >= .05 in BOTH directions (natural->code and
  code->natural). Null: <= .02 in either direction.
- pred_f_context_cross_corpus_transports: CONTEXT cross-corpus penalty <= .05 in BOTH directions. Null: >= .10 in either.

## Reading rules
b and d decide whether R536-style low-rank TOKEN projectors (32-128 dims) are honestly supported in situ. c decides whether
§2690's context-target pass was an artefact of in-sample fitting. e/f re-test the §2688/§2689 contrast (token corpus-
specific, context transportable) with the floor removed. Failures are preserved; nulls are reported alongside. No circuit
claim; no explained-fraction change. Frozen inputs: in-situ module 71cf276e..., §2690 results bacf00a8..., checkpoint
680d6c26... (via the module), row caches 666a3201... / 6cf514e7... (via the module).
