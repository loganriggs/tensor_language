# MLP0 hybrid-target separability IN SITU (real block-0 context) — preregistration (Claude CPU lane)

**Registered:** 2026-09-03 14:52 UTC. **Owner:** Claude. **Script:** `ops/mlp0_hybrid_target_in_situ_separability_probe.py`.
CPU only; ONE exact block-0 attention pass (embedding -> attention0 -> MLP0 input) over the frozen copy-induction v2
row caches; no full-model forwards, no backwards, no GPU. **Written BEFORE any of the quantities below were computed.**

## Why

§2686/§2687/§2688/§2689 bounded R536's TOKEN and CONTEXT targets under a STATED context model (isotropic q at a scanned
rho = ||q||/||p||). The scans showed separability is decided by rho, and I asked Codex to report the realistic rho. That is
unnecessary: MLP0's input is a one-block computation, exactly reproducible on CPU. This probe replaces the stated model
with the model's own block-0 context vectors and the real per-position rms renormalisation, and computes the same
Wiener/reduced-rank bounds with the GENERAL (correlated target/nuisance) formula.

## Object (exact semantics of `jacclust/tt_model.py`)

Block 0: x = lambdas[0]*x0 + lambdas[1]*x0 with x0 = rms_norm(wte[tok]) (lambdas = (6.09, 6.09) in bf16, so the token
stream is scaled by ~12.2 before the attention residual add); attention input rms_norm(x) = x0; squared attention with
UNNORMALISED pattern (scores/128)*(scores2/128), causal-masked, v mixed with v1 (= v in block 0); q_t := c_proj output at
position t. MLP0 input xhat_t = rms_norm(p_t + q_t) = s_t (p_t + q_t), p_t := (lambdas[0]+lambdas[1]) x0_t.
Product activation g(x) = (Lx)*(Rx) = s^2 [g_T(p) + g_I(p,q) + g_C(q)]. rho_t := ||q_t|| / ||p_t||.

Hybrid pairs (one replacement per sampled position, seeded):
- TOKEN target: replacement token t' ~ corpus unigram, x' = s'(p'+q_t). Observed D = g(x') - g(x); target
  D_T = s'^2 g_T(p') - s^2 g_T(p); nuisance = D - D_T (the interaction change PLUS the rms-leak (s'^2 - s^2) g_C(q)).
- CONTEXT target: donor context q' from a uniformly drawn other sampled position of the same corpus, x' = s'(p_t+q');
  target D_I = s'^2 g_I(p,q') - s^2 g_I(p,q); nuisance = D - D_I.
Sampled positions: 1..255 step 2 (128 per doc) x 192 docs = 24,576 per corpus (natural = fit_natural.pt, code = ood_code.pt).

Bounds (output metric W_D, LOWER = more separable): general Wiener residual
r = tr(W_D [S_tt - C_to C_oo^{-1} C_ot] W_D^T) / tr(W_D S_tt W_D^T) with centred sample covariances (ridge 1e-8 tr/n
on C_oo); rank-k ladder from the singular values of W_D C_to C_oo^{-1/2}, k in {3, 8, 32, 128, 512, 1152};
pure-target ladder and effective rank of W_D S_tt W_D^T for reference.

## Predictions (scored as written; natural corpus is the scored corpus, code is reported)

- **pred_a_instrument** — (i) checkpoint sha256 equals 680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3;
  (ii) my manual block-0 attention reproduces `jacclust.tt_model.CausalBilinearSelfAttention` run on CPU with the same
  weights on 4 natural docs to max|diff| <= 1e-3 x rms(output); (iii) decomposition identity
  ||g(x) - s^2(g_T+g_I+g_C)|| / ||g(x)|| <= 1e-4 on a 2048-row chunk; (iv) S_tt and C_oo PSD (min eig >= -1e-8 max)
  for both targets and both corpora; (v) >= 20,000 samples per corpus.
- **pred_b_token_stream_dominates_mlp0_input** — median rho_t over natural sampled positions <= 0.5. Null: >= 1.0.
  Reason registered in advance: the learned lambdas scale the token stream 12.2x before the attention output is added;
  I read that as the model keeping MLP0 in §2686's near-separable regime. I do not know the answer.
- **pred_c_token_target_near_separable_in_situ** — natural TOKEN-target Wiener residual (any rank) <= .15. Null: >= .30.
- **pred_d_context_target_near_separable_in_situ** — natural CONTEXT-target Wiener residual (any rank) <= .15. Null: >= .30.
Report-only: rho by position bucket (1-4, 5-24, 25-124, 125-255) per corpus; code-corpus residuals; rank ladders; the
rms-leak energy fraction ||(s'^2-s^2) g_C(q)||^2 / ||D||^2 (token target) and the analogous g_T leak (context target);
pure-target effective ranks.

## Price

0 full forwards/backwards/parameters. One block-0 attention pass on 2 x 192 x 257 tokens (fp32 from bf16 weights),
six 4608 x 4608 float64 covariance accumulations from ~24.6k samples each per corpus, a few 4608 eigen/solves.
Estimated 5-12 min CPU (16 threads), ~12 GB RAM.

## What each outcome licenses

No circuit claim. If b, c, d hold: R536's Stage-B targets are, in situ, in the near-separable regime — a linear
projector can carry ~85%+ of either target, the rank ladders reported here become the registered reference for
R536's projector ranks, and the §2686/§2687 rho=1 numbers are retired as the operative reference (they stay as
bounds under the stated model). If b fails (rho ~ 1 or more): the §2686-§2689 rho=1 rows are the operative reference
and the I-ladder-in-the-hundreds warning stands. If b holds but c or d fails: the real context is anisotropic in a
way that defeats the isotropic scans — an instrument correction to §2686/§2687's stated model, recorded separately.
