# MLP0 token-target linear separability probe — preregistration (Claude CPU lane)

**Registered:** 2026-09-03 14:12 UTC. **Owner:** Claude (parallel CPU probe; zero forwards; weights only).
**Script:** `bilinear_quotient/ops/mlp0_token_target_linear_separability_probe.py`
(math in `ops/mlp0_hybrid_separability_lib.py`, hash-frozen). **Sibling:** context-target probe (separate registration).

## Object

R536's hybrid-pair addendum (13:42) registers two portable MLP0 targets on the 4608-dim product activation
g(p,q) = g_T(p) + g_I(p,q) + g_C(q). The TOKEN target: observed difference Dg = Dg_T + Dg_I (donor token, base
context); a learned projector P must return Dg_T. The addendum's registered null is "there may be no fixed
low-dimensional projector that performs this separation". This probe computes, exactly from the weights, the
IRREDUCIBLE residual of the best LINEAR map — any rank (Wiener bound) and rank-k (reduced-rank regression) — in the
W_D-weighted output metric (what downstream sees). Lower residual = more separable (residual 1 = nothing recovered,
0 = exact separation). This is a LOWER BOUND on what R536's orthogonal-projector DAS can achieve in product/output
space; downstream CE may be more forgiving. It is a structural prediction of R536's dimension ladder before any GPU.

## Input model (stated, not fitted)

Token part p_t = wte_t / rms(wte_t), uniform over the 50257 trained tokens (proxy for the token part of the
normalized MLP0 input, as in §2673/§2676). Context part q: zero-mean, E[qq^T] = rho^2 I, independent of the token
(all-contexts reading; only 2nd moments needed for this target). rho = ||q||/||p|| is scanned over {0.25, 0.5, 1, 2}
because the realistic ratio is not known on CPU — R536 Stage-B1 can read it from real activations. Under this model
Dg_T and Dg_I are exactly uncorrelated (Dg_I is linear in q, E[q]=0), so the Wiener map is
P* = S_T (S_T + S_I)^{-1}, S_T = Cov_tokens(g_T), S_I = rho^2 [(L S_p L^T)o(RR^T) + (L S_p R^T)o(RL^T) + transpose
+ (R S_p R^T)o(LL^T)] with S_p the centered token covariance. Rank-k: residual_k = tr(D S_T D^T) - sum_{i<=k}
sigma_i^2(D S_T (S_T+S_I)^{-1/2}). Reference: the pure-target ladder (rho=0) = energy outside the top-k
eigen-directions of D S_T D^T.

## DISCLOSURE — partially observed before registration

While smoke-testing the library (14:05 UTC) I ran the rho=1 token-target case once and SAW: output-metric Wiener
residual 0.342 (product metric 0.469); rank ladder k=3/8/32/128/512: .845/.756/.626/.517/.394; pure-T ladder
.832/.727/.564/.392/.134, pure-T output effective rank 317; MC traces within 0.5% of closed form; normalized cross
0.0013. The bars I had DRAFTED but not registered were "rho=1 residual >= .25" (would pass) and "rank-32 residual
>= .75" (would FAIL — a 32-dim linear read recovers 37% of the T-target variance, more than I guessed). Those rho=1
numbers are therefore REPORTED, NOT SCORED. The scored predictions below concern only unseen quantities (the
rho scan's shape) and the instrument.

## Predictions (scored exactly as written; arms named)

- **pred_a_instrument_closed_form_matches_monte_carlo** — at rho=1, 4000 MC hybrid pairs (seed 1): MC
  E||Dg_T||^2 and E||Dg_I||^2 each within 5% of 2 tr(S_T), 2 tr(S_I); |normalized cross inner product| <= 0.05;
  S_T min eigenvalue >= -1e-8 x max. Instrument-invalid otherwise (no other clause is read).
- **pred_b_residual_monotone_and_large_at_rho2** — output-metric Wiener residual is strictly increasing across
  rho = 0.25 < 0.5 < 1 < 2 AND residual(rho=2) >= 0.50. Null: residual(rho=2) < 0.35 (nuisance saturates —
  separability barely degrades with context scale).
- **pred_c_near_separable_at_small_context** — output-metric Wiener residual at rho=0.25 <= 0.15 (when context is
  a quarter of the token norm, a linear map recovers >= 85% of the T target). Null: >= 0.30.

## Price

0 forwards, 0 backwards, 0 deployed parameters; CPU ~1-2 min per rho (4608x4608 float64 eigh/solve + one
1152x4608 SVD), ~6 GB RAM. Reproduction tolerance: 1e-6 relative (float64 closed form); MC seed pinned.

## What each outcome licenses

Nothing about circuits or explained fraction. If pred_b/pred_c hold: the token-target's linear separability is a
strong function of context scale, so R536 must report the realistic rho before its dimension ladder is
interpretable, and the rho=1 numbers (rank-32 recovers ~37%, no rank recovers the last ~34%) bound the ladder.
If pred_c fails (residual large even at rho=0.25): the two branches are entangled at every scale and the
"no fixed projector" null is structurally forced.
