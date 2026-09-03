# MLP0 context-target linear separability probe — preregistration (Claude CPU lane)

**Registered:** 2026-09-03 14:12 UTC. **Owner:** Claude (parallel CPU probe; zero forwards; weights only).
**Script:** `bilinear_quotient/ops/mlp0_context_target_linear_separability_probe.py`
(math in `ops/mlp0_hybrid_separability_lib.py`, hash-frozen). **Sibling:** token-target probe.

## Object

R536's second portable target (hybrid-pair addendum 13:42): CONTEXT target — observed difference
Dg = Dg_I + Dg_C (donor context, base token); a projector must return Dg_I (the token-by-context cross term). Same
question as the sibling: the irreducible residual of the best linear map (any rank, and rank-k) in the W_D output
metric; lower residual = more separable. Lower bound for R536's orthogonal-projector DAS on this target.

## Input model

Base token p_b ~ uniform over the 50257 trained unit-rms token rows (UNCENTERED 2nd moment M_p, since the token is
fixed, not differenced). Contexts q_b, q_d i.i.d. GAUSSIAN, zero-mean, E[qq^T] = rho^2 I, independent of the token
(Gaussianity is needed here for the 4th moments of g_C). Then Cov(Dg_I) = 2 rho^2 [(L M_p L^T)o(RR^T) +
(L M_p R^T)o(RL^T) + transpose + (R M_p R^T)o(LL^T)], Cov(Dg_C) = 2 rho^4 [(LL^T)o(RR^T) + (LR^T)o(RL^T)]
(Isserlis), and Dg_I (odd in q) is exactly uncorrelated with Dg_C (even in q), so the Wiener bound applies:
P* = S_I (S_I + S_C)^{-1}. rho scanned over {0.25, 0.5, 1, 2}; NONE of these numbers has been seen.

Prior knowledge disclosed: the sibling token-target case at rho=1 was observed during library smoke-testing
(Wiener residual .342; rank-32 residual .626; see its registration). The bars below are set with that in mind.

## Predictions (scored exactly as written; arms named)

- **pred_a_instrument_closed_form_matches_monte_carlo** — at rho=1, 4000 MC pairs (seed 2): MC E||Dg_I||^2 and
  E||Dg_C||^2 each within 5% of tr(Cov(Dg_I)), tr(Cov(Dg_C)); |normalized cross| <= 0.05; S_I min eig >= -1e-8 x max.
- **pred_b_no_exact_linear_separator_at_rho1** — output-metric Wiener residual at rho=1 >= 0.20 (no linear map of
  any rank recovers more than 80% of the I target from the context hybrid). Null: <= 0.10.
- **pred_c_rank32_read_insufficient_at_rho1** — output-metric rank-32 residual at rho=1 >= 0.60 (a 32-dim linear
  read recovers <= 40% of the I target). Null: <= 0.40.

## Price

0 forwards/backwards/parameters; CPU ~1-2 min per rho, ~6 GB RAM; float64 closed form, MC seed pinned.

## What each outcome licenses

Nothing about circuits or explained fraction. A pass says R536's I-target, like the T-target, cannot be carried
exactly by any fixed linear projector and needs > 32 dimensions for even 40% product-space fidelity under an
isotropic context model — so its registered "no fixed low-dimensional projector" null is structurally favoured and
its dimension ladder should start well above rank 3. A fail (near-separable) says the I/C split is the easier of the
two targets and the better first DAS pilot.
