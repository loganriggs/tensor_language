# MLP0 context-target CROSS-CORPUS separability probe — preregistration (Claude CPU lane)

**Registered:** 2026-09-03 14:50 UTC. **Owner:** Claude. **Script:** `ops/mlp0_context_target_cross_corpus_separability_probe.py`
(math: `ops/mlp0_hybrid_separability_lib.py` + `ops/mlp0_hybrid_separability_corpus_lib.py`, hash-frozen). Zero forwards.

## Object

CONTEXT target (observed Dg_I + Dg_C -> Dg_I; §2687). §2686/§2687 used a UNIFORM token distribution. R535 found the equality S/R coordinates corpus-unstable (natural
3/6 sign cells) and my R536 power-gate proposal makes cross-corpus stability the binding clause. This probe asks the
exact linear-regime version: does the best linear separator (Wiener map P* = S_tgt(S_tgt+S_nui)^{-1}, rho=1) fitted under
the NATURAL token distribution differ from the one fitted under the CODE token distribution by more than within-corpus
sampling noise, and does a projector fitted on one corpus lose fidelity on the other? Here the token distribution enters only through the UNCENTERED base-token moment in S_I (S_C is corpus-free), so a smaller corpus effect is expected a priori.

## Arms

Token unigram distributions from the frozen terminal-copy-induction v2 row caches (outcome-irrelevant inputs; only
token ids are read): natural = fit_natural.pt rows (192 x 257), code = ood_code.pt rows (192 x 257); within-corpus
halves = documents 0:96 vs 96:192. Maps: P_nat, P_code, P_nat_h0, P_nat_h1, P_code_h0, P_code_h1, and P_uniform
(the §2686/§2687 map, reproduced through the weighted code path as the instrument check). Metrics: Codex's response
distance d(P_a,P_b) = ||W_D(P_a-P_b)Sigma^{1/2}||_F / mean(||W_D P_a Sigma^{1/2}||_F, ||W_D P_b Sigma^{1/2}||_F) with
Sigma = pooled natural+code total covariance; transfer penalty pen(a->b) = residual(P_a under corpus b) -
residual(P_b under corpus b) (output metric; LOWER residual = more faithful). Within-corpus penalties pen(h0->h1),
pen(h1->h0) per corpus are the noise floor.

## Predictions (scored as written)

- **pred_a_instrument_uniform_map_reproduces_frozen_result** — uniform-weight path reproduces §2687 rho=1 residual .275 within 1e-4;
  each corpus has >= 2000 distinct tokens; all covariances PSD (min eig >= -1e-8 max).
- **pred_b_cross_corpus_map_disagreement_exceeds_split_half_noise** — d(P_nat,P_code) >= 2 x max(d(P_nat_h0,P_nat_h1),
  d(P_code_h0,P_code_h1)). Null: <= 1.2 x.
- **pred_c_cross_corpus_transfer_penalty_is_material** — pen(nat->code) >= .05 AND pen(code->nat) >= .05, each >= 3 x the
  larger within-corpus half penalty of the destination corpus. Null: either pen <= .02.

## Price

0 forwards/backwards/parameters; ~7 Wiener solves (4608, float64) + covariance passes; ~2-4 min CPU, ~8 GB RAM.

## What each outcome licenses

No circuit claim. If b and c hold: even the exact linear separator of this target is corpus-specific — a projector
fitted on one corpus is not transportable — so R536's clause-B cross-corpus gate is structurally expected to bind and
a single-corpus DAS fit cannot be read as the model's variable. If both fail: the linear regime is corpus-robust and
any R536 corpus instability would be a fitting/power artifact, not a structural one — which would move clause B from
binding to diagnostic.
