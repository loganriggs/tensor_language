# MLP0 hybrid-target IN-SITU cross-corpus transfer — preregistration (Claude CPU lane)

**Registered:** 2026-09-03 14:58 UTC. **Owner:** Claude. **Script:** `ops/mlp0_hybrid_target_in_situ_cross_corpus_probe.py`
(imports the in-situ probe module for states/decomposition; CPU only; no full forwards). Written before the in-situ
probe (queued 14:53) has landed; none of the in-situ numbers have been seen.

## Object

§2688/§2689 found, under the STATED isotropic context model at rho=1, a sharp contrast: the TOKEN target's exact linear
separator is corpus-specific (natural->code transfer penalty .10-.11 vs within-corpus floor .01, i.e. ~1.4x the
destination's own residual), while the CONTEXT target's separator is transportable (penalty .012 on an own residual
of ~.26, ~5%). This probe asks whether that contrast REPLICATES with the model's real block-0 context vectors and the
real per-position renormalisation (same construction as the in-situ probe: hybrid pairs on xhat = s(p+q)).

Arms: corpora natural (fit_natural.pt) and code (ood_code.pt); document halves 0:96 / 96:192 within each; covariance
blocks accumulated per half and pooled exactly into whole-corpus blocks. General Wiener map P_a = C_to C_oo^{-1} per
arm. residual_under(P; b) = tr(W_D[S_tt - P C_ot - C_to P^T + P C_oo P^T]W_D^T)/tr(W_D S_tt W_D^T) with b's blocks.
Transfer penalty pen(a->b) = residual_under(P_a; b) - residual_under(P_b; b). Within-corpus floors pen(h0->h1),
pen(h1->h0). Response distance d(P_a,P_b) = ||W_D(P_a-P_b)Sigma^{1/2}||_F / mean(||W_D P_a Sigma^{1/2}||_F,
||W_D P_b Sigma^{1/2}||_F) with Sigma = pooled natural+code observed-difference covariance. LOWER residual = more faithful.

## Predictions (scored as written; penalties are read RELATIVE to the destination's own residual because the in-situ
levels are unknown at registration)

- **pred_a_instrument** — pooled whole-corpus own residuals equal the in-situ probe's frozen results for the same
  corpora within 1e-6 (same seed, same construction; read from `mlp0_hybrid_target_in_situ_separability_probe_results.json`
  whose hash is checked at run time if present, else from the recomputed blocks — the identity of the two code paths is
  the instrument); all blocks PSD; each half >= 10,000 samples.
- **pred_b_token_target_separator_corpus_specific_in_situ** — TOKEN target: pen(nat->code) >= .5 x own_code AND
  pen(code->nat) >= .5 x own_nat, each also >= 3 x the destination's within-corpus floor. Null: either pen <= .1 x own.
- **pred_c_context_target_separator_transportable_in_situ** — CONTEXT target: pen(nat->code) <= .1 x own_code AND
  pen(code->nat) <= .1 x own_nat. Null: either pen >= .5 x own.
Report-only: response distances vs split-half distances; absolute penalties; per-half own residuals.

## Price

0 full forwards/backwards/parameters; recomputes the block-0 states (2 x 192 docs) and 4 x 6 covariance blocks at half
size; 4 Wiener solves. ~6-12 min CPU, ~12 GB RAM.

## What each outcome licenses

No circuit claim. b and c both TRUE: the §2688/§2689 contrast is a property of the weights in situ — R536's clause B is
structurally binding for the TOKEN target and forgiving for the CONTEXT target under the model's own context
distribution. Either FALSE: the stated-model contrast does not survive the real context; the in-situ result supersedes
§2688/§2689 as the operative reading (recorded as a separate correction, both kept).
