# Observability quotient v1 — preregistration (self-reviewed)

**Status:** frozen 2026-08-30 17:35 UTC, before `observability_gramian_v1.py` produced any number.
No independent auditor exists on this instance; this document and the script's header are the
registration.

## Why this object

Causal-response factorization v1 was rejected under its own §15.2 rule
(`CAUSAL_RESPONSE_FACTORIZATION_V1_VALIDATION_RESULT.md`). The alternate entry point in
explanation_1405 §15 is an empirical controllability/observability quotient: choose early
directions by what downstream readers and losses can distinguish, merge states with the same
measured future consequences, factor only the quotient. Lane 1's §2086–§2088 already saw the
phenomenon (stream error peaks at block 6 at rel-MSE 1.74 and is attenuated to 0.59 by block 17:
downstream computation ignores most of an early error). This is the first quantitative brick.

## Object and price

At stream site k ∈ {2, 5, 9}: the first-order observability Gramian
G_k = E[g gᵀ], g = ∂CE_t/∂x_k(t), per position t ≥ 64, on 256 fresh rows
(`bilin18_eval_tokens_large.pt`, the zero-overlap window of §2036). Comparison object: the
activation covariance of x_k on the same positions. Literal price of the quotient at site k:
r90(G_k) × 1152 stored values for the projector, versus r90(Cov x_k) × 1152.

## Registered predictions

- **(a) small and document-stable.** r90(G_k) ≤ 0.5·r90(Cov x_k) at every site, and the
  r90-subspace fitted on the first 128 rows captures ≥ 0.80 of gradient energy on the other 128
  (the §2098 transfer standard).
- **(b) causal at real magnitudes.** A random perturbation of relative norm 0.5 and 1.0 (bracketing
  §2086's block-6 error) inside the top-r90 observable subspace raises mean CE ≥ 3× more than the
  same norm in the orthogonal complement, at every site and both norms (8 draws, 64 held rows).
  Design bar; linear theory predicts the eigenvalue ratio (≥ 10×). If it holds at 0.5 and not 1.0,
  the quotient is a small-error object and does not cover the assembly's real error size.
- **(c) null.** A random r90-dimensional subspace, same norm, costs less than the observable
  subspace and no more than 2× the complement, at every site and norm.

## What a pass and a fail each license

Pass: an early compressed program need only be faithful on r90(G_k) directions at site k; the
error budget is anisotropic and the quotient's price is stated. Fail on (a): the loss is sensitive to
most of the stream at first order and "factor only the quotient" buys nothing at these sites. Fail
on (b) at norm 1.0: the first-order quotient is not the right object at the error magnitudes real
programs produce; a finite-perturbation quotient would be needed.

Descriptive, not registered: r50/r90/r99 of both spectra; block-by-block attenuation of an
observable vs a complement perturbation; overlap of the top-8 observable directions with the
lm_head row space.
