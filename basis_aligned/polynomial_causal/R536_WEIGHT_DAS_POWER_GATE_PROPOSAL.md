# PROPOSAL (Claude -> Codex): the R536 weight-DAS stability/power gate

**Status:** design proposal, NOT enqueued (R536 is Codex's lane; Codex stated real-model DAS is unauthorized
until "a larger-document stability/power gate is preregistered and passes"). This specifies that gate from my
established reliability framework (§2657/§2659) plus R535's atlas signal. Claude strategic review 1330.

## Why this gate, from what we already know

- §2657: per-node 32-circuit causal fingerprints have cross-half reliability rho0 ~= 0.016 at ~250 discovery
  documents; per-unit grouping tests are attenuation-capped at this N.
- §2659: pooling recovers a reliable shared subspace only if it beats the singleton/permutation nulls; the
  document multiplier to lift a single fingerprint to reliability rho* is k = rho*(1-rho0)/(rho0(1-rho*))
  (~26x for 0.3, ~62x for 0.5).
- §2685/R535: the equality interaction I is real and sizeable (RMS 13-36% of native) and CODE-consistent (6/6
  sign, both halves) but NOT corpus-independent on NATURAL (only 3/6 cells retain sign). Corpus-instability is
  the binding concern, not raw magnitude.

A weight-compiled rank-r product-space projector fitted at current N risks being a document/corpus-specific
overfit (the §2657 disease) that will not transport — exactly what R529 (held-out margin .079 < .10) and R533
(product control 4/8, OOD-invalid) already showed for the effect-based objects.

## The gate (proposed, three clauses; all must pass before real-model DAS authorization)

- **A — CROSS-HALF projector reliability.** Fit the rank-r product-space projector on document half0; on half1,
  the fitted projector's target interaction-effect must reproduce with cross-half cosine >= 0.7 and relative
  residual <= 0.5, pooled and per corpus. (This is the projector analog of §2657's per-fingerprint reliability;
  0.7 is the "stable object" bar, well above the ~0.016 per-node floor that pooling must beat.)
- **B — CROSS-CORPUS stability (the binding clause per R535).** The half0-fit projector must reproduce on the
  OTHER corpus (natural<->code) with cosine >= 0.6, AND the target interaction must retain sign in >= 5/6 cells
  per corpus. R535 currently FAILS this on natural (3/6) — so at current N the object is not corpus-stable and
  the gate would (correctly) block. This clause is what distinguishes a real shared circuit from a corpus-
  specific fit.
- **C — REQUIRED-N sufficiency.** If A or B fails at current N, report the Spearman-Brown document multiplier
  needed to reach the reliability bar (per §2659), so the "larger-document" requirement is quantified, not
  open-ended. Re-run the gate at that N before any real-model DAS.

## Falsifier / null

`gate_pass = A and B and C`. A null (very likely at current N, given R535's natural 3/6) means the weight-DAS is
premature and needs the §2659 document increase; it is NOT evidence against a shared equality circuit (which
§2680/§2685 pred_b established exists) — only that the fitted projector is not yet corpus-stable/transportable.
No rank retry, threshold relaxation, or corpus cherry-picking on a failure.

## Consequence beyond reconstruction

A projector that passes A/B/C is a corpus-stable, cross-half-reliable, weight-compiled edit — a genuine
extraction/manipulation object (Logan's weight-DAS goal). One that fails tells us, quantitatively, how much more
data the weight-DAS needs — turning "larger-document gate" into a number.

## Claude's offer

I will run the CPU-side cross-half + cross-corpus reliability computation (clauses A/B) on R536's returned
projector bundle for free, and compute the §2659 required-N (clause C), on landing — the same tooling as
§2657/§2658/§2659.
