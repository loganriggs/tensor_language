# Hourly strategic review — 2026-09-01 20:36 UTC

## Current goal

Compile bilin18 into a smaller predictive, manipulable tensor program whose parts are justified by held-out
computation and causal downstream effects. For attention0 specifically, replace the architectural-head basis with
the smallest executable query/key/composition/payload program that preserves language-model behavior, then decide
whether its coordinates have stable semantics.

## What changed in the last hour

- Rung 424 found a near-lossless continuous six × six × 32 decomposition of the realized QK1 × QK2 × payload
  product, with +0.000200 nat SELECT damage, but retained every native generator.
- Rung 425 reproduced it on unused documents.
- Rung 426 found the first surviving cross-head sparse token vocabulary: G54 is 18.8446% smaller than 18 independent
  dictionaries, improves product error 1.4740→0.64375, and survives same-token permutation and no-native-QK tests.
  Its individual atom-identity predicate missed one raw-pair bar, while routed output and CE strongly depended on
  the learned coupling.
- Rung 427 reproduced the 426 document ordering on fresh rows; the G72-over-I72 CE margin remains positive but
  shrinks from .00317 to .00084 nat.

## Is the coupled sparse score experiment still the best next step?

Yes, with bounded claim level. It changes both the query/key factorization and the training objective, directly
answers the user's requested second section-14.3 extension, and resolves the specific ambiguity left by 426: whether
atom coupling is weak in raw pair geometry because factor MSE learned the wrong coordinates, or because no stable
discrete composition exists. It is cheap, uses the already validated physical execution path, and has controls that
separate real query×key relations from capacity.

It must not displace the continuous physical generator if it only improves interpretability while remaining around
+.02 nat damage. Rung 424 is roughly two orders of magnitude better in CE but has no saving; rung 430 is useful only
if it either moves sparse fidelity materially toward 424 or identifies stable atom-pair structure. The comparison is
therefore complementary rather than winner-take-all at this stage.

## Alternatives reconsidered

1. **Direct continuous composite generator now.** Highest adoption relevance because 424/425 are near-lossless.
   Retain as the next parallel scientific family after the short sparse screen. It must emit score modes without
   native Q/K and solve the full value broadcast, with matched ordinary-rank and existing shared-QK baselines.
2. **Downstream-62 metric first.** Could choose the right basis, but the universal damage ray makes raw certificate
   counts easy to game. Defer until there is an executable candidate; then use document-disjoint response vectors,
   tag permutation, and matched CE/price.
3. **Pure semantic inspection of rung-426 atoms.** Premature because D failed and rotations/restarts remain
   untested. Rung 428's atom-pair concentration and restart matching are the minimum gate before naming.
4. **More atoms/k/rank tuning.** Rejected for now: it would read SELECT and does not change the scientific object.
5. **Full Tucker/CP of the dense QK×payload tensor.** Retained only if sparse pair stability fails or as the direct
   continuous generator parameterization. Raw coefficient-space decompositions without downstream/product metrics
   duplicate old nulls.
6. **Head-6 characterization.** Mechanistically interesting after 423, but lower leverage for the global goal than
   deciding whether the new sparse vocabulary composes. It can proceed independently without blocking 428.

## Frozen decision after review

Run rung 430 once at its preregistered 512-atom k27+k27 and k36+k36 prices. Do not tune after SELECT. If coupled
training passes computation but not stability, retain the sparse generator and treat its atoms as a non-unique
coordinate system. If it misses computation or the strong null fires, close this sparse-composition budget and move
immediately to the direct continuous composite generator. If it passes both, fresh/OOD and 62-behavior tests precede
any semantic name or adoption claim.

This review changes no registered rung-430 bar, arm, seed, split, or objective. The number was repaired after the
review because the red-team lane had already claimed and executed rungs 428 and 429; no scientific content changed.
