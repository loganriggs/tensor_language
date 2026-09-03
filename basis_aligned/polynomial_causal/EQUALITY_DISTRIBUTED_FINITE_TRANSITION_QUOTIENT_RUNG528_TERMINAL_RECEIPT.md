# Rung 528 terminal receipt: whole post-MLP12 transitions are similar but not interchangeable

**Completed:** 2026-09-03 11:13 UTC

**Status:** valid strong null at the complete-transition grain

## Computation and instrument

For each of the four correctly oriented equality-score actions `N/P/Z7/Z8`, rung 528 captured the complete raw
residual change from the common score-absent run immediately after MLP12. It physically inserted each change back
into the score-absent trajectory and ran four real suffixes: native, attention14 held to its absent write, MLP17 held
absent, and both held absent. Circuit fingerprints were member-minus-matched-control CE effects over 32 fixed circuit
families.

The instrument passed exactly. Direct native replay, all four self-inserted boundaries and logits, the embedding skip
state, and the attention first-value state had maximum error zero. All transitions and continuation patches were
live. The run executed and reconciled `1,984` forwards, zero backwards, 1,984 boundary captures, 1,488 boundary
insertions, and 1,488 continuation-write patches. Runtime was `39.12 s`; peak GPU memory was `4,087,619,072` bytes.

## Frozen discovery result

All three action pairs were materially active and beat the permutation and wrong-sign controls by very large margins,
but none passed every pre-registered proportional-equivalence clause.

| Pair | fitted positive scale | D0 circuit cosine | D0 relative error | D1 circuit cosine | D1 relative error | result |
|---|---:|---:|---:|---:|---:|---|
| `N` vs `P` | 0.596 | 0.841 | 0.540 | 0.903 | 0.574 | fail |
| `N` vs `Z7` | 0.807 | 0.914 | 0.405 | 0.949 | 0.392 | fail |
| `N` vs `Z8` | 0.721 | 0.919 | 0.394 | 0.931 | 0.399 | fail |

The D0 requirements were cosine at least `.90` and relative error at most `.35`. `Z7` and `Z8` clear the cosine
bar but miss the error bar. For a scalar fit on the same vector, those clauses are related: a `.35` relative-error
bar effectively requires cosine about `.937` or better. Their unexplained squared circuit-response fractions are
therefore about `16.4%` and `15.5%`, not zero. `P` is farther away.

The aggregate task effects were much closer: all task cosines were at least `.995`. Each continuation separately
also preserved the broad relation for `Z7/Z8`, with circuit cosines `.894--.955`. Wrong-sign cosines were negative
(`-.851` and `-.796`), while permutation 95th percentiles were only `.192--.222`. Thus this is not a dead effect,
sign-gauge failure, chance circuit alignment, or one continuation hiding a disagreement. It is a real statement that
the complete state changes contain a large common task direction plus enough action-specific circuit response to
prevent whole-state interchangeability.

No scaled physical substitution, confirmation documents, or held-out circuit families opened because discovery
returned zero candidates. There is no circuit, quotient, or compression claim.

## Post-result diagnostic that changes the next object

Using only the already-open aggregate discovery fingerprints, each action was put in native-action units with the
three frozen scales. For each target, the mean of the other three aligned responses was formed without using the
target. This leave-one-action-out consensus improved the `Z7` response to cosine `.950/.953` and relative error
`.313/.314` on D0/D1. `Z8` improved to cosine `.934/.933` and error `.357/.360`. The residual between each target and
its leave-one-out consensus was not stable across halves: residual cosines ranged from `-.044` to `.197`.

This is descriptive, not yet causal evidence: averaging response fingerprints is not the same as averaging and
inserting the actual boundary states through the nonlinear suffix. It nevertheless supplies a discriminating next
hypothesis. The whole transition may be a shared computation plus unstable or implementation-specific residue, so
the next test should use a leave-one-action-out state consensus and compare it with every single-donor replacement.
That changes the object rather than weakening rung 528's bars.

The two-continuation factorial interaction was material: its norm was `13.3--19.8%` of the native-continuation
fingerprint across actions and halves. The next test must therefore retain all four continuations rather than revert
to single-component patch rankings.

## Frozen consequence

Close proportional equivalence of complete post-MLP12 transitions. Do not lower the `.35` bar, refit on D1, choose
only a favorable continuation, or use rank/quantization. A prospective successor may test the explicitly different
shared-plus-private hypothesis by physically inserting a leave-one-action-out consensus state on new documents and
held-out circuit families.

Immutable artifacts:

- result SHA-256: `f931e5fb6f618b002203ce1e870a8ad4442ed3a38a7475809754ab2de91554b6`
- sufficient-statistics SHA-256: `c17db82832a76daba23f74e57e75abc258093c6820c79c93a62d8d29b6143d38`
- runner SHA-256: `69e728bae2b67fcdc30beebbdc0e65981646d6dbfe474743e37d46e22cd89427`

The independent terminal audit recomputed every scale, circuit/task cosine, and relative error from the aggregate
sufficient statistics, verified all frozen hashes and call counts, and confirmed that every conditional stage stayed
sealed. It found three material pairs and zero pre-control passers, matching the result.
