# Rung 531 terminal receipt: equality score-factor sharing

**Completed:** 2026-09-03 12:13 UTC

## Outcome

The instrument passed and the registered strong null is true. Across all 12 directed pairs among `L5H5`, `L7H3`,
`L8H3`, and `L8H4`, no pair shared both multiplicative score factors and no pair shared exactly one factor under the
frozen held-out thresholds and causal-prefix permutation control.

This rejects **scalar-gauge equality of the complete factor functions on equality edges**. It does not prove that
the heads share no lower-level vocabulary, nor that downstream computation cannot treat different-looking factors
as equivalent when composed with a particular companion. Those require a different, downstream-effect-defined
test.

## Frozen artifacts

- runner SHA-256: `e2eb9bd2674247c1fa1c0e25a50d4e747b899a2883899f3074bf809bc676f71e`
- result SHA-256: `016d4e7babaf2fa562ee254e76ea8c354a7448ddb9fb70cf4be6c835c77354ab`
- sufficient-statistics bundle SHA-256: `62f3a224eee35b067a79297f967410c2eb342df13e868f9bc31f2ad4de534442`
- audit SHA-256: `f0d6894addd3251c7dd7b36d346475789cf8b50fc47d624faef2b56383c15f2d`
- checkpoint SHA-256: `680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3`

## Instrument and price

- exactly 125 frozen-model forwards over rows `0:500`
- no backward passes, learned vectors, activation interventions, validation rows, or OOD rows
- factor product versus the independently replayed parent product: maximum absolute difference `0`
- parent factor reconstruction error: at most `5.09e-14`
- every split/control and every factor/product norm live
- only aggregate dot products retained; no tokens, logits, states, or per-edge factors retained

The independent audit reloaded the bundle, regenerated all 12 pair reports and four prediction gates, reconciled all
125 calls, and confirmed that rows `500:1000` remained sealed.

## What bound

The best held-out factor cosine was `85.98%` (`L5H5 -> L7H3`, second mapped factor), below the registered `90%` bar;
its relative error was `51.07%`, above the `45%` bar. For the three parent-authorized product pairs into `L8H4`, the
best factor cosines were:

- `L5H5 -> L8H4`: `83.10%` and `70.79%`;
- `L7H3 -> L8H4`: `85.89%` and `71.54%`;
- `L8H3 -> L8H4`: `84.41%` and `77.25%`.

All three selected the swapped factor assignment, and the assignment was stable across both confirmation halves.
Their permutation margins were large, so the moderate similarity is real structure rather than a simple positional
match. It is nevertheless not accurate enough to identify either entire factor as the same computation.

## Decision

Close one-to-one factor matching; do not lower the thresholds or add rank. The result-conditioned successor should
define sameness through downstream computation: compose a frozen source factor with the target's native companion,
then compare its causal effect across the 32 discovery and 30 held-out circuit families. That tests the user's
interaction-determined basis directly and can expose factor reuse even when raw score matrices differ.
