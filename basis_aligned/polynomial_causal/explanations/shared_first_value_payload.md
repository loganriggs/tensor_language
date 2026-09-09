# Shared token values do not replace the contextual circuit payload

The new full-model screen rejects replacing the contextual values in four
previously identified heads with the shared first-layer token values alone.
This is a result on the actual545.9-million-parameter bilin18 model, separate
from the four-layer associative-lookup experiments. No smaller circuit has
been extracted from this replacement.

## The proposed reusable computation

Each selected attention head reads values

    value = (1 - lambda) * contextual_value + lambda * first_value.

The first value is computed from token embeddings before any attention and
reused by multiple layers. Contextual values are projected from each layer's
current state. The candidate removed contextual value projections in the
fixed L9H1,L9H4,L11H3,L15H5 union, at every position. Queries, keys, the shared
producer and all later states were recomputed normally. This was a literal
producer deletion, not a patch of cached donor head outputs.

The shared term has a negative coefficient in layers9 and11: lambda=-.65625
and-1.75. Layer15 has lambda=.55859375. These are not convex mixture weights,
and their signs alone do not determine the signs of downstream task effects.

The cohort contains32 previously licensed two-command rows with four endpoint
cells each:128 sequences. One command concerns will/had, the other is/was.
All texts, phases and templates were already opened. FIT/HOLDOUT are historical
split labels here; this new intervention does not supply fresh or OOD text.

## What the measured replacement did

The [registered screen](../BILIN18_SHARED_FIRST_VALUE_ONLY_V1_PREREGISTRATION.md)
ran through the managed GPU runner in3.228 seconds:128 forwards and512 sequence
evaluations. Nine primitive controls and all instrument checks pass. Every
non-padding token was scored, including native errors; there were1572 FIT
and1608 HOLDOUT token positions. Full distribution means all50304 vocabulary
entries, rather than a selected answer contrast.

| All-four replacement metric | FIT | HOLDOUT | Registered limit |
|---|---:|---:|---:|
| Mean teacher KL, nats/token | .048704 | .042429 | .001 |
| 99th-percentile teacher KL | .760107 | .611856 | .01 |
| Temporal paired-effect relative error | .828315 | .853024 | .01 |
| Is/was paired-effect relative error | .633685 | .604640 | .01 |
| Joint interaction / complete effect RMS | .189398 | .179812 | .01 |

The [result](../BILIN18_SHARED_FIRST_VALUE_ONLY_V1_RESULT.json) preserves each
template and both individual group deletions as diagnostics. Group A is
L9H1/H4; group B is L11H3/L15H5. Their combined effect differs from the sum
of their individual centered-logit effects by18–19%. Neither a group subset
nor a new mixture coefficient is adopted after the all-four failure.

The earlier capability license compared each command's answer against its
specified foil. Full-vocabulary native accuracy is a different measurement:
temporal .4375/.34375 and is/was .375/.328125 in FIT/HOLDOUT. The new screen
does not discard these native errors or convert answer/foil capability into
a claim of full-vocabulary correctness. Its fidelity target is the actual
native distribution, including mistakes.

The proposed deletion would omit589824 contextual value weights if it were
valid and compiled. It is not valid, and the hook prototype realizes no
savings: all545902902 native parameters remain. Shared first values still
exist architecturally, but they do not by themselves perform the required
payload computation at these consumers.

## Consequence for circuit identification

The full objective remains a previously unspecified reusable computation
with explicit inputs, operation and consumers; held-out/OOD prediction;
independent execution; accurately predicted removals and joint use; and a
smaller structural description including adapters and opaque weights.
This screen changes one dependency claim: the contextual payload producers
cannot be deleted under the registered fidelity standard. It establishes
neither a new circuit satisfying those properties nor a whole-model
impossibility theorem.

The [next factorial](../BILIN18_VALUE_COMPONENT_FACTORIAL_V1_PREREGISTRATION.md)
separately removes contextual values, shared first values, and both at the
same fixed consumers. It tests whether their signed command effects oppose
one another and whether their joint effect is additive. This is a causal
role test, not a replacement nomination. Removing a shared input at selected
consumers is distinct from deleting its producer everywhere. The new
primitive passes seven controls, including whole-head-zero equivalence and
restoration after an exception; the trained factorial is not yet run.

The preceding small-model [query-interface work](join_write_response_curve.md)
closed independent squared channels, proper linear blocks, and a single
shared linear gate. Those narrowly scoped certificates do not justify saying
that a coupled computation cannot reuse work. This full-model branch returns
to an actual shared input and its intervention semantics.
