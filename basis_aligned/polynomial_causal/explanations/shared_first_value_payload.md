# Shared token values do not replace the contextual circuit payload

The new full-model screen rejects replacing the contextual values in four
previously identified heads with the shared first-layer token values alone.
This is a result on the actual 545.9-million-parameter bilin18 model, separate
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

## Component cuts and command interactions

The [component factorial](../BILIN18_VALUE_COMPONENT_FACTORIAL_V1_RESULT.json)
now completed on the managed GPU in 3.238 seconds. Native replay and the
independent whole-head-zero oracle agree exactly. Shared and contextual
command effects have positive cosines (.381–.943), rejecting the proposed
opposition despite negative mixture coefficients. Their independent-effect
approximation also fails: full-token joint interaction is 10.4–12.0% of the
complete joint effect. Removing only the shared inputs gives mean KL
.001504/.001407, still above .001. Neither component-only replacement passes.
These are consumer-local cuts, not deletion of the shared producer everywhere.

The [full-response command-mode screen](../BILIN18_SHARED_RESPONSE_COMMAND_MODES_V1_RESULT.json)
completed in 1.765 seconds. For each world and token, it decomposes the full
centered-logit shared-removal response over the four command cells. Orthogonal
projection gives an exact lower bound for any command-independent response:
37.4%/37.3% relative RMS error over all tokens, 24.5%/22.3% at the temporal
query, and 47.8%/51.4% at the is/was query. These are L2 response bounds,
not KL bounds or fitted predictors. Reconstruction, Parseval, parent KL
replay, and the earlier query's future-command zero all pass.

The [integer producer audit](../BILIN18_SHARED_VALUE_PRODUCER_INCIDENCE_V1_RESULT.json)
provides a positive exact result: all 795 aligned source positions satisfy
count(token00)+count(token11)=count(token01)+count(token10). Consequently,
any token-only value map has zero mixed command mode on these worlds.
The full removal response nevertheless has a mixed term, giving a lower
bound of 7.2–7.3% for any additive command response over all tokens, and
13.7–15.4% at the later query. The interaction therefore arises downstream
of shared-value production. This does not identify which reader creates it.

## Next: locate the interaction in the native read

The [mathematical review](../THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-09_2249.md)
derives an exact routing/payload convolution. If S=P U is the native shared
read and hats denote balanced command modes, then

    S_hat11 = P_hat11 U_hat00 + P_hat01 U_hat10 + P_hat10 U_hat01,

because U_hat11=0. The first term is joint routing; the other two cross
single-command routing with the other command's payload. The
[registered next screen](../BILIN18_ROUTED_SHARED_COMMAND_INTERACTION_V1_PREREGISTRATION.md)
keeps every fixed head and both queries, validates the native contraction,
and tests whether the crossed terms alone suffice. Its primitive passes
eight planted controls; trained capture and scoring have now completed. One initial
toy negative control had effect norm exactly equal to its strict >1 bar;
its planted payload amplitude was doubled before any trained run, preserving
the bar. No scientific threshold or trained result changed.

Non-additivity does not invalidate circuit compositionality: a reusable
computation can predict its interaction explicitly. All these results use
the same opened cohort and all 545902902 native parameters; fresh/OOD
prediction, independent extraction, and structural reduction remain unproved.

The preceding small-model [query-interface work](join_write_response_curve.md)
closed independent squared channels, proper linear blocks, and a single
shared linear gate. Those narrowly scoped certificates do not justify saying
that a coupled computation cannot reuse work. This full-model branch returns
to an actual shared input and its intervention semantics.

## Native read result and next causal test — 23:06 UTC

The [read-convolution result](../BILIN18_ROUTED_SHARED_COMMAND_INTERACTION_V1_RESULT.json)
passes instrument and material-interaction gates, but rejects crossed-only
sufficiency. At the later query, the mixed read is 3.25%/3.58% of the complete
shared read RMS, FIT/HOLDOUT. Dropping joint routing leaves 77.8%/75.0%
relative mixed-read error. These are pooled norms over separate layer/head
entries; no cross-layer sum is interpreted as one intervention. The earlier
query has exactly zero mixed read. Native oracle max absolute error is
2.14e-5, relative RMS 1.33e-7; convolution closes to 1.42e-14. All fourteen
capture/algebra controls pass. Managed execution took 1.880 seconds, with
32 full forwards, 128 sequences and 192 extra local attention contractions.

A material interaction therefore exists already in these native readers.
This does not establish that it explains the whole final-output interaction.
The joint router could receive a previously computed joint-command state,
or generate its interaction from separate command features downstream.
The [next fixed-boundary experiment](../BILIN18_UPSTREAM_MIXED_COMMAND_STATE_V1_PREREGISTRATION.md)
removes only the mixed residual-state command mode after block8, preserving
mean/single-command modes and both shared first-value and embedding inputs.
It tests whether one upstream component mediates the four readers and whether
its output effect is selective. This cross-cell internal intervention is
explicitly priced; it is not yet an independently identified semantic variable.
Six algebra controls pass; native integration is pending.
