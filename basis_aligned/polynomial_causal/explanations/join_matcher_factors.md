# What do the two factors of a contextual join matcher compute?

Earlier causal tests identify L2H1 as the forward join writer and L2H2 as the
primary backward writer. Their complete product-attention scores respond to
middle-entity matching: relabeling only keys or only values breaks the route;
relabeling both restores it. That evidence does not assign separate semantic
operations to the two native dot products. The
[existing join dossier](cold_query_composition_reconstruction.md) contains
the removal, source-port and held-case evidence, as well as the failed whole-head
semantic kernel replacement. That replacement remains closed.

## A complete shared matcher is ruled out on the unrestricted state domain

At fixed query/source positions, write each normalized dot product as
`f_j(u,v) = u^T A_j v`, where
`A_j = Wq_j^T Rquery^T Rsource Wk_j /32`.
Reusing the first native function as the second, up to a scalar, requires
`A2 = alpha A1` on arbitrary normalized state inputs. This is a necessary
coefficient identity, independent of a factor's Q/K coordinate gauge.

[JOIN_MATCHER_COMMON_FACTOR_V1](../JOIN_MATCHER_COMMON_FACTOR_V1.json)
checks the two previously nominated heads at query position3 and source1,
both binding-value positions. It compares the full128-by-128 forms without
choosing another head, position or rank. Native product correspondence on
deterministic normalized-state fixtures is1.39e-16. Proportional/independent
controls pass.

The best scalar residual is .999990 for H1 and .999960 for H2. Each pair has
a nonzero2-by-2 coefficient minor computed exactly as a rational number from
the stored FP64 folded forms; exact hexadecimal inputs and numerator/denominator
are in the receipt. Thus duplicating one complete matcher cannot explain the
other stored form. No weight savings or behavioral approximation follows.

This does not rule out sharing on a smaller reachable or intervention domain.
The certificate concerns stored folded coefficients; correspondence to native
operations is numerical, not interval-certified. It is not a lower bound on
teacher KL or on every possible factorization of the whole circuit.

## The next semantic question is restricted to actual middle-label interventions

[MATCHER_FACTOR_TRUTH_TABLE_V1](../MATCHER_FACTOR_TRUTH_TABLE_V1_PREREGISTRATION.md)
uses the opened32-world, six-order, four-case middle-match cohort. It extracts
Q1K1/32 and Q2K2/32 on all four original pair-to-pair source cells. Each factor
is separately tested for suppression under both unmatched label interventions
relative to both matched cases, with a fixed .10 RMS-ratio bar. The joint
product's known truth table is a positive-control replay.

Six CPU controls pass product reconstruction, live factors, a true predicate,
constant and dead negatives, and the requirement that both mismatches suppress
the factor. Managed trained-case integration is next. This is a factor screen,
not yet an independent causal-factor identification or a reduced executable
model. A positive nomination requires native factor interchange and fresh
confirmation; a null retains the coupled product rather than fitting a new
mixture after inspection. All opaque weights remain charged.

## Individual factors fail; selectivity belongs to their product

[MATCHER_FACTOR_TRUTH_TABLE_V1](../MATCHER_FACTOR_TRUTH_TABLE_V1_RESULT.json)
completed in .836 seconds on768 opened cases. Mechanical prediction A and
joint-score replay D pass with exact zero discrepancy. Individual-factor
predictions B/C fail. Factor1 mismatch/matched RMS ratios span .399–.800;
factor2 ratios span .087–.333, but neither passes both mismatch cases in any
complete population/orientation group. The joint ratios remain .0265–.0686.
Do not promote one near-threshold factor or silently drop source-role cells.

The [role-overlap audit](../MATCHER_FACTOR_ROLE_OVERLAP_V1.json) retains all
four destination/source key/value cells. It compares native product energy
with the exact average obtained by permuting factor2's four cells within each
row, preserving both individual row norms. The latter is
`mean_rows(mean_cells(a²) * mean_cells(b²))`; explicit enumeration of all24
permutations checks the formula. It is a diagnostic energy baseline, not a
valid native circuit intervention.

For the forward join, mismatch ratios rise from .059–.069 native to .116–.124
under this baseline. Matched responses align on the same role cells more
strongly than mismatches. Backward results are less uniform, especially OOD,
so role alignment is not a universal explanation of product selectivity.

## Interchanging one factor also fails through its actual consumer

Raw-factor norms can preserve irrelevant directions, so a second
[saved-score audit](../MATCHER_FACTOR_INTERCHANGE_SCORE_V1.json) interchanges
factor1, factor2 or both while retaining the other recipient factor. All
roles, worlds and orientations are retained. Native joint replay is exact,
and the mixed identity `(a'-a)(b'-b)` closes1.78e-15.

Neither individual transfer suppresses both mismatch cases below .10 in a
complete group; joint transfer does. This rules out the tested individual
gate interpretation even through the fixed native multiplicative consumer.
The full-factor null has not been revised into a selected-cell success.

Matched-both donors expose a composition issue. In the backward join, either
single-factor hybrid can have score RMS around2.4 times baseline (maximum2.53),
while transferring both gives .966–.979. Forward joint ratios are .960–1.030.
These are aggregate norms, not equal per-case scores or full-output predictions.
They nominate the pair as a unit that should travel together, but do not yet
establish a portable semantic matcher.

The next [pair transport experiment](../MATCHER_PAIR_TRANSPORT_V1_PREREGISTRATION.md)
will transfer the matched donor factors individually and jointly into the
recipient's selected native score cells, keeping its values and other routes
fixed. The same binding gate is reused across query hops0–3. A compiled score
change becomes an explicit L2 write delta followed by the exact final-layer
source-edit executor; the native oracle changes attention cells directly.
Thirteen controls pass all four cells, route removal, identity, live individual
routes and mixed effects, query independence and restored hooks.

Joint portability must pass full-output KL and quantitative change relative
to native route-removal vectors. If it fails, no scalar/entity fit or selected
cell/head/hop rescue follows. Native trained-case integration is unfinished.
All387968 export coefficients and background computation remain charged; no
structural reduction or completion of the four-property goal is claimed.

## Matched pair transport fails full-output fidelity

[MATCHER_PAIR_TRANSPORT_V1](../MATCHER_PAIR_TRANSPORT_V1_RESULT.json) completes
in6.79 seconds with mechanical/composition predictions A/C passing and pair
portability B failing. Full native/export output agreement is1.42e-13. The
native oracle changes score cells directly; the compiled executor applies
their residual write delta with live final-layer normalization.

On hop3, joint-transfer query KL is9.994/12.019 for the forward IID/OOD groups
and3.145/2.697 backward. Query changes relative to native route-removal RMS
are1.227/1.292 forward and .921/.788 backward. Matched aggregate score norms
did not make the gate interchangeable. No individual arm, scalar/entity fit,
selected role or selected hop is adopted as a rescue.

The [pointwise audit](../MATCHER_PAIR_POINTWISE_V1.json) retains all192 matched
source pairs and every source-role cell. There are108 value-to-value score
sign flips. Full score-vector relative changes span1.432–1.610, with cosines
between -.349 and .005 despite similar overall RMS. This explains the missing
information in the earlier aggregate comparison.

Sign-flip cases account for most transfer KL, but that observation is not a
causal sign-only explanation. Same-sign IID pairs still have mean hop3 KL .087
forward and .537 backward. Native value content and the downstream reader are
coupled to the gate; no learned sign correction follows from this census.

The next question concerns the downstream operation on the entire join write,
using the earlier independent port-restoration evidence. With one attention
layer remaining, its read has an exact cubic numerator and a live RMS factor.
The [response-curve derivation](join_write_response_curve.md) gives an executable
test of the fixed forward-linear/backward-quadratic hypotheses. Twelve CPU
controls pass; the trained-case audit remains next. Exact curve closure is
intervention infrastructure, not a structurally reduced model.


## Grouping the endpoint writer with its readers

The [degree audit](join_write_response_curve.md) rejects fixed forward-linear
and backward-quadratic reads. The next [cross-layer screen](../CROSS_LAYER_ENDPOINT_TRANSPORT_V1_RESULT.json)
uses the existing direct endpoint embedding field through L2 head1 and all four
final heads, with native routing and RMS conditioned on each matched context.
All32 opened worlds, three forward orders and four hops give384 paired
comparisons/768 executions. CPU wall time1.859 seconds; independent write and
whole-final-read oracles close7.11e-15. No head or sample was selected.

This grouping changes the sign picture. At hop3, writer vectors have negative
cosine on27/48 IID and30/48 OOD pairs. The complete endpoint paths have none;
their mean full-vector cosines are .998735/.999340. This supports conditional
sign compensation across the writer-reader boundary. It does not establish a
portable sufficient circuit: full path relative changes remain .22525/.21721,
and prediction of physical endpoint-write removal errs .36223/.36127. The
registered1% grouped-transport and physical-removal hypotheses both fail. Every
other hop group fails too; no hop3-only success is promoted.

The path freezes final keys and normalization. Physical removal updates them,
so these objects cannot be interchanged in the explanation. All387968 native
export coefficients and the token-derived background remain required. The
next CPU diagnostic will use these already saved native logits and physical
removals to determine whether changing removal effects are compensated by
remaining computation. No fitted signs, gains, heads or enlarged payload field.

A [separate metadata overlay](../QUERY_HOP_METADATA_CORRECTION_V1.json) also fixes
576 stale expected-answer fields in the earlier pair-transport rows. Their actual
answer fields were already correct; no registered pair-transport scores change.
The cross-layer runner uses the shared token-derived hop-fork helper and needs
no such overlay.
