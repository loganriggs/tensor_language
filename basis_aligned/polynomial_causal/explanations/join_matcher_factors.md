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
