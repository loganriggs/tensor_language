# Composition of two joins at one late binding

After invariant gates, endpoint paths and physical removal transport failed,
test a context-dependent nonlinear composition rule. Do not assume amplitudes
are invariant. For a->b->c->d with binding b->c later than both adjacent
bindings, L2H1 writes the forward join b->d from c->d, while L2H2 writes the
backward join a->c from a->b. Their destination supports overlap. Query a/hop3
uses b->d; query F^-1(a)/hop3 uses a->c. Both queries read the same document.

Use the existing middle-match base worlds, seeds14909/14910, exactly the two
orders where B2 is later than B1 and B3. All32 opened worlds, two orders, two
query origins, four hops =512 requests. Generate answers from actual tokens
with the shared fork helper. No capability-based exclusion or new holdout claim.
Writes are the complete fixed native four-cell L2H1/H2 messages, not expanded
endpoint embedding fields or fitted entity gains. Their source cells differ;
their two destination positions coincide. Native supports/heads fixed a priori.

Let native source x include writes u,v. For each remaining reader, numerator
P(x) is cubic and source gain is g(x)^3, with g the actual affine-free RMS.
Retain the complete single-write numerator curves

    Pu(a)=P(x+(a-1)u), Pv(b)=P(x+(b-1)v).

The candidate joint read at x' = x+(a-1)u+(b-1)v is

    background + sum_sources g(x')^3 [Pu(a)+Pv(b)-P(x)].

This keeps all univariate degrees and exact shared normalization; only mixed
numerator terms around native x are omitted. It is exact on both axes a=1 or
b=1. Whether the remaining joint interaction is mostly shared normalization is
the scientific question. This does not repair the failed single-write degree
truncations. All native prefix/reader weights remain charged; a pass identifies
a conditional composition rule, not a smaller whole-model program.

A instrument: exact executor agrees with independent native pattern cuts plus
scaled write restoration for every (a,b) in {0,.5,1}², abs<=1e-9 and relative
<=1e-10. Candidate axes obey the same bars. Same destination support and write
reuse across all eight queries checked. Planted separable-numerator control and
live mixed-term negative establish the approximation's meaning.

B native causal eligibility: per population/query origin, native hop3 accuracy
>=.9; removing its designated join lowers its mean gold probability >=.5;
the other query origin's mean absolute gold-probability change <=.05. Report
all other hops and both removals, with no eligibility filtering. Failed
selectivity means these writes are not independently selective circuit units.

C composition: for every population/query/hop group and each a,b in {0,.5},
candidate joint error divided by the actual centered-logit interaction RMS
of L(a,b)-L(a,1)-L(1,b)+L(1,1) must be <=.01, floor1e-6. Also report joint
teacher mean/p99/max query KL and error relative to the full joint effect;
those cannot replace the primary interaction bar. No head, scale, field or
task subset rescue. If C fails, mixed numerator coupling is required.

Reuse exact_source_edit_reference, join_contribution_context_reference,
join_write_read_degree_reference, query_hop_fork_reference and the shared
field scorer. CPU threads2, batch8 FP64, alarm180s, tensors below256MiB.
No GPU is needed for this bounded screen unless the measured CPU cap is hit;
any GPU successor must use the managed queue. Preserve native checkpoint and
all existing nulls. Expose background and all387968 export coefficients.
