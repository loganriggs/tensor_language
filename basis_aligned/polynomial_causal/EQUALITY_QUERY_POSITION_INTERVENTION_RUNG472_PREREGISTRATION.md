# Rung 472 preregistration: exact query-position intervention

Registered after rung471 and before opening any query-position or complement intervention outcome.

## Decision

Rung471's target-specific first-order response predicts exact per-token MLP-removal effects much better than visible
context variables, and the query position has the largest absolute contribution in every MLP, source, and register.
However, signed regional averages and source agreement do not transfer. This rung asks whether the query-position
piece is an exact causal subcircuit or only looks large because the linear response contains cancellation.

## Fixed examples and interventions

- Reuse exactly rung471's first-two-positive-targets-per-document coordinates in code validation and both natural
  waves. Code discovery is used only to report the already-frozen first-order expectation; no new fit is allowed.
- Sources: native matcher `N` and frozen transplanted matcher `H`.
- For each selected target at query position `q`, capture the equality-absent product vectors for MLP8/9/12. In the
  source trajectory, replace products by those absent values at:
  1. `query`: position `q` only;
  2. `non_query_prefix`: every position before `q`;
  3. `full_prefix`: every position through `q`.
  Positions after `q` are never patched. Later layers recompute after every intervention.
- Run `query` separately for MLP8, MLP9, and MLP12 and jointly for their union. Run `non_query_prefix` and
  `full_prefix` for the union. The union interaction at query is `union query - sum(individual query)`; the spatial
  interaction is `full_prefix - query union - non_query_prefix`.
- Score exact CE change at the selected positive target and the mean change on the fixed off-target tokens in the same
  documents. No target outcome selects a coordinate or arm.
- Zero deployed saving/addition; no rank, product-index selection, new data role, or SEALED attention0 result.

## Registered predictions

### A. Instrument

Hashes, coordinates, supports, source scales, and arm counts match. Native replay relative-squared error is at most
`1e-12`; MLP factor reconstruction is at most `1e-10`; an empty position mask is exact; `full_prefix` per-target CE
matches rung470's all-position union removal to `1e-9` nat; every position patch fires exactly; SEALED remains closed.

### B. Query position is an exact causal component

Under both sources on code validation and both natural waves, query-union effects predict full-prefix per-token effects
at Pearson at least `.55`. Their four context means have cosine at least `.80` and projection onto full-prefix between
`.25` and `1.75`.

### C. Query is more informative than the rest of the causal prefix

In every window/source, query-union has at least `.15` higher Pearson with full-prefix than non-query-prefix, or at
least `20%` lower RMSE. This comparison uses no fitted scale.

### D. Cross-MLP composition is resolved at query position

At least two individual MLP query interventions have positive Pearson with their complete-MLP per-token effects in
every window/source. The query union-minus-singletons interaction must have the same sign under N/H within each
window and either norm below `.003` nat or cross-source four-cell cosine at least `.70`.

### E. Selectivity

For query-union, the absolute off-target CE change is at most half the absolute selected-target mean and at most `.01`
nat in every window/source. Both fixed halves of each window must preserve the target-versus-off-target ordering.

## Strong null and route

The strong null is true if A fails, if query-union is no more predictive than non-query-prefix on held-out code, or if
no natural window/source has a positive query/full relationship. A full pass identifies an exact query-position
subcircuit and licenses a within-query, downstream-defined split. If query is causal but source/register-specific,
retain separate operating regimes rather than forcing one shared component. If exact query intervention fails despite
the first-order dominance, the kernel is a predictive diagnostic only and the next object must model nonlinear
query/context interaction.
