# Rung 531 preregistration: factor-level sharing inside the equality score

**Registered:** 2026-09-03 11:58 UTC

**Owner:** Codex

**Status:** CPU screen in progress; no model outcomes inspected

## Question

Four attention heads, `L5H5`, `L7H3`, `L8H3`, and `L8H4`, have already passed causal tests showing that their
equality-related attention score can often substitute for another head's score, up to a fitted sign and scale.
That score is not primitive. For query position `i` and key position `j`, the model computes

```text
first_score_h(i,j)  = <q_h(i),  k_h(j)> / 128
second_score_h(i,j) = <q2_h(i), k2_h(j)> / 128
attention_score_h(i,j) = first_score_h(i,j) * second_score_h(i,j).
```

This rung asks whether the shared computation lies in either multiplicative factor, rather than treating an entire
head or the already-multiplied score as the semantic unit. This directly addresses cross-head grouping and
within-head splitting. It is not a rank-reduction or storage experiment.

## Frozen objects

- Heads: `L5H5`, `L7H3`, `L8H3`, `L8H4` in that order.
- Directed source-to-target pairs: all 12 ordered pairs of distinct heads.
- Data: the already-frozen natural equality corpus used by rungs 498--501, split before fitting into discovery and
  confirmation documents. Validation/OOD rows remain sealed during this screen.
- Edges: genuine equality-fetch edges selected by the frozen induction mask. Non-equality causal edges and a
  within-document key-position permutation are controls.
- For every pair, test both assignments: direct (`first -> first`, `second -> second`) and swapped
  (`first -> second`, `second -> first`). Choose the assignment using discovery only: minimize the sum of the two
  branch relative squared errors after scalar fitting, breaking an exact tie in favor of direct.

## Exact fitted computation

For a source pair of factor arrays `(a,b)` and target arrays `(c,d)`, and for either direct or swapped assignment,
fit one scalar per branch on discovery edges:

```text
alpha = <a,c> / <a,a>
beta  = <b,d> / <b,b>.
```

The held-out predictions are `c_hat = alpha*a`, `d_hat = beta*b`, and
`product_hat = c_hat*d_hat`. Report cosine and relative root-mean-square error for each branch and the product.
Also compare `alpha*beta` with the independently fitted scale from the source product `a*b` to the target product
`c*d`. This check detects an apparently good branch fit whose gauges do not reconstruct the known score relation.

No vector basis is learned. The only fitted quantities are two scalars and the discovery-only direct/swap choice.

## Predictions

### A — instrument and known product relation

All factor identities, reconstruction checks, split identities, and control permutations are exact, and the known
source-to-target product relation is recovered on confirmation edges. This must pass before interpreting factors.

### B — both multiplicative factors are shared

At least one directed pair has, on both confirmation halves:

- both factor cosines at least `0.90`;
- both factor relative errors at most `0.45`;
- reconstructed-product cosine at least `0.90` and relative error at most `0.45`;
- each factor cosine at least `0.15` above its permuted-edge control; and
- the discovery-selected direct/swap assignment agrees across both confirmation halves.

If B passes, the evidence supports grouping two heads below the head boundary: the same two factor computations are
combined, possibly in reversed order.

### C — exactly one factor is shared

At least one directed pair has exactly one factor meeting the B factor thresholds on both confirmation halves; the
other factor has cosine below `0.70` or relative error above `0.65` on at least one half. The shared factor must beat
its permutation control by `0.15` on both halves. If C passes while B fails, the heads reuse one input relation but
compose it with different companion relations.

### D — factor sharing explains the known product portability

For every pair counted by B or C, the branch-derived product must predict the held-out target product better than a
single fitted scalar times the source product by at least `0.05` relative error, or isolate one shared branch under C
while the whole product remains causally portable according to the frozen parent result. Otherwise the factor fit is
only a descriptive reparameterization of the already-known product.

### Strong null

A passes, but neither B nor C passes. Then the causally portable equality score is a product-level computation; its
particular `q-k` and `q2-k2` factorization is not stable across these heads. The next decomposition should operate on
the score function or downstream equivalence class, not retry factor matching with more rank or looser thresholds.

## Interpretation boundary and successor

This CPU rung is a **screen**. B or C cannot establish causal identification by itself. A positive result opens one
managed-GPU successor fixed before its outcomes: physically replace only the selected target factor with the scaled
source factor, retain the target's other factor and value/output path, and measure the already-calibrated equality
circuit plus unrelated-circuit and key-permutation controls on sealed rows. A null closes this factor-alignment route.
