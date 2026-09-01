# Corrected mixed104 exact-bill design — 2026-09-01 02:14 UTC

## Decision

The corrected mixed104 program has passed prediction, certificate, fresh-text, and signed a16 gates, but
it does **not** yet have an adoptable total storage price.  Historical totals combined rounded anchors with
an incremental “pattern values” ledger.  The current hook harness also executes native modules before
overwriting their outputs, so counting the loaded PyTorch model would charge tensors the deployable program
does not need.  Neither quantity is the semantic compiled-program bill.

## Three prices that must not be mixed

1. **Hook-harness footprint:** the full 546M native model plus fitted/reconstructed tensors used for
   evaluation.  This is an experimental implementation cost, not the deployed compiler claim.
2. **Historical incremental ledger:** rounded pattern/replacement components used to compare old rungs.
   It is useful provenance but cannot license a new literal total after the config-identity correction.
3. **Deployable semantic bill:** the unique stored tensors and executed operations required to compute the
   stated tensor program without first running an overwritten native operation.  This is the adoption price.

Rung 295 computes (3), while retaining (1) and (2) as separate reconciliation columns.

## First exact line item

The physical mixed104 construction has 38 unique motif heads in blocks 2–9 and all 72 heads in blocks
10–17: 110 unique replaced heads.  Each head has four Q/K projection maps.  A selected rank-104 map stores
one `128×104` left factor and one `104×1152` right factor.  Therefore

$$
N_{QK}=110\cdot4\cdot(128+1152)\cdot104=58{,}572{,}800
$$

stored scalars.  Contiguous top96 stores `54,067,200`, so the smallest-eight companion costs exactly
`4,505,600` additional QK scalars and buys `0.0038465` CE plus two certificates.

This exact physical line already exceeds the historical 52.9M mixed-pattern headline, proving that the old
rounded total must not be transported.

## Required manifest

The final manifest will name every dependency and count it once:

- shared embeddings, output map, normalizations, and tied-storage decisions;
- native attention/value/output tensors that the semantic program still calls;
- QK factors, with explicit layer/head/map/index sets;
- native block-1 value matrix and removal of the context-blind `a1v` table;
- active CP MLP tensors and any native MLP tensors retained by the semantic program;
- tables, class probes, dictionary coefficients, biases, routing state, and constants;
- temporary state separately from stored parameters;
- executed multiply/add counts separately from storage, with sequence-length dependence explicit.

Every row needs: semantic role, source operation, shape, multiplicity, scalar count, dtype/bytes, shared-key,
and a live source assertion.  Shared keys prevent double-counting tied or reused tensors.  The total must
reconcile against a standalone execution dependency graph; object traversal over the hook harness is only
a diagnostic control.

## Adoption rule

No `{price → damage, certificates}` Pareto headline is published until the deployable manifest is complete
and independently sums from its rows.  If a standalone semantic dependency cannot be specified for a hook,
that component remains “price unresolved” rather than being silently charged from an old anchor.
