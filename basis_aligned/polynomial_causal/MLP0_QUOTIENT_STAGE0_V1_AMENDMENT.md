# MLP0 quotient Stage-0 v1: prospective implementation amendment

## Why a new namespace is required

The first collector attempt at `skip=17000, N=192` terminated without a result
artifact.  It is permanently classified as development-only and its evaluation
window is treated as exposed.  Before failing, independent source audit found four
outcome-independent implementation defects:

1. it used `KL(p_O || p_T)` for the token-table/live contrast instead of the
   registered `KL(p_T || p_O)`;
2. it binned `||MLP0-mu_token||` instead of the registered raw pre-MLP0 residual
   norm;
3. it silently excluded positions 0--63, so it did not implement the full 16-cell
   grid;
4. it reshaped 50,304-column padded logits as if they had 50,257 columns and
   crashed before scoring.

The attempt also lacked the separate collector authority required by
`MLP0_CAUSAL_QUOTIENT_SPEC.md`.  No scientific arm result, bootstrap statistic, or
table comparison was produced, so these repairs do not condition on an outcome.

## Frozen v1 changes

- The fit construction remains `N=960, skip=80`, exactly matching
  `mlp0_downstream_clusters.py`.
- The prospective evaluation window is changed once to `N=192, skip=21000`.
- Rows must be materialized from the pinned local FineWeb parquet into the versioned
  v1 namespace before any v1 model forward.  The receipt must prove document,
  full-row, and prefix-32 disjointness from fit and every receipt-backed prior role.
- The fourth cell axis is the norm of the raw residual immediately before MLP0's
  RMS normalization: block-0 lambda-mixed stream plus block-0 attention output.
- All positions 0--255 are scored; halves are 0--127 and 128--255.
- The T/O KL and signed CE effects have independent orientation.  KL is always
  `KL(p_reference || p_candidate)` with T as reference; CE harm is `CE_T-CE_O`.
- Logit reshaping uses the runtime padded vocabulary width; token lookup tables remain
  GPT-2's 50,257 reachable input IDs.
- V1 writes a new atomic result path and preserves per-document x cell sums/counts so
  every simultaneous bootstrap gate can be independently recomputed.
- A collector authority must bind source, row receipt, model checkpoint/config, table
  and assignment hashes, scales, output path, and exclusive lock before scoring.

All margins, K=64 constructions, bootstrap seed/count, minimum support, coverage,
and required gate logic remain unchanged.  A v1 pass still licenses only the finite
global-deployment screen, never arbitrary-background interchangeability.
