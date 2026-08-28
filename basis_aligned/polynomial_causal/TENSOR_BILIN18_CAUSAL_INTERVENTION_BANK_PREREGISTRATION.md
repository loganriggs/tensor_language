# Preregistration: shared-routing causal intervention bank

Date: 2026-08-28

## Question

Does the complete shared-QK program transport context robustly across a distribution
of interventions, and if rank512 does not, is ordinary shared-routing rank alone enough
to repair it at rank640?

The opened rank512 fixture passed at 0.91485 recovery / 0.95651 cosine. A first new
fixture failed at 0.89290 / 0.94535. Neither fixture is used in this bank.

## Frozen candidates

- shared-QK rank512 plus exact dense MLPs: 503,436,726 stored values;
- shared-QK rank640 plus exact dense MLPs: 516,707,766 stored values.

Both bases are fitted only from the existing skip80 fit rows by the unchanged
activation-covariance objective. No candidate sees bank outputs during fitting.

## Frozen bank

There are 16 token-replacement interventions:

1. eight natural-prefix interventions: the first two rows of the prospectively
   hash-registered skip31000 and skip35000 cross-task roles, each changed at positions
   16 and 96;
2. eight new synthetic interventions: two affine token generators, each changed at
   positions 8, 32, 96, and 160.

Replacement offsets are fixed in source before the run. For an intervention at position
$p$, only logits at positions $p+1,\ldots,255$ enter the metric, so the current and all
downstream input tokens are identical between base and changed arms.

For native delta $d_i$ and program delta $\hat d_i$:

$$
R_i = 1 - \frac{\lVert \hat d_i-d_i\rVert_2^2}{\lVert d_i\rVert_2^2},
\qquad
C_i = \frac{\langle d_i,\hat d_i\rangle}
{\lVert d_i\rVert_2\lVert\hat d_i\rVert_2}.
$$

The distributional summary uses 10,000 fixed-seed nonparametric bootstrap resamples of
the 16 fixtures. A candidate is robustly admitted only if:

- the one-sided 95% bootstrap lower bound of mean recovery is at least 0.90;
- the one-sided 95% bootstrap lower bound of mean cosine is at least 0.95;
- at least 75% of individual fixtures simultaneously have recovery at least 0.90 and
  cosine at least 0.95;
- every fixture has nonzero native and program causal signal.

These are new distributional gates, not a retrospective relaxation of either failed
single-fixture gate.

## Predictions and decisions

- **A, fixture/authority gate:** all 16 unique fixtures are present and natural sources
  pass their registered serialized and raw tensor hashes.
- **B, ownership/price gate:** both programs are storage-disjoint, contain no native
  modules/calls/tables, have total input support, and match their exact stored-value
  prices.
- **C, rank512 robust gate:** rank512 meets all distributional thresholds.
- **D, rank640 rescue gate:** if C fails, rank640 meets all distributional thresholds.
- **E, paired capacity evidence:** rank640 improves mean recovery by at least 0.01 and
  mean cosine by at least 0.005, and does not lower recovery on more than 25% of fixtures.

If C passes, rank512 is the minimum tested robust causal point. If C fails but D and E
pass, rank640 is admitted and insufficient rank explains the immediate instability. If
both C and D fail, ordinary activation-covariance rank is pruned as the next response;
the next candidate is the preregistered causally weighted generalized-SVD basis. No
threshold is changed after the result is visible.
