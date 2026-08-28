# Compilation-mask cut-rank v1 scoring amendment

Date frozen: 2026-08-28

Status: post-measurement, pre-outcome scoring completion.  The sealed measurement
receipt and tensor schemas/counts were available when this amendment was written,
but no cell value, development selection, heldout metric, or bootstrap result was
read.  This amendment cannot alter the mask registry, split, fitted families,
point-estimate gates, or scientific claim in the original preregistration.

## Bootstrap completion

Use 2,000 shared source-document cluster-bootstrap draws with replacement and seed
`2026082851`.  Every draw samples exactly the observed number of documents and uses
the same document multiplicities for all 64 cells and both target currencies.
Aggregate correct counts, CE sums, and scored-token denominators before forming any
accuracy, CE, B0 contrast, anchored interaction, ratio, R-squared, NRE, or spectral
tail.

The original-data development split selects rank, ridge, and baseline family once.
Every bootstrap draw refits that fixed selected rank/ridge pipeline, including its
train-only RMS, with the exact registered eight-restart ALS algorithm.  It also
refits the fixed selected baseline family and its fixed ridge, where applicable.
It never reselects on bootstrap validation or heldout values.

Registered one-sided 95% bounds use literal sorted order statistics without
interpolation: rank `ceil(0.05 * 2000) = 100` for a lower bound and rank
`ceil(0.95 * 2000) = 1900` for an upper bound, both one-indexed.  A missing,
non-finite, or zero-denominator replicate makes the corresponding gate fail closed;
draws may not be discarded.

## Historical singleton currency and fail-closed limitation

The committed S1834 source
`basis_aligned/bilinear_quotient/ops/site_cost_table_results.json` contains the 34
top-1 singleton costs as fractions of the historical live-minus-fully-compiled
gap.  For the registered raw-percentage-point top-1 singleton baselines, multiply
those exact stored fractions by the prospectively fixed historical endpoint stake
`100 * (0.3932 - 0.1355) = 25.77` percentage points.  Bind the exact source bytes,
the two endpoint constants, the conversion, and the resulting 34 values in the
development/result receipt.  These historical numbers receive baseline credit
only, never fit-target or bootstrap-outcome credit.

No already sealed source reconstructs the corresponding 34 CE singleton costs.
Reusing top-1 costs in CE units or deriving them from the current 8x8 outcomes is
forbidden.  The scorer therefore reports CE point and bootstrap metrics against
the executable additive/count-depth baselines, but marks the registered CE
baseline family incomplete.  Consequently the original eight-condition useful
pass is **unevaluable and non-promotive**, regardless of all numerical values.
Individual registered predicates must still be reported truthfully; they cannot be
combined into `true` until a prospectively authorized independent CE-singleton
source exists.

## One-shot boundary

The scorer must bind committed scorer/core/amendment bytes, the exact sealed
measurement receipt and payload hashes, and all constants above in a create-only
authority before loading outcome tensors.  It then publishes a development-only
selection receipt before its single heldout finalization attempt.  Result and
receipt publication are atomic/create-only; any post-authority error leaves a
non-promotive failure receipt and cannot reuse that output namespace.
