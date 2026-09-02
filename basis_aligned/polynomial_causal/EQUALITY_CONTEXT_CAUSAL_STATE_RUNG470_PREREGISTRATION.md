# Rung 470 preregistration: continuous context predicts the exact MLP correction

Registered after rung469 and before opening any rung470 per-token removal outcome.

## Question

Rung469 showed that separately averaging the quadratic downstream reader and equality-induced MLP state loses the
important pairing between them. This rung keeps the actual causal effect paired and asks whether a small, explicit
function of repeat context predicts it.

The intended quotient is operational: two positive-copy tokens are treated as the same state when their repeat
distance, number of previous occurrences, and query position predict the same signed effect from removing the
MLP8/9/12 equality correction. This is useful only if the rule transfers to held-out code and natural text.

## Fixed data, interventions, and targets

- Roles and windows are unchanged from rung469: code0:96 for fitting, code96:192 for held-out validation, and the two
  fixed natural waves0:96 and96:192 for cross-register validation.
- Sources are the native matcher `N` and frozen transplanted matcher `H`.
- For every token in `all_positive`, record the exact per-token CE change after replacing all4,608 products in MLP8,
  MLP9, or MLP12, or all three together, by their same-document equality-absent products. Later layers recompute.
- The fifth target is the exact interaction: union effect minus the sum of the three individual effects.
- No gradient, target-window outcome, token identity, product index, rank, new data role, or SEALED attention0 result
  enters a prediction. Literal deployed saving and addition are both zero.

## Fixed predictors

For a positive-copy token at query position `q`, let `d` be the distance to its most recent previous occurrence and
`n` the number of previous occurrences. The eight continuous features are frozen as

`[1, log(1+d), log(1+d)^2, log(1+n), log(1+n)^2, log(1+d)log(1+n), q/256, (q/256)^2]`.

Each nonconstant column is standardized using code-discovery means and standard deviations only. Fit one ridge
regression per source and target with fixed penalty `1e-3` on non-intercept coefficients. No penalty or feature sweep
is permitted.

The matched baseline has four code-discovery means for the fixed Cartesian cells
`near/one`, `near/multiple`, `far/one`, `far/multiple`, where near means `d<=16`. The constant baseline is also
reported. Target windows cannot alter either baseline.

For cross-MLP grouping, divide each MLP's code-discovery target by its root-mean-square size, pool MLP8/9/12, and fit
one shared eight-feature coefficient vector per source. Freeze the three discovery RMS scales. Compare this shared
rule with the three separately fitted rules on every validation window.

## Registered predictions

### A. Instrument

Frozen hashes, rows, scales, masks, and support match; native/replay relative-squared error is at most `1e-12`; MLP
factor reconstruction is at most `1e-10`; empty patches are exact; every registered source and full-product patch
fires; expected forward count matches; SEALED remains closed. Reaggregating saved per-token effects must reproduce
rung467 code-validation and rung468 natural complete-MLP four-cell vectors to `1e-9` nat.

### B. Held-out code prediction

For the union under both sources on code validation, the continuous rule must have Pearson correlation at least `.30`,
reduce root-mean-square prediction error by at least `15%` relative to the fixed four-cell baseline, and reproduce the
four context-cell means at cosine at least `.85` with projection between `.50` and `1.50`.

### C. Natural-text prediction

For the union under both sources in both natural waves, the code-frozen continuous rule must have Pearson correlation
at least `.20`, reduce root-mean-square error by at least `10%` relative to the fixed four-cell baseline, and reproduce
the four context-cell means at cosine at least `.80` with projection between `.25` and `1.75`.

### D. Shared cross-MLP context law

At least two of MLP8/9/12 must, under both sources and in every validation window, have shared-rule correlation at
least `.20`; the shared rule's root-mean-square error may be at most `15%` worse than that MLP's separate rule and must
beat its fixed four-cell baseline. The same module pair must qualify in code validation and both natural waves.

### E. Predictable composition

For the exact union-minus-singletons interaction under both sources, the continuous rule must reproduce the held-out
code four-cell vector at cosine at least `.75`. On each natural wave it must either reproduce the interaction at cosine
at least `.65` with the correct sign of its projection, or correctly predict an interaction norm below `.003` nat.

## Strong null and route

The strong null is true if A fails, if B fails under either source, or if the continuous rule does not beat the fixed
four-cell baseline on any natural source/wave. A full pass identifies a simple, code-trained context law for the exact
causal correction and its composition. A code-only pass makes the law register-specific. If even held-out code fails,
repeat distance/count/position are insufficient state variables; the next object must add a measured downstream-use
variable or a paired response kernel, not rank or product-coordinate tuning.
