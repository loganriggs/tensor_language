# Rung 508 instrument repair: explicit float32 family-summation remainder

Status: frozen after the first managed CUDA smoke retained only numerical diagnostics, before any family task effect,
selection, confirmation, or composition outcome was retained.

The smoke passed exact calls, replay, factorization, family coverage, and liveness for all21 family terms,
`FULL_NAMED`, and one joint patch. Its only failed check was the relative squared difference between:

1. summing21 family outputs after applying `Down` to each separately; and
2. applying `Down` once to the sum of the21 family hidden products.

The observed discrepancy was `8.78394e-10`, above the registered `1e-10` limit. The hidden family products themselves
sum to the complete named product with relative squared error `1.04037e-14`. The output discrepancy is therefore
float32 summation order, amplified by taking a small difference between score conditions, not a missing source or
overlapping family.

Preserve that failed pre-repair number. For each condition `a`, explicitly store the nonselectable arithmetic
remainder

`eta_a = semantic_named_output_a - sum_over_21(separately_projected_family_output_a)`.

The repaired score-change identity is

`semantic_change = sum_over_21(family_change) + (eta_a - eta_absent)`.

Require its relative squared closure error to be at most`1e-12`. `eta` cannot enter a family term, selector,
singleton removal, joint removal, same-output label, or composition fit. `FULL_NAMED` remains exactly the sum of the
21 intervention terms and therefore also excludes `eta`. No scientific arm, document, threshold, or route changes.
Only a passing repaired no-outcome smoke may open the rung508 science run.
