# More calibration reduces the empirical-metric transfer gap

21 September 2026, 09:34 UTC. Follow-up to [the empirical readout experiment](research_update_2026-09-21_0926_empirical_polynomial_metric_readout.md).

**Using the larger existing calibration pool improves the same graph's transfer error, but still does not meet the full accuracy requirement.** The primary now passes the15%absolute third-component cap. It fails the stricter requirement to stay within10%of the separate baseline. No candidate is adopted.

## A controlled change in coverage

We repeated the source-moment readout fit using16,384states instead of1,536. The additional232prefixes were already captured for covariance estimation; they are distinct from all32original calibration/evaluation prefixes. Together with the original24fitting prefixes, this makes256prefixes of64tokens each.

The graph, feature directions, affine-centering convention, coefficient penalty and five empirical weights were unchanged. The output coefficients were again solved exactly. All six quadratic source reads remain targets. There are still399source products and896,198floating coefficients.

This comparison uses the **source-read loss**, not the downstream-sensitivity loss: the larger cache does not retain the later inputs needed to construct that loss for components1and2. We did not invent or substitute those inputs.

## Completed fixed-direction comparison

Percentages below are third-component errors on the same448previously opened states, not fresh text.

| Empirical source weight | Fit on1,536states | Fit on16,384states | Coefficient error with larger pool |
|---|---:|---:|---:|
|0,coefficient-only control|16.09%|16.09%|8.63%|
|0.01|16.05%|16.05%|8.63%|
|0.1|15.83%|15.80%|8.64%|
|1,registered primary|15.41%|14.91%|8.82%|
|10|16.42%|14.43%|9.43%|

The primary's full opened component errors are **2.14%,2.47%,14.91%**. Its coefficient error remains within1.10times the coefficient-only control, so that guard passes. The component-value gate still fails: the separate third-component baseline is11.94%, requiring at most13.14%under the relative rule.

The weight10control no longer shows the same transfer deterioration as its small-pool counterpart. It also remains within the coefficient guard, but still fails the relative component requirement. This is evidence that calibration coverage matters for this objective, not permission to select that control as a passing result.

The old1,536-site fitting diagnostic for the new primary is14.42%third-component error, closer to its14.91%opened error than the earlier sensitivity-weighted fit. Those old states are only a subset of the new fitting pool; this number is not the new full-pool training error.

Both panels are historical calibration diagnostics without retained document identities. Increasing the number of prefixes does not establish that they are independent documents. No new native model forward or fresh intervention test was used.

## Next comparison: allow the feature directions to adapt

We have implemented and registered full-batch variable projection for this objective: Adam updates the shared and private input directions, while exact constrained solves update the output coefficients at every step. This directly completes the continuous direction-fitting step under the richer metric.

The planned managed run compares empirical weights0,1and10, with two identical-seed1%perturbation starts per setting,1,000cosine steps and learning rate0.005. All settings use the full16,384-state pool; weight1is primary. They retain the same399-product graph, all three component requirements and the coefficient-fidelity comparison against the equally continued weight-zero control.

Five synthetic loss, gradient and envelope-derivative checks pass below2e-15. A native-width preflight with128states agrees with dense evaluation below4e-16and has finite direction gradients. These verify the fitting instrument; they do not predict success of the native run. No direction-fit outcome is claimed in this report.

## Evidence

- [Expanded fixed-direction results](../../direct_tensor_match/EXPANDED_EMPIRICAL_READOUT_V1.json).
- [Variable-projection checks](../../direct_tensor_match/EMPIRICAL_SOURCE_VARPRO_PREFLIGHT_V1.json) and [native-width preflight](../../direct_tensor_match/EMPIRICAL_SOURCE_NATIVE_PREFLIGHT_V1.json).
- [Registered direction comparison](../../direct_tensor_match/EMPIRICAL_SOURCE_DIRECTIONS_PLAN_V1.json).

Stable semantic features, fresh/OOD fidelity, standalone extraction and selective reusable circuits remain unproved.
