# The root now has an exact function metric

2026-09-20 21:02 UTC

The four quadratic bank features feed ten pairwise products at the root.
Their covariance is now computed exactly under the declared Gaussian input
law, using input moments through degree eight. The bank features themselves
are not assumed Gaussian. A small independent quadrature check agrees to
1.1e-15, and the calculation uses the actual saved FP32 bank parameters.

The resulting 10×10 covariance matrix has condition number362.3. The output
frame's small deviation from orthogonality is explicitly included. An output
rank4 relaxation can retain at most99.778% of centered root energy. This is
an upper bound, not proof that four products attain it.

The next registered comparison fits4/6/8root products with Adam/Muon, two rates
and two seeds. Four root products would bring the full six-product-bank program
to10products and24280coefficients. Its acceptance bar includes complete quartic
prediction and literal graph replay, not just the small root loss. The fitting
objective and gradients are implemented and checked; native-root optimization
has not yet run.

The hourly review records the broader result: choosing the correct function
metric has mattered more than repeatedly changing ranks. Prediction and
computational sharing improved, but primitive identity and semantic selectivity
remain unresolved. Subsequent refinements reuse the earlier validation panels,
so they are diagnostics now, not new untouched evaluation.

[Exact moment check](../../direct_tensor_match/ROOT_FUNCTION_METRIC_V1.json),
[saved-program metric](../../direct_tensor_match/ROOT_ARCHIVE_METRIC_V1.json),
[next refactor](../../direct_tensor_match/ROOT_PRODUCT_REFACTOR_PLAN_V1.md),
[hourly review](../../HOURLY_STRATEGIC_REVIEW_2026-09-20_2101.md).
