# Derivative-only minimax: partial prediction success, overall failure

The six-atom dictionary and support remain frozen from the shared source-plane experiment. For each family group, all four output coefficient sets are chosen solely from native analytic gradients and Hessians, minimizing the worst discrepancy from the full quadratic across18 predefined amplitude settings. Error budgets are the full quadratic's predicted number-effect norms. Native finite outcomes are loaded only after all coefficients are frozen. Poisoning those loaded outcomes leaves the coefficient hash unchanged.

Construction-heldout, already opened contexts pass: worst number9.680%, modal0.886% of native number budget. Calibration contexts fail number12.194%; modal2.868% passes. The all-cell gate therefore fails. Dictionary support and fitting-method development used earlier opened results, so this is not prospective OOD evidence despite construction exclusion when the plane was fitted.

A SLSQP line-search warning initially stopped evaluation. The corrected acceptance test independently constructs a feasible dual witness and checks primal-dual gap below1e-7 and stationarity below1e-10. Optimizer status is recorded but is not treated as a mathematical certificate. Prediction thresholds are unchanged. The receipt stores every solver diagnostic.

## Why the worst cell fails

Congruent-attractor, near_greeted|plural, unitB:

- Full quadratic versus native:8.938% number error.
- Dictionary approximation versus full quadratic:6.037%.
- Combined error:12.194%.
- Cosine between these error vectors:0.300.

The exact vector identity is independently checked: total error = full-quadratic remainder + dictionary approximation. Neither component alone exceeds10%, but they partially reinforce. This explains why fitting the analytic quadratic well does not guarantee finite-native fidelity, and why the outcome-fitting oracle could pass while the derivative-only fit fails. It does not establish that either error source alone is the unique cause.

The representation uses44 conditional values per example plus10 shared plane values and6 shared pair indices, versus80 dense quadratic values. Producers still require full native source directions, Jacobians/readers and Hessians. Furthermore, minimax coefficients depend on the other examples grouped into the same family: this is an explicit limitation for standalone per-input extraction and deployment, not a free invariant dictionary. No selective intervention or reusable complete circuit is established.

Receipts: DERIVATIVE_ONLY_MINIMAX_V1_RESULT.json and DERIVATIVE_ONLY_MINIMAX_ERROR_AUDIT.json. Executors: derivative_only_minimax.py and audit_derivative_only_minimax.py. The prerequisite oracle fit remains outcome-dependent and cannot be substituted for this predictor.

The next useful distinction is per-input stability versus batch-dependent fitting. Test the same frozen dictionary with singleton coefficient selection, then compare prediction and coefficient changes. A robust circuit should not rely on arbitrary unrelated examples to define its computation. Fresh native-context validation remains required after the coefficient-generation rule is frozen.
