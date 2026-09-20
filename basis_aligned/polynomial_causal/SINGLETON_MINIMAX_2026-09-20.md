# Per-input extraction check: shared minimax features

The previous grouped derivative-only minimax fit passed opened construction-heldout contexts but failed calibration. Its coefficients depended on other examples in the group. We kept the same six atoms, source ports,18 amplitude settings and four outputs, and tested genuinely per-input coefficient generation.

For a singleton, the residual norm becomes absolute value. The minimax fit is therefore exactly a linear program with inequalities D c - y <= t and -D c + y <= t. The original SLSQP path missed the1e-7 primal-dual gap check slightly despite reporting success. The LP replacement retains the same mathematical problem and prediction gates. Independently checked primal residuals, dual objective and stationarity validate the solve.

| Fitter | Calibration number | Opened heldout number | Heldout modal / number budget |
|---|---:|---:|---:|
|Original grouped minimax|12.194%|9.680%|0.886%|
|Per-input, per-input budgets|26.977%|37.219%|6.387%|
|Per-input, original group budgets|15.359%|10.104%|1.037%|

All three fail the overall10%number/5%modal conjunction. No rounded borderline pass:10.104% exceeds10%. The group-budget control shows that changing normalization is a major confound in the singleton comparison; it does not demonstrate that peer-dependent optimization alone causes the large regression. The control still needs group-derived budgets and is not an independent per-input generator.

A separate native-derivative fixture checks the true singleton implementation under batch permutation and removal of the other example. Coefficients remain identical (maximum differences zero). Thus its peer independence is implemented correctly, but it does not preserve the desired behavioral accuracy.

The conditional storage remains44 values/example plus10 shared plane values and6 shared pair indices. Full native source and derivative computation remains required. Dictionary discovery and fitting-rule selection used opened data; no prospective OOD, selective-removal or reusable-circuit claim follows. This closes the proposed six-atom per-input route under the tested normalization rules; it does not establish an impossibility theorem for all dictionaries or objectives.

Receipts: SINGLETON_DERIVATIVE_MINIMAX_V1_RESULT.json, SINGLETON_GROUP_BUDGET_MINIMAX_V1_RESULT.json and SINGLETON_MINIMAX_INDEPENDENCE_RESULT.json. The shared evaluator now accepts an explicit fitter and output path, avoiding another duplicated data loader/scorer. Original grouped results are preserved. Further changes must address the feature dictionary or finite response, rather than treating the grouped partial success as a standalone extracted circuit.
