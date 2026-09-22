# Can simple input conditions predict the remaining errors?

22 September 2026, 06:22 UTC. Completed CPU diagnostic, about 1.4 seconds; no new model fit or circuit adoption.

**Token identity predicts some error concentration, but explains little of the missing signed computation. Position does not help beyond a constant correction.** This addresses a different question from counting 16 output coordinates: can we identify observable input conditions associated with the errors?

We inspect the two completed Gaussian-Adam corrections to the pure quartic contribution through MLP16 and MLP17. Each receives the actual 1,152-dimensional normalized MLP16 state. We focus on fixed output coordinates 4–15, the twelve smaller outputs with poor reconstruction. These are projections of a selected model contribution, not native neurons or established semantic variables. Other residual terms, attention, biases, and mixed terms are excluded.

At each token, the residual is native scalar output minus candidate scalar output. We divide each output's residual by that output's native calibration RMS before analysis, so large output magnitudes cannot dominate. This differs from the unbalanced residual-energy distribution in earlier reports; percentages from those metrics are not directly interchangeable.

Using only 6,144 calibration states, we fit descriptive tables predicting that residual vector from: a constant mean; token position 0–63; current token identity; or position plus token identity. Token means receive 20 pseudo-observations at the baseline mean to reduce noise from rare tokens. Unseen token IDs receive the baseline prediction. The separate 16,384-state panel supplies evaluation labels only, with no refitting or parameter selection. It is already opened research data, not a fresh OOD test.

| Residual predictor | Squared error remaining, seed 25001 | Squared error remaining, seed 25002 |
| --- | ---: | ---: |
| No correction |100%|100%|
| Calibration constant |96.79%|96.76%|
| Position |97.01%|97.00%|
| Token identity |90.88%|90.31%|
| Position plus token |91.12%|90.56%|

These are squared-error ratios, not relative RMS errors. Token identity removes about 6.1–6.7% of error beyond the constant predictor, below the registered 20% criterion. Position slightly worsens the constant predictor and fails its 10% criterion. No table is exported as a replacement circuit.

**Predicting error magnitude is easier than predicting the residual vector.** Separate calibration tables estimate each state's standardized squared error. Rank evaluation states by this predicted risk, then measure the actual error in the highest-risk 10%:

| Risk predictor | Error share in highest-risk 10%, seed 25001 | Seed25002 |
| --- | ---: | ---: |
| Position |10.71%|10.59%|
| Token identity |30.30%|30.69%|
| Position plus token |28.11%|28.42%|

The registered prediction specified the combined model capturing at least 30% in both starts. It fails. The token-only ranking exceeds 30% descriptively, but does not rescue the failed combined-model prediction. This suggests some error concentration is predictable from current-token conditions. It does not show that the token is the causal source of error, nor that the signed missing computation has been recovered.

About 59.3% of evaluation token positions have IDs seen during calibration. Unseen IDs carry about 27.6–27.9% of standardized squared error. Coverage matters when interpreting the table, and strong token associations can reflect context confounds.

The prior Gaussian derivative audit complements this result: the largest 64 original input coordinates carried only about 6.9% of residual gradient energy. That is evidence against a tiny fixed coordinate subset under that Gaussian measure, not a proof about actual text conditions or nonlinear features. Together, the observations argue against either a handful of raw input coordinates or a simple token/position lookup being the missing circuit.

The hybrid Gaussian/native-sensitivity learner remains queued. Its frozen comparison will test new learned polynomial features; these descriptive tables do not change its fitting objective or selection rule. Full output coverage, semantic selectivity, reusable components, fresh OOD prediction, and literal simplicity remain open.

Appendix: native candidates are GaussianAdam seeds 25001/25002, fixed CP1001 parent, eight new quartic atoms per small output. Calibration is 96 documents × 64 positions; evaluation 256 × 64. Evaluation token/state hashes and candidate hashes pass. Planted token and position controls plus unseen-token fallback pass. CPU float64, two threads, no GPU work. [Protocol](../../direct_tensor_match/RESIDUAL_TOKEN_POSITION_PLAN_V1.md), [all measurements](../../direct_tensor_match/RESIDUAL_TOKEN_POSITION_V1.json), [implementation](../../direct_tensor_match/audit_residual_token_position.py), [input derivative audit](../../direct_tensor_match/RESIDUAL_INPUT_SENSITIVITY_INTERPRETATION_V1.md).
