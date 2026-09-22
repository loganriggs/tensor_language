# Native optimizer comparison: Adam wins this test, but no method recovers the components

22 September 2026, 06:02 UTC. All six native fits and their larger-panel evaluations are complete.

**L-BFGS does not reproduce its toy advantage on this native problem.** It improves the smaller-output errors by about 4–5%, compared with Adam's roughly 7%; Muon changes them very little. Every optimizer fails the registered 15% improvement requirement. This is a comparison of specific configurations and budgets, not a general optimizer ranking or proof that the function class cannot work.

## What is held fixed?

The parent is the CP512 approximation to the pure quartic contribution through the last two MLPs, projected onto 16 fixed output directions. At each token its input is the 1,152-dimensional normalized MLP16 state. The experiment adds eight learned quartic products to each smaller output, coordinates 4–15, and leaves outputs 0–3 unchanged. These coordinates are not established semantic features.

All methods use the same two initial seeds, 96 new atoms, exact covariance-Gaussian weight-contraction objective, output weights, normalized forward factors and analytically fitted linear coefficients with ridge 1e-6. No text labels enter these fits. The Gaussian mean and covariance come from calibration data. Other residual, attention, bias and cross terms lie outside the selected polynomial; native normalization and softcap are not replaced by it.

The added capacity costs 288 products and 442,464 coefficients. It is a discovery experiment, not a smaller final circuit. The full third-order tensor objective and broader native paths remain separate requirements.

## Completed results on the opened larger panel

Small-output error is the RMS of twelve per-output relative errors. Responses are output differences between 2,494 fixed matched-state pairs; they are not full-model interventions. The value panel contains 16,384 states from 256 previously inspected documents.

| Method / start | Small-output value error | Small-output response error |
| --- | ---: | ---: |
| Parent |56.58%|58.09%|
| Adam 25001 |52.59%|54.08%|
| Adam 25002 |52.54%|54.13%|
| Muon 25001 |56.58%|58.08%|
| Muon 25002 |56.57%|58.07%|
| L-BFGS 25001 |53.99%|55.47%|
| L-BFGS 25002 |54.10%|55.65%|

L-BFGS ran 250 quasi-Newton iterations, taking 254 and 255 objective/gradient evaluations. It reached the iteration budget, not the hard 500-evaluation cap, and did not stop immediately. Its first-251-evaluation checkpoints also fail the original-panel component criterion. Final objectives were -0.353 and -0.319 versus Adam's selected -0.512 and -0.518; more negative is better for this common profiled objective. Because a constant teacher-energy term is omitted, percentage objective gain is not percentage reconstruction gain.

The two L-BFGS fits took 228.1 seconds and used about 2.32 GB peak allocated GPU memory. The two Adam fits took about 224 seconds within their four-arm run, whose peak was about 1.96 GB. These are observed settings, not identical optimizer-state or evaluation budgets. All L-BFGS integrity checks pass: normal-equation residual below 6e-13, exported correction drift below 1.6e-6, and CPU replay of original evaluation metrics below 2e-10 absolute. Outputs 0–3 stay unchanged.

## Could better readout coefficients or sharing rescue the learned features?

We gave each new feature bank the evaluation answers as an explicit oracle diagnostic. Keep the parent and every learned input direction fixed, then compare:

1. Each smaller output may use only its own eight new products.
2. Each smaller output may reuse all 96 new products.

The second option adds 1,056 readout coefficients, with no additional quartic products. We solve separately for values and matched-state differences; these two optimal readouts are not one model. They are not exported or presented as generalizing predictions.

| New feature bank | Local oracle: values | Shared oracle: values | Local oracle: responses | Shared oracle: responses |
| --- | ---: | ---: | ---: | ---: |
| Adam 25001 |49.99%|44.96%|52.62%|46.66%|
| Adam 25002 |49.82%|45.20%|52.54%|46.91%|
| Muon 25001 |55.97%|52.68%|57.46%|53.27%|
| Muon 25002 |56.14%|52.77%|57.77%|53.92%|
| L-BFGS 25001 |52.18%|46.26%|53.70%|47.59%|
| L-BFGS 25002 |52.40%|46.40%|54.24%|47.68%|

Sharing helps, but the prediction of at least 15% improvement over the local oracle for both Adam starts and both metrics fails. Even the best shared readouts leave large errors. This is a finite-panel limitation of these fixed new-feature banks with the parent frozen; it does not bound a jointly refitted parent, different directions, Tucker/HT or a broader DAG.

Column scaling only improves numerical conditioning and does not change the fitted spans. Float64 SVD projections are checked against independent QR residual energies and residual orthogonality. The complete receipt records rank and conditioning for every solve. No new fit used the evaluation labels for deployment.

## Research consequence

The toy optimizer results were useful controls, but did not predict the native ranking. On this native target, neither the tested optimizer substitution nor readout sharing repairs component fidelity. Further fixed-readout sweeps are poorly justified. A subsequent experiment needs a concrete reason to improve the scalar computations themselves, the native effect metric, or their initialization—not simply a more favorable aggregate score.

The normalization audit supplies a separate reason to improve the metric: raw scalar error can pass while the state-weighted native removal fails. It does not establish that normalization-aware fitting will succeed. The goal remains an extracted, reusable, selectively manipulable, OOD-predictive simple circuit; none of the six fits meets it.

[Native L-BFGS receipt](../../direct_tensor_match/NATIVE_LOCAL_LBFGS_V1.json) · [Larger-panel L-BFGS results](../../direct_tensor_match/NATIVE_LOCAL_LBFGS_FOLLOWUP_V1.json) · [Oracle plan](../../direct_tensor_match/LOCAL_FEATURE_READOUT_ORACLE_PLAN_V1.md) · [All oracle results](../../direct_tensor_match/LOCAL_FEATURE_READOUT_ORACLE_V1.json) · [Native Adam/Muon report](research_update_2026-09-22_0555_native_residual_learning.md) · [Normalization audit](research_update_2026-09-22_0559_normalization_error.md).
