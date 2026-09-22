Fixed shared quadratic producers: larger candidate-pool capacity and transfer

22 September 2026, 00:05 UTC. CPU diagnostic completed in 1.77 seconds. No GPU experiment or new candidate adoption is claimed.

The same 144 quadratic producers and 768 candidate root products used by the registered Gaussian support-exchange experiment can jointly satisfy the opened-panel root1 sensitivity and same-token response requirements when the readout is fitted to evaluation labels. This removes the earlier fixed-512 dictionary obstruction for this larger pool. It does not establish that a 512-subset can do so, that the weight/Gaussian objective selects such a subset, or that the solution transfers.

| Fit | Sensitivity error, start1101 | Sensitivity error, start1102 | Same-token response error, starts1101/1102 |
|---|---:|---:|---:|
| Calibration uniform |20.68%|23.51%|16.26% /29.92%|
| Calibration sensitivity-weighted |20.77%|22.21%|25.55% /42.63%|
| Evaluation sensitivity oracle |7.85%|8.68%|24.27% /21.72%|
| Evaluation oracle with exact cached pair responses |8.00%|8.77%|0.000355% /0.000355%|

The last row fits evaluation labels and constrains the 20 independent pair differences. Its tiny remaining response discrepancy comes from the pre-existing cached-target/native-reference difference (3.546e-6 relative), not a discovered predictive model. No oracle coefficients are exported. The unconstrained oracle alone still has poor pair responses; combining constraints matters.

The existing 512-product evaluation oracles had sensitivity minima9.70%/10.79%, and9.93%/11.01% with pair constraints. Adding products relaxes that finite numerical limit; it does not cure calibration transfer. All new least-squares designs have rank768 at rcond1e-12 after column scaling. Stationarity residuals are below8e-15; constrained residuals below5e-16. Token-hash checks and target/pair replay remain active. These are numerical least-squares diagnostics, not interval-certified bounds or population guarantees.

The full-pool program would cost1344 variable products,1357824 floating coefficients and1536 indices. The queued experiment retains only512 root products and1088 total products, with unchanged quadratic producers. Its all-pool mixed-objective score will distinguish a greedy-search gap from the available objective improvement inside this pool. Neither objective-score bound implies a bound on intervention error.

Decision: retain the registered mixed-metric support experiment unchanged. The new CPU evidence makes dictionary sufficiency plausible at the larger pool budget but shows that fitting/transfer and the joint response criterion remain material. Do not respond to a negative registered result by declaring general hierarchical or arithmetic-DAG failure. Do not use these opened-panel oracle labels in the weight-first candidate fitting.

Artifacts: [CPU results](SHARED_POOL_RESPONSE_CAPACITY_V1.json), [shared audit with --all-pool option](audit_shared_response_capacity.py), [original512 comparison](SHARED_RESPONSE_CAPACITY_V1.json), [registered GPU experiment](GAUSSIAN_SUPPORT_EXCHANGE_PLAN_V1.md).
