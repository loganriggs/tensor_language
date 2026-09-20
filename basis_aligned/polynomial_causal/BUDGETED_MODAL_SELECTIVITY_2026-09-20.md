# Budgeted modal direction: exact nulling was overly restrictive

Earlier adaptive edits forced three modal gradients exactly to zero. Low collateral came with weak number effects. We red-teamed that negative result by allowing nonzero collateral during derivative-only direction selection while retaining the original native requirements: at least80% of the unitB number effect and at most10% modal-to-number ratio.

For each input, maximize signed number gradient n.a over a in[-1,1]^5 subject to abs(M.a)<=kappa*n.a. Two budgets were fixed before native evaluation: kappa0.05 and0.10. The smaller budget reserves room for nonlinear error; it does not change the native10%gate. LP primal-dual gaps, stationarity and constraint residuals pass. All amplitudes were frozen from existing gradients before the new native calls, with source receipt hashes verified. Both datasets are now opened; this intervention-rule test is not prospective OOD.

Native instrumentation passes: prior unitB/exact-null replay exactly0, native/reference effects7.09e-6,36prefix and120native+120reference suffix calls, no new derivatives or finite-outcome fitting.

| Dataset | Gradient budget | Native joint passes | Median target retention | Maximum modal ratio |
|---|---:|---:|---:|---:|
|Original32cells|5%|23/32|1.136|8.667%|
|New-construction16cells, now opened|5%|12/16|1.180|6.584%|
|Original32cells|10%|27/32|1.338|16.889%|
|New-construction16cells, now opened|10%|12/16|1.354|11.531%|

The5%budget achieves35/48 joint passes versus2/48 for exact nulling. Every5%budget cell meets the collateral bar; all13 failures concern target strength. Four new-construction failures are all attractor edits in the reversed-order after_speaking construction. The10%budget passes39/48 jointly but exceeds native collateral in eight cells. Neither meets the all-cell criterion.

The quadratic predictor passes both budgets in everycell, worst number6.203% and modal0.549% of number budget. Predicted versus native5%budget pass membership differs in only one cell, a near-threshold retention miss; the other failures were predicted before native execution. Thus local prediction is useful for selecting controls, while the five-source tradeoff remains restrictive in specific contexts.

This corrects the earlier implication that low strength is inherent to any selective five-source edit: exact modal nulling unnecessarily sacrifices target strength. It does not establish a reusable circuit. Amplitudes still depend on full native gradients, semantic controls cover only three modal contrasts, fixed shared directions failed, and full source generation remains charged. Original exact-null, shared-direction and native-capability failures remain intact.

Receipts: BUDGETED_MODAL_DIRECTIONS_V1.json; ../bilinear_quotient/circuits/followups/native_budgeted_modal_v1_result.json; BUDGETED_MODAL_NATIVE_CPU_AUDIT.json; BUDGETED_MODAL_FAILURE_DIAGNOSIS.json. Executors: budgeted_modal_direction.py, managed run_native_budgeted_modal_v1.py and audit_budgeted_modal_native.py.
