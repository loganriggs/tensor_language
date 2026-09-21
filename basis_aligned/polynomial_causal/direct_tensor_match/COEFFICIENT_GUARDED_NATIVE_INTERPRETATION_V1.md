**A coefficient guard preserves most of the text improvement, but does not recover component fidelity.**

Two frozen CP512 dictionaries, seven budgets each, exact shifted-Gaussian readout fitting under an exact regularized coefficient-deterioration constraint. Runtime 4.91 seconds. Numerical integrity and preregistered tradeoff pass; preregistered component gate fails. No feature directions or graph connections changed.

|Budget / captured coefficient score|Text error, seeds 1001/1002|Sampled coefficient error|Root1 same-token response error|Root1 sensitivity error|
|---|---:|---:|---:|---:|
|0|40.87 / 39.03%|97.78 / 97.90%|29.54 / 26.49%|54.23 / 52.54%|
|0.01|17.97 / 14.75%|97.79 / 97.90%|27.76 / 23.37%|34.67 / 36.16%|
|0.1|11.85 / 13.18%|97.96 / 97.96%|19.59 / 22.57%|27.91 / 30.12%|
|**1, preregistered primary**|**9.79 / 10.03%**|**99.60 / 99.39%**|**14.27 / 16.29%**|**23.11 / 25.64%**|
|10|9.57 / 10.17%|114.71 / 114.07%|13.08 / 15.36%|20.39 / 21.76%|
|Unconstrained|9.60 / 10.33%|126.77 / 126.66%|13.11 / 15.53%|20.43 / 21.82%|

Budget 100 reaches the unconstrained endpoint in both starts. Lower error is better; zero prediction is 100%. The sampled coefficient diagnostic is not an exact full-Frobenius certificate. Budget ratios are relative to the captured regularized coefficient score, **not percentages of teacher error**. The primary ratio preserves a weak baseline near 100% coefficient error; passing that relative gate does not establish good global fidelity. Both starts still fail the registered 10% same-token response criterion and sensitivity preservation versus the cheaper empirical384 baseline. No post-hoc budget is promoted.

**What the convex result proves:** given these fixed feature dictionaries, each point globally minimizes the stated regularized Gaussian objective under its coefficient constraint, to the numerical KKT tolerance. It does not minimize text error, component error, or the objective over all possible feature directions/DAGs. In particular, Gaussian optimality cannot establish that no other readout could pass the component tests.

**Executed CPU follow-up separates that distinction.** In [fixed-dictionary response capacity](FIXED_CP_RESPONSE_CAPACITY_V1.json), root1 alone is fitted by SVD least squares with no ridge. All feature designs have rank512 at the stated numerical cutoff. Calibration-only fits use6144states; evaluation has2048states. Input/target row ordering is checked against token hashes. The initial hash assertion exposed the cache's65-token hash convention versus the sensitivity cache's64-token convention; checking both documented conventions resolves it without changing rows or removing the check.

|Readout fit|Evaluation root1 sensitivity error, starts1001/1002|Same-token response error|
|Calibration uniform|14.93 / 16.67%|14.53 / 12.09%|
|Calibration sensitivity weighted|16.01 / 16.75%|46.30 / 24.46%|
|Evaluation sensitivity **oracle**|6.21 / 6.48%|29.41 / 29.70%|
|Evaluation sensitivity + exact pair constraints **oracle**|6.32 / 6.61%|0.000355 / 0.000355%|

The last row imposes20 independent linear pair constraints (30 directed rows), using cached native target differences. Independent native pair references agree within3.55e-6 relative; the pair constraints themselves solve within4.8e-14. The constrained readout is a weighted least-squares projection with the equalities enforced. Calibration sensitivity error for this evaluation-fitted oracle is19.48/22.74%, showing the reverse transfer gap too.

These oracle rows use evaluation labels and are **not held-out results or deployable candidates**; no coefficients are exported. They show the fixed512-feature dictionaries can simultaneously express the desired local responses and below10% sensitivity error on this finite panel. Hence the component failure cannot be attributed solely to lack of representational capacity on that panel. Nor is lower Gaussian error sufficient to identify the correct readout. Calibration distribution/response coverage and the chosen objective remain substantive problems. This does not resolve global coefficient fidelity, finite native removals, OOD prediction or stable semantic identity.

Next scientific choice: test whether a response-aware calibration objective transfers across prefix groups before introducing more feature parameters. Keep this a diagnostic beside pure-weight discovery, not a redefinition of the goal. If calibration response supervision improves only training responses, reject that route and change the dictionary/objective rather than repeating readout sweeps.

[Native result](COEFFICIENT_GUARDED_NATIVE_V1.json) · [Preregistered plan](COEFFICIENT_GUARDED_NATIVE_PLAN_V1.md) · [CPU implementation](audit_fixed_cp_response_capacity.py).
