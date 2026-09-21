**Native output sensitivity exposes errors hidden by the ordinary scalar metric.**

Managed runtime2.20s, previous8cellremoval errors reproduce exactly, zeroedit0: instrumentPASS. Hypothesis that rawscalarerror is already>=10%everywhere FAILS. Weak-strength sensitivity-weighted approximation within20%relative ofactualerror in everycell PASSES. All previous causal-selectivity and response-fidelity failures remain.

|Domain/condition|Raw scalar error|Sensitivity-weighted scalar error|Actual quarter-removal error|Actual full-removal error|
|---|---:|---:|---:|---:|
|FineWeb newline|5.34%|24.89%|26.08%|31.07%|
|FineWeb other|14.26%|24.24%|24.43%|29.46%|
|Code newline|22.19%|25.82%|25.94%|26.16%|
|Code other|11.79%|11.71%|11.69%|12.08%|

For a fixed state, let z(s) be native final logits after adding s*d, d=-w1/(mean(h17²)+eps). Compute J=z'(0) through actual final RMSNorm and softcap. For native scalar t and approximate scalar h, weighted relative error is sqrt(sum((h-t)²||J||²)/sum(t²||J||²)). This is a state-dependent output-sensitivity metric, not unweighted tensor Frobenius error or ordinary input covariance. It describes local response and does not eliminate higher-order nonlinear errors. Full-removal baseline linearization errors reach37–43% on FineWeb; the weak-weighted prediction should not be advertised as an exact finite intervention model.

Independent CPU analytic-Jacobian controls match forward automatic differentiation to1.3e-16 and central differences to1.8e-10. They include three input scales. Numerical warnings from runner concern generic absolute-noop detection; referenceeffecterrors retainrelativeenergy denominators and previouserrorsreplayexactly.

Actual CPU successor on saved pertoken scalars/sensitivities:

- FineWebnewline highest-sensitivity10%ofpositions contribute88.86%ofweighted squared error but17.90%ofreference response energy. The ordinary scalar score hides consequential states. Weight-only effective sample size is12.79of106; it is a concentration statistic, not a count of independent observations.
- Independently fitted in-sample affine corrections reduce weighted errors24.89→12.55%FineWebnewline,25.82→13.56%code newline. They still miss10%; these are diagnostic oracles fitted on the evaluated cells, not candidate models or transfer results.
- FineWebnonnewline barely improves24.24→24.10%; codeother11.71→9.63%. No common scalar correction has been established.

Implication for decomposition: minimizing unweighted output or coefficient error can underweight states where a component has high downstream effect. For general residual error e(x), local logit error has quadratic form e(x)^T M(x)e(x), where M includes the actual denominator and final Jacobian. For this rank-one component it reduces to one scalar weight ||J||². A fixed covariance matrix is a different approximation to this state-dependent geometry. Fit on separate calibration states and evaluate on held states before claiming improvement; concentrating a fit on the observed11high-sensitivitynewlinepositions would overfit a diagnostic.

This changes the next fitting question from indiscriminately widening the dictionary to comparing existing feature learning under an explicitly downstream-sensitive metric, preserving coefficient and function objectives as controls. It does not make the feature a unique semantic unit, repair selectivity, or demonstrate OOD/extraction/reuse. User's broad arithmetic-circuit objective remains open; this is evidence about its reconstruction metric.

[Plan](ROOT_REMOVAL_GEOMETRY_PLAN_V1.md) · [Matched native results](ROOT_REMOVAL_GEOMETRY_V1.json) · [Sensitivity concentration and oracle audit](ROOT_SENSITIVITY_WEIGHT_AUDIT_V1.json) · [Analytic derivative controls](LOGIT_DIRECTIONAL_RESPONSE_CONTROLS_V1.json) · [Derivative implementation](logit_directional_response.py).
