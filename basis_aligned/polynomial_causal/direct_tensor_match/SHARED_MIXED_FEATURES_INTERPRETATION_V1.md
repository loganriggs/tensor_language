# Learning shared quadratic producers helps, but component fidelity still fails

22 September2026,01:08UTC. Completed managed run SHARED_MIXED_FEATURES_NATIVE_V1.json:1205.47seconds, peak2.71GB allocated.11Muonupdates per start,144quadraticproducers each4products, fixed512rootpairs,16outputreadouts.1088variableproducts/1353728floats/1024indices unchanged. Both seeds1101/1102. Exact mixed coefficient/shiftedGaussian objective; mean/covariance data-informed, no text output labels fitted. Best objective checkpoint, not best evaluation checkpoint.

| Metric | Seed1101 before→after | Seed1102 before→after |
|---|---:|---:|
| Opened text16output error |20.375→12.187%|19.842→13.252%|
| Root1 sensitivity-weighted error |39.213→22.056%|42.270→19.824%|
| Root1 same-token response error |36.118→25.846%|38.990→36.353%|
| Sampled global coefficient error |98.792→96.966%|99.352→97.572%|
| Isotropic Gaussian probe error |87.585→92.973%|88.756→92.658%|
| Improvement in captured regularized score |5.068%|4.797%|

Integrity PASS, learning PASS, component FAIL. The score improvement is relative to the negative teacher-constant-omitted profiled objective, not a5%relative reduction in full tensor error. The isotropic probe metric worsens while the data-informed objective/text fit improves; these are distinct measures and must remain separate. No broad polynomial recovery claim.

Learning producers makes a substantial difference compared with merely choosing more rootpairs from the same fixed pool. Nevertheless the larger CP512 programs remain stronger on this panel (6.32/6.59%text), at1536products/2385920floats versus1088/1353728 here. Neither family has established the full selective-circuit goal.11steps is a short profile-limited budget and cannot establish an architecture-level impossibility.

Per-feature follow-up: learned shared seed1101 has errors6.73/14.69/19.88% on output0/1/2; seed1102 has6.40/20.12/22.14%. Feature3 remains42.17/51.97%. Features4–15 range71.7–151.3% and70.8–123.9%. The worst10%states carry52.1/58.1%of total squared error. This confirms that pooled gains do not establish uniform recovery, and high relative errors in small coordinates are not just a CP-specific issue.

No finite-removal or fresh-panel result for these new candidates yet. The pending256documentpanel preregistered five older artifacts; do not silently add these and describe their inclusion as preregistered. CPU artifact prediction reproduces completed GPU scores within1e-5. Existing gradient controls and native profile passed before execution; failure is not explained by known gradient/export errors, but longer optimization and objective choices remain open.

Decision: retain as an improved shared baseline, not an adopted circuit. Next compare completedCPparentrefit and the frozen freshpanel. Output-balanced producer optimization is now a plausible distinct hypothesis, since simple readout reweighting is algebraically insufficient (OUTPUT_BALANCING_INTERPRETATION_V1.md). Any new objective must explicitly trade dominant-feature accuracy for small-feature/response fidelity and preserve literal price and independent evaluation.

References: SHARED_MIXED_FEATURES_PLAN_V1.md; SHARED_MIXED_FEATURES_NATIVE_V1.json; LEARNED_SHARED_RESIDUALS_V1.json; audit_learned_shared_residuals.py. All metrics above refer to selected MLP16→17purequartic16outputsubspace; not full model, nor all residual/cross terms.
