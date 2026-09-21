# Fixed-direction failure and a response-informed alternative

21 September2026. V2 retains exactly the eight registered direction sets, objectives, cost and fidelity bars. It corrects an implicit Python-list to float32 amplitude conversion in export; V1 remains preserved. Final loss/executor agreement improves to at most1.41e-13. All numerical instrument and cost checks pass; all fidelity outcomes remain failures.

For quadratics $q_k(c)=c^\top G_kc+2b_k^\top c+d_k$, any nonnegative weights summing to one give

$$\min_c\max_k q_k(c)\geq\min_c\sum_k w_kq_k(c).$$

The weighted quadratic minimum is evaluated with its eigendecomposition, retaining numerical range residuals. Across all eight candidates, the fitted objective minus this numerical lower bound is at most1.95e-11. All discarded-subspace linear residuals are zero. These are floating-point diagnostics, not interval-certified bounds. The primary lower bound is1.0305377136134, above the acceptance threshold1; its worst error ratio is about1.015. This excludes successful amplitude-only fitting in these tested fixed dictionaries to strong numerical evidence; it does not exclude other directions, first-read changes or general DAGs.

The primary active mixture places weight.90958 on covariance coefficient error,.09007 on component3 value error, and.000355 on component1 value error. Gradient auditing uses this mixture and permits arbitrary changes to the second quadratic forms. The strongest signed-square derivatives in covariance coordinates are-.000109,-.0000448,-.011669 for components1,2,3 respectively. All agree with independent central differences to absolute discrepancy below1.1e-9; the full native ratio replay is below4.7e-14.

Thus component3 has a concrete response-informed direction substantially different from merely choosing leading coefficient-residual eigenvectors. A negative derivative of this weighted mixture does not guarantee a decrease in the maximum constraint. The next graph edit must replace a correction node, retain14 total products, refit amplitudes, and independently reassess all original requirements. These448 states remain opened proposal data. No new OOD, extraction, manipulation, stable semantics or adoption evidence has been established.

[Native convex results](CONVEX_SOURCE_V2.json), [gradient result](CONDITIONAL_SOURCE_GRADIENT_V1.json), [conditional functional equations](conditional_source_constraints.py), [independent derivative check](audit_conditional_source_gradient.py).
