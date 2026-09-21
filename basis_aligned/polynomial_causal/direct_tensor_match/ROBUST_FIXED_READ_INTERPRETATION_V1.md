**Changing these correction coefficients cannot satisfy the robustness target**

The registered eight-round cutting-plane fit completed in29.87seconds. It changed only the second-read weights of the32existing correction squares for component three. Directions, first-read coefficients, all other components and graph cost were fixed. The original eight coefficient/value/derivative requirements and the new worst-read target were normalized so acceptance requires a maximum squared ratio at most1.

The fit fails. The final finite-cut optimum is1.50505733, while an exact separation call finds the trial's global objective30.1391. Thus the finite approximation has not converged to the global minimax optimum. **The final trial is rejected; it is not an improvement on the original graph.** No replacement artifact was saved.

Nevertheless, an independently reconstructed finite-cut lower bound is informative. For nonnegative weights summing to one, a weighted combination of the finite convex quadratics is bounded above by their maximum. Its unconstrained minimum is therefore a lower bound on the constrained minimax target:

$$
q_k(x)=x^\top G_kx+2b_k^\top x+c_k,
\qquad
\min_x\max_k q_k(x)
\ge \bar c-\bar b^\top\bar G^{-1}\bar b.
$$

Rebuilding all16adversarial cuts from the recorded iterations yields a bound of **1.50505722**. The weighted matrix is positive definite (minimum eigenvalue7.15e-11), stationarity residual2.36e-22, and the finite objective replays exactly. Cut radius excess is at most7.11e-15. The saved certificate contains the finite quadratic matrices, weights, witness inputs and minimizer for independent reconstruction.

The original graph supplies a feasible global upper bound of **1.53395977**. Consequently the optimum for this fixed-direction, one-read edit lies numerically between1.50506 and1.53396. The large separation gap at the final trial does not invalidate the finite lower bound, but prevents describing that trial as the global optimum. This is a numerical certificate with a substantial margin above1, not an interval-arithmetic proof.

**Decision**

Stop coefficient-only retries in this32-direction bank. Meeting this combination of source robustness and existing fidelity requirements requires changing the available directions, freeing additional reads, or changing the explicitly priced representation. This does not establish that a general shared arithmetic DAG cannot work.

A next direction proposal can use the weighted active-constraint gradient to identify a missing quadratic direction. Any such edit must be compared with a random-direction control, retain complete graph costs and be refitted against the original safeguards. A passing candidate would still require generated-interface checks and untouched behavioral evaluation; arbitrary ball inputs are not evidence of realizable text behavior.

The exact-export replay passed at3.34e-15 and cost was unchanged. There was no accepted candidate, so the preregistered generated-interface adoption audit was not invoked. Original data/graph artifacts remain unchanged.

[Fit history](ROBUST_FIXED_READ_V1.json) · [Independent bound audit](ROBUST_FIXED_READ_BOUND_AUDIT_V1.json) · [Finite certificate](ROBUST_FIXED_READ_FINITE_CERTIFICATE_V1.npz) · [Audit implementation](audit_robust_fixed_read_bound.py).
