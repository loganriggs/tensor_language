# Even oracle readouts cannot recover the smaller outputs in the current CP dictionaries

22 September 2026, 04:17 UTC.

**The current 512-feature dictionaries lack the span needed to reconstruct the smaller output coordinates accurately on the already measured text states.** This statement is stronger and narrower than the earlier failure of Gaussian fitting: it uses the best unregularized readout on those exact states, with direct access to their reference outputs.

This is an oracle capacity diagnostic, not a trained model or a held-out result. We fit no new feature directions and export no replacement.

Each dictionary consists of 512 scalar quartic features, each a product of four learned linear forms. Their values form a matrix $A$ with 16,384 rows and 512 columns. The reference $Y$ has 16 output coordinates for the selected pure MLP16→MLP17 quartic path. For each dictionary, solve

$$
\min_C\|AC-Y\|_F^2.
$$

Because each column of $C$ is independent, this simultaneously minimizes every output coordinate's squared error. Changing the output weighting cannot improve any one coordinate beyond its minimum while keeping the same feature span. For matched-state responses, we separately replace $A,Y$ by their differences on the existing 2,494 pairs and solve another least-squares problem. The value and response optima are different readouts; we do not claim one readout achieves both minima.

| Dictionary | Oracle pooled value error | Oracle small-output value RMS | Oracle pooled response error | Oracle small-output response RMS |
| --- | ---: | ---: | ---: | ---: |
| Seed 1001 | 3.89% | 37.72% | 6.19% | 34.93% |
| Seed 1002 | 3.96% | 38.24% | 6.21% | 35.21% |

Every individual output numbered 4–15 still has roughly **26–49% value error** and **26–46% response error**, even with this oracle advantage. The prediction that neither dictionary could bring all small outputs below 10% passes; in fact none of the twelve reaches that threshold. The pooled optimum remains much better because large outputs dominate it.

These are finite-panel numerical minima within two specific frozen dictionaries. They are not lower bounds on CP with different learned factors, Tucker, HT, arithmetic DAGs, larger dictionaries, programs with added constants or other operations, or population error. They do not rule out a simple missing computation. They do establish that additional readout tuning alone cannot meet a 10% per-output criterion on these measured states for these dictionaries.

All four design matrices have full numerical column rank 512. Condition numbers range from about 4,421 to 13,957. Float64 SVD least squares with relative cutoff 1e-13 agrees with an independent QR projection; residual orthogonality is also checked. The full singular spectra, agreement tolerances and all output errors are in the results. This is numerical evidence rather than an interval-certified algebraic bound.

This resolves an ambiguity left by the Gaussian-versus-text diagnostics. Better calibration information genuinely improves readouts, but it cannot supply the missing directions inside these fixed feature spans. Prioritize the queued learner that adds new quartic features to outputs 4–15, then test transfer and native interventions if it improves. That learner may still fail from optimization or its Gaussian training measure; its failure would not prove that all richer representations fail.

[Protocol](CURRENT_CP_ORACLE_CAPACITY_PLAN_V1.md) · [All per-output minima and numerical diagnostics](CURRENT_CP_ORACLE_CAPACITY_V1.json) · [Executable audit](audit_current_cp_oracle_capacity.py).
