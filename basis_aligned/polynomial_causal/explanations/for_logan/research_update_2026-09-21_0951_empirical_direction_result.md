# Direction fitting improves the empirical match, but the circuit gate still fails

21 September 2026, 09:51 UTC. **The expanded empirical objective helps, but does not recover a sufficiently faithful compact circuit.** All six registered fits completed in 434.7 seconds. The selected primary passes the coefficient-fidelity guard and independent execution checks, but fails the third-component comparison against the separate baseline. No new candidate is adopted.

**The controlled comparison**

All fits use the same 399-product graph and 896,198 floating coefficients. Adam updates input directions; an exact constrained solve updates output coefficients at every step. Each setting uses two 1%-perturbation starts, 1,000 steps, initial rate 0.005, and the same 16,384 calibration states.

The objective combines original coefficient error with measured source-quadratic output error, weighted by $\lambda$. These six source measurements feed three later component computations; their accuracy is a surrogate for those components' actual behavior. A coefficient-only control receives the same additional optimization budget.

Within each setting, the winner is selected by its fitting objective, not its evaluation error. The registered primary is $\lambda=1$; the larger weight is descriptive.

| Selected setting | Coefficient error | Component 1 | Component 2 | Component 3 |
|---|---:|---:|---:|---:|
| Coefficient-only, $\lambda=0$ | 7.39% | 2.05% | 2.24% | 17.11% |
| Primary, $\lambda=1$ | 7.94% | 1.50% | 1.85% | 14.78% |
| Strong empirical weight, $\lambda=10$ | 9.31% | 1.95% | 2.25% | 15.04% |

The primary meets the 15% absolute component limit, but the separate third-component baseline is 11.94%. The allowed 10% degradation therefore requires at most **13.14%**. The primary's coefficient error is within 1.10 times the equally continued coefficient-only control, so that guard passes. Stronger empirical weighting does not improve the selected third-component result and would fail the same coefficient guard.

The earlier fixed-direction empirical fit was 14.91% on component three. Allowing directions to adapt reaches 14.78% while reducing coefficient error from 8.82% to 7.94%. That is a modest behavioral gain, not a resolved failure. These are the same previously examined 448 diagnostic states, not a fresh test.

**Useful private computation, uncertain identity**

Removing the primary's centered private quadratic branch increases third-component error from 14.78% to 22.66%, a 34.8% error reduction from retaining it. The other two components are exactly unchanged because the branch only feeds the third. This verifies algebraic selectivity, not semantic selectivity.

Across the two local restarts, the private quadratic form has cosine agreement **0.914**, below the previously used 0.99 identity standard. Its particular form is therefore still not stably identified, even though the branch is useful. These are near-parent starts, not independent discovery from random initialization.

**Sensitivity depends on which supplied input changes**

For component three, the primary improves the conditional derivative error with respect to later $h$ from 18.34% to **14.23%**, compared with the coefficient-only control. But its earlier-$z$ derivative error worsens from 19.18% to **19.92%**. The 512-product comparison program's later-$h$ error is 12.79%.

The graph still receives both states from the native model. Neither conditional derivative includes the missing upstream computation connecting them. Thus the evidence does not establish end-to-end extraction or robustness.

**Independent checks and next step**

A separate audit reconstructs dense native quadratic forms from each exported graph, evaluates all three components, recomputes original coefficient error, and counts physically stored coefficients and distinct products. Maximum replay discrepancy is below $4\times10^{-15}$. It reproduces all registered verdicts. The implementation checks support retaining the negative result; they do not establish that the optimizer found the best possible graph.

The next step repairs a concrete information gap in the existing calibration cache: it retained the later-state projection only for component three. We will replay the **same 232 additional training prefixes** and save the later states and all three component measurements, checking the existing inputs and third-component values for consistency. This enables tests of the actual conditional component operation rather than only the source-read surrogate. It adds no new examples and is not fresh validation.

The broader program is still decomposition followed by arithmetic graph search. This run tests a continuous fit within one topology; general graph search, stable semantic units, standalone extraction and fresh selective reuse remain unresolved.

Evidence: [all six fits and selection](../../direct_tensor_match/EMPIRICAL_SOURCE_DIRECTIONS_V1.json), [independent export and private-branch audit](../../direct_tensor_match/EMPIRICAL_SOURCE_DIRECTIONS_AUDIT_V1.json), [later-input response](../../direct_tensor_match/LATER_INPUT_RESPONSE_V1.json), and [registered cache completion](../../direct_tensor_match/EXPANDED_COMPONENT_INPUTS_PLAN_V1.json).
