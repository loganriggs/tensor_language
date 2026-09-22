# Sharing quadratic intermediates: more compression, a failed primary fidelity gate

22 September 2026, 03:59 UTC.

Replacing pairs of input readers together can create shared quadratic computations, but the larger registered edit does not preserve the candidate well enough. **Both 128-edit candidates fail the 1% parent-response limit**, with 3.45–4.33%error on the original matched pairs. Worst individual-output response deviations are 8.12–11.60%, also above the 5% limit. No candidate is exported or promoted.

This is a second-stage graph experiment on the same frozen quartic CP models with 512 terms parents. The target function is still their approximation to the pure MLP16→MLP17 contribution in 16 fixed output directions. The experiment measures both edit fidelity relative to those parents and reconstruction relative to the native target. Those are different comparisons.

## The proposed edit

Each quartic term is written as a product of two quadratic intermediates, $q_i(x)p_i(x)$. We propose replacing $q_i$ by another existing quadratic $q_j$ and compensating its scale in the output coefficient:

$$
q_i p_i\longrightarrow\beta q_j p_i,
\qquad
\beta=\frac{\mathbb E[q_iq_jp_i^2]}{\mathbb E[q_j^2p_i^2]}.
$$

This scalar minimizes the individual term's Gaussian squared edit error. Proposals come from centered quadratic correlations under the existing calibration mean/covariance; they are ranked by exact downstream eighth-moment edit energy, weighted by the term's output vector. The rule changes at most one quadratic per quartic atom and prevents later removal of a retained target quadratic.

No text output labels or evaluation pairs select the edits. The score is local: cross terms between simultaneous edits are not included in selection. We therefore evaluate the combined program explicitly. We reuse the signed-reader compiler and exact product reassociation pass to count actual reachable multiplications.

## Primary result and the smaller setting

| Seed | Quadratic edits | Remaining linear readers | Distinct products | Original-panel parent value deviation | Parent response deviation | Worst output response deviation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1001 | 64 | 1,920 | 1,472 | 0.22% | 0.62% | 1.73% |
| 1001 | 128 | 1,792 | 1,407 | 0.85% | 4.33% | 11.60% |
| 1002 | 64 | 1,920 | 1,472 | 0.39% | 0.65% | 4.62% |
| 1002 | 128 | 1,792 | 1,408 | 0.89% | 3.45% | 8.12% |

The registered primary is 128 edits, not the best-looking row after inspection. Its pooled value deviations are below 1% on the original panel, but its response and individual-output limits fail. The larger opened panel also exposes significant weak-coordinate changes: seed 1001 has 12.39% worst-output value deviation despite only 1.14% pooled deviation.

The 64-edit setting is a descriptive tradeoff. It removes 128 private readers, matching the reader count of the earlier 128 linear-reader edits. Both methods use 2,238,464 reader/readout/writer floats under the same array accounting. Quadratic sharing uses 1,472 products, versus 1,527/1,525 after linear sharing and exact reassociation. However, the quadratic edit's worst original-panel output response deviations are 1.73% and 4.62%, worse than the earlier linear-sharing version. More product reuse is not free fidelity.

At 128 quadratic edits the array count is 2,091,008 floats, with 1,407/1,408 reachable products. Array counts retain the 512-column readout and exclude routing indices; they are a conservative coefficient price rather than an exported compact graph size. A repeated quartic root can account for an extra product saving. These logical counts are not runtime measurements.

## Native accuracy remains a separate obstacle

On the 16,384-state opened panel, the 64-edit candidates have native pooled value errors 7.38/7.57%, and native small-output response errors 58.12/59.53%. The 128-edit versions remain near 58–60% small-output response error. Thus preserving much of the parent's behavior does not repair its inaccurate smaller components.

This gives a concrete comparison of graph assumptions: sharing individual linear readers is less disruptive here; sharing whole quadratic products saves more multiplications at the same reader count, but can damage finite changes and weak output effects. A future global refit or response-aware selection would be a new hypothesis, not a reason to reinterpret this failed primary as successful.

Five independent Gaussian quadrature controls validate the quadratic Gram and the optimal scalar/edit-energy formulas. Native modified-reader predictions replay an independently assembled quadratic-substitution computation within 1.48e-16 relative error. The CPU screen took about 12 seconds on two threads. These checks support treating the fidelity failure as evidence about the proposed edit, rather than an observed indexing or moment bug.

[Plan](CP_QUADRATIC_REUSE_PLAN_V1.md) · [All counts, native errors, per-output edit errors and proposals](CP_QUADRATIC_REUSE_V1.json) · [Moment formulas and controls](quadratic_reuse_moments.py) · [Native audit](audit_cp_quadratic_reuse.py).
