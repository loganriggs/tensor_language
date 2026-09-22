# Is the improvement consistent across documents?

22 September 2026, 07:36 UTC.

**The Gaussian-trained correction gives a modest, reproducible improvement on the existing evaluation panel. It does not solve the large component errors.** Resampling whole documents gives a relative improvement of about 7%, with intervals well below the previously stated 15% target. This is a completed CPU analysis, not a new GPU fit.

We are approximating the pure degree-four contribution through MLP16 and MLP17. Each input is a 1,152-dimensional normalized MLP16 state. We measure that contribution in 16 fixed output directions; these are projections of one selected computation, not 16 datapoints or 16 established semantic concepts. The CP512 baseline is fixed. Each fitted correction adds eight products of four learned linear forms to each of outputs 4–15; outputs 0–3 remain exactly unchanged. Other residual contributions, attention, biases, and mixed terms are outside this target.

The panel contains 256 documents, with 64 token-position states from each. Those positions are correlated. To estimate sampling uncertainty without treating all 16,384 states as independent, we drew 2,000 bootstrap samples of **256 whole documents with replacement**. The parent and both candidate fits use the same resampled documents. No coefficients are refitted.

For output coordinate $g$, define its relative RMS reconstruction error by

$$
\epsilon_g=\sqrt{\frac{\sum_n(\widehat F_g(x_n)-F_g(x_n))^2}{\sum_n F_g(x_n)^2}}.
$$

Our smaller-output score gives the twelve output coordinates equal weight:

$$
E=\sqrt{\frac1{12}\sum_{g=4}^{15}\epsilon_g^2},\qquad
\text{relative improvement}=1-\frac{E_{\rm candidate}}{E_{\rm parent}}.
$$

Each bootstrap sample recomputes both numerator and denominator from its resampled documents. We do not average per-document relative errors, which would define a different metric.

| Frozen candidate | Relative improvement over parent | 95% document-bootstrap percentile interval |
| --- | ---: | ---: |
| Gaussian Adam, seed 25001 | 7.05% | 6.67–7.46% |
| Gaussian Adam, seed 25002 | 7.15% | 6.79–7.53% |

These are relative reductions, not percentage-point reductions. The parent error is about 56.58%; the candidates have about 52.59% and 52.54% error. Thus the improvement is measurable while the absolute reconstruction remains poor for these components. The earlier per-output and residual-tail results still apply.

The intervals assume documents are exchangeable sampling units. They condition on the two frozen fits and this already-opened evaluation panel. They do not include training randomness, correct for earlier research decisions made using this panel, or establish performance on another distribution. Per-output intervals in the JSON are marginal intervals, not simultaneous coverage guarantees. We did not bootstrap matched response pairs as independent observations: pairs can share documents, so that would require a different dependence-aware analysis.

The GPU hybrid experiment remains queued at this checkpoint. This diagnostic does not change its fitting metric, capacity, or registered success criteria. The full third-order tensor, selective interventions, reusable components, and a smaller executable circuit remain unresolved.

Implementation: CPU float64, two threads, seed 39221, 2,000 resamples. Controls verify count-based resampling against explicit repeated documents, replay the previously reported errors, verify candidate/parent hashes, and confirm protected outputs remain identical. No new candidate is exported.

[All numerical results](../../direct_tensor_match/LOCAL_DOCUMENT_BOOTSTRAP_V1.json) · [Executable analysis](../../direct_tensor_match/audit_local_document_bootstrap.py) · [Sample counts and residual distribution](research_update_2026-09-22_0617_hybrid_metric_and_error_distribution.md).
