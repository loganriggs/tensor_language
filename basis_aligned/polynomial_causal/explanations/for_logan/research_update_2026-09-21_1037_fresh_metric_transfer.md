# Fresh interventions pass absolute limits, but expose a domain tradeoff

21 September 2026, 10:37 UTC. **All three wider graphs pass every absolute accuracy limit on the new panel. None passes every comparison against both same-storage baselines.** The mixed objective helps the difficult component on FineWeb relative to covariance-only fitting, but hurts it on code. This is evidence of conditional predictive accuracy and a real metric tradeoff, not adoption of a complete circuit.

**What was tested**

Three frozen 592-product graphs were fitted with covariance-only, mixed, or native-isotropic coefficient objectives. Their two frozen comparison programs each use 1,152 products in independent pair banks, fitted in covariance-shaped or native-isotropic geometry. Coefficient storage is nearly equal: 1,342,028 versus 1,340,940 floats. Earlier arithmetic accounting showed that the graph's fewer nonlinear products do not imply fewer total scalar multiplications.

The sixth panel contains 32 new FineWeb documents and 16 new code files, with 4,818 and 3,212 same-token donor pairs across distinct documents/files. Every candidate receives the same 48 native captures. Earlier panels and known cached FineWeb excerpts were excluded as specified in the frozen plan.

For each component and their sum, we compare the vocabulary-logit effect of removing the original versus reconstructed contribution. **Natural** means the original state. **Hybrid** means the later state after replacing the preceding MLP contribution with a donor's contribution. **Change** is the difference between hybrid and natural removal effects. The actual final MLP, RMSNorm, unembedding and softcap remain in the evaluation.

Across two domains, four component selections, three intervention families and three token cohorts, there are 72 comparisons per graph. Natural/hybrid errors must be at most 15%; change errors at most 20%. Relative comparisons allow at most 10% more error than each matched baseline.

| Frozen graph | Absolute failures / 72 | Failures against covariance baseline | Failures against isotropic baseline | Failures against either |
|---|---:|---:|---:|---:|
| Mixed, registered primary | **0** | 10 | 2 | 10 |
| Native isotropic | **0** | 15 | 4 | 15 |
| Covariance-only | **0** | 11 | 6 | 13 |

The primary therefore passes the instrument and absolute checks, and fails both relative-baseline checks. Its earlier coefficient-fit covariance guard also remains failed. The other arms are prespecified comparisons, not replacements chosen after seeing the new data.

**The metric choice transfers differently across domains**

The table shows component-three error on all eligible sites. The complete receipt retains every component and token cohort.

| Domain / effect | Covariance graph | Mixed graph | Isotropic graph | Covariance pair baseline | Isotropic pair baseline |
|---|---:|---:|---:|---:|---:|
| FineWeb natural | 11.59% | 7.69% | 7.90% | 9.27% | 11.09% |
| FineWeb hybrid | 10.23% | 7.70% | 7.01% | 6.72% | 7.84% |
| FineWeb change | 9.21% | 7.53% | 7.55% | 8.38% | 9.92% |
| Code natural | 1.57% | 2.59% | 3.26% | 1.62% | 2.28% |
| Code hybrid | 1.60% | 2.32% | 2.98% | 1.62% | 2.10% |
| Code change | 4.22% | 4.04% | 4.07% | 3.73% | 4.14% |

Native-coordinate fitting does not universally improve behavioral fidelity. On this panel it helps the FineWeb component relative to the covariance-only graph, while covariance-only fitting is better on code's natural and hybrid effects. Neither coefficient norm alone predicts the preferred approximation for all interventions.

Combined outputs do not erase the tradeoff. For the sum of all three components, FineWeb natural-effect error is **1.39%** for the mixed graph versus **1.78%** for the covariance pair baseline. On code it is **1.97% versus 1.43%**. The complete component tests remain necessary because cancellation in the sum can hide individual errors.

**Red-team checks and uncertainty**

Native source replay error is $1.70\times10^{-7}$ and final-state replay is zero at the measured precision. The run completed in 10.66 seconds. A separate CPU audit recomputes every point summary from per-recipient records and reproduces the registered verdicts.

The audit uses 4,000 paired recipient-document/file bootstrap resamples. For the primary's third-component hybrid effect on code, the error ratio to the covariance pair baseline is **1.43**, with a pointwise 95% interval **1.30–1.57**. For the combined hybrid effect it is **1.23 [1.11, 1.37]**. Six primary comparisons have lower interval endpoints above the allowed 1.10 ratio against that baseline.

These intervals keep the donor mapping and fitted programs fixed and are not adjusted for multiple comparisons. They do not quantify fitting or donor-selection uncertainty. Some individual misses remain uncertain: the FineWeb third-component hybrid ratio is **1.15 [1.04, 1.28]**. The broader failure is not supported only by such borderline cells.

The first enqueue attempt was refused because the static experiment checker could not see prediction fields inherited from the reusable evaluator. The wrapper was changed to expose those same computed fields explicitly; no scientific threshold or outcome was changed. The corrected job passed the gate and ran once.

**What this changes for the research direction**

We now have broader absolute accuracy than the earlier narrow graph achieved, with no absolute failures on this fresh panel. But the result does not establish a universally superior decomposition, a runtime reduction, or stable semantic features. Both native intermediate inputs remain supplied; upstream extraction and independent reuse are still open.

The next mathematical question is how a compact computation can preserve the directions that different input distributions and downstream operations need, without simply increasing all widths or choosing a winner after evaluation. The present evidence supports examining that tradeoff explicitly; it does not support another claim based only on lower coefficient error or fewer product nodes.

Evidence: [full native results](../../direct_tensor_match/DUAL_FRESH_NATIVE_V1.json), [independent point and bootstrap audit](../../direct_tensor_match/DUAL_FRESH_AUDIT_V1.json), and [frozen plan](../../direct_tensor_match/DUAL_FRESH_PLAN_V1.json). The full circuit objective remains open.
