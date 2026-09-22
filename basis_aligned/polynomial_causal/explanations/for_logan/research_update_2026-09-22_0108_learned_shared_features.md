# Learning the shared features: substantial improvement, still inaccurate components

22 September 2026, 01:08 UTC.

The experiment that was running during the residual discussion has finished. We allowed the **quadratic intermediate features themselves to change**, instead of only refitting their output weights or choosing which products to retain. Text-output reconstruction improved substantially at the same program size, but individual components remain too inaccurate for adoption.

The target is still the pure quartic path through the last two MLPs, projected onto the same 16 fixed output directions. It is not the entire model. Each candidate computes 144 quadratic features, combines them through 512 products, and reads those products into 16 outputs. Each quadratic feature uses four products of learned linear input directions. This costs 1,088 variable products in total, plus its linear coefficients and additions.

| Measurement on the existing evaluation panel | First initialization | Second initialization |
| --- | ---: | ---: |
| Pooled error before learning | 20.37% | 19.84% |
| Pooled error after learning | **12.19%** | **13.25%** |
| Feature 1 response error between matched token states | 25.85% | 36.35% |
| Feature 1 sensitivity-weighted value error | 22.06% | 19.82% |

The registered implementation checks and learning criterion passed. The component criterion required both response and sensitivity errors below 10% for both initializations; it failed.

Your concern about error distribution remains decisive. Output 0 has about 6–7% error, while outputs 1–2 have about 15–22%. The twelve small output directions have roughly 71–151% error across these candidates. More than half of squared error comes from the worst 10% of token states. These are improved shared-feature baselines, not uniformly faithful recovered components.

The stronger CP programs still have lower pooled error, about 6.3–6.6%, but use 1,536 variable products rather than 1,088. They also retain their own small-feature weaknesses. This comparison describes an accuracy–cost tradeoff; it does not establish that either representation identifies semantic circuits.

The run took about 20 minutes for two initializations, with 11 optimization steps each. That short budget is enough to show learning helps, but not enough to conclude that this architecture cannot fit further. It matched weights under the coefficient-plus-data-informed-Gaussian objective, without fitting text output labels. Error on isotropic Gaussian probes worsened, despite the text improvement, so the gain is specific to the metric and region being emphasized.

A useful constraint for the next experiment: giving small outputs more loss weight cannot improve independently fitted readouts over a fixed feature dictionary when regularization is weighted consistently. It must change shared feature learning or another coupled constraint. Five toy controls verified this distinction. We should therefore test any balancing hypothesis at the feature-learning stage and retain the separate intervention checks.

The larger panel of 256 new documents is queued for native-state capture. It evaluates the five previously frozen candidates; this newly completed fit is not silently added to that preregistration. No new-panel accuracy result is available yet.

[Full results and limitations](../../direct_tensor_match/SHARED_MIXED_FEATURES_INTERPRETATION_V1.md) · [Per-feature measurements](../../direct_tensor_match/LEARNED_SHARED_RESIDUALS_V1.json) · [Why readout weighting alone cannot fix the problem](../../direct_tensor_match/OUTPUT_BALANCING_INTERPRETATION_V1.md).
