# Normalizing Gaussian probes did not close the prediction gap

2026-09-20 20:34 UTC

Frozen candidates were tested on two independent sets of 2,048 paired artificial
inputs, generated using calibration input mean and covariance only.

| Input law | Four-product quadratic error | Quartic error |
|---|---:|---:|
| Gaussian | 13.15–13.66% | 11.91–12.53% |
| Same draws normalized to native radius | 12.59–12.68% | 11.74–12.11% |
| Previously reused text panel | 22.00% | 18.38% |

Normalization makes artificial errors farther from text errors. The registered
law-improvement prediction failed; ordering and precision predictions passed.
No normalization-based fitting sweep is justified by this comparison alone.
The radius mismatch was real, but its proposed explanation of the prediction
gap did not survive this test. Normalization also changes mean and covariance.

An input-only follow-up checks moments in the frozen learned directions.
Calibration second moments agree with the Gaussian by construction, yet the
covariance of the four quadratic products differs by **31.81%**. Generalized
variance ratios range **0.40–3.39**. Standardized third moments reach 1.36 in
magnitude; fourth moments reach 5.82 versus Gaussian 3. Thus covariance alone
misses measurable structure in the interactions the candidate uses. These
measurements do not establish how much of its output error this explains.

The next bounded test compares single-Gaussian and two-component-mixture
moment predictions, including a random-split control and a held-out panel.
It uses input statistics only; no model outputs are fitted. All candidate
prediction results remain limited to the selected folded polynomial, with no
new OOD or semantic-circuit claim.

[Probe receipt](../../direct_tensor_match/NATIVE_NORMALIZED_PROBE_V1.json),
[feature moment audit](../../direct_tensor_match/FEATURE_MOMENT_AUDIT_V1.json),
[next diagnostic](../../direct_tensor_match/MIXTURE_MOMENT_DIAGNOSTIC_PLAN_V1.md).
