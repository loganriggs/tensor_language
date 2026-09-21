# Sharing the linear terms preserves context effects but loses some source fidelity

21 September 2026, 02:41 UTC.

We held the adaptive512-product interaction fixed and compressed its two first-order linear maps into shared features. Context-only effects remained exactly unchanged, as the algebra predicts. The256-feature version reduced weight storage40%, but narrowly failed the registered source-preservation check.

## Native comparison

| Linear representation | Total weights | FineWeb source error | Code source error | CE added: FineWeb / code |
|---|---:|---:|---:|---:|
| Exact maps | 4,423,680 | 25.67% | 20.69% | 0.00551 / 0.02367 |
| 64shared features | 1,990,656 | 29.30% | 24.37% | 0.01466 / 0.07633 |
| 128shared features | 2,211,840 | 28.16% | 23.31% | 0.01178 / 0.06198 |
| 256shared features | 2,654,208 | 27.00% | 22.20% | 0.00921 / 0.04021 |

All retain512variable products. Context-only errors are identical across variants:55.37% FineWeb and39.55% code. The exact baseline reproduces its prior result, and the invariance check passes with zero discrepancy in the recorded summaries.

At256features, source errors increase5.17% and7.28% relative to exact linear maps, exceeding the5% allowance. The CE<0.05 check passes for this primary candidate, but passing CE does not override the failed intervention gate. No fresh confirmation or adoption is claimed.

## Why context is protected

Write the approximation around calibration means as

$$
\hat F(n,m)=\hat B(n-\bar n,m-\bar m)
+\hat J_n(n-\bar n)+\hat J_m(m-\bar m)+c.
$$

The context-dependent source difference is

$$
[\hat F(n,m')-\hat F(n,m)]
-[\hat F(\bar n,m')-\hat F(\bar n,m)]
=\hat B(n-\bar n,m'-m).
$$

The linear terms cancel, so changing their factorization cannot change this delta. We inject that same delta into a fixed native residual background; the identical downstream nonlinear result is therefore expected. This is a controlled invariance, not evidence that the rest of the function is preserved.

## Successor diagnostic: fit the input roles separately

The shared basis above was optimized for the paired sum of the two linear contributions. With correlated inputs, that objective can hide directions needed when one input changes independently. A CPU screen compared three objectives at the same256-dimensional output width:

| Calibration objective | Error in paired linear sum | Error in midpoint linear contribution | Error in source linear contribution |
|---|---:|---:|---:|
| Paired sum | 10.36% | 11.82% | 9.36% |
| Separate roles | 10.96% | 9.35% | 4.23% |
| Source contribution only | 25.79% | 23.15% | 1.84% |

“Separate roles” selects directions from the sum of the two individual output second moments, excluding their cross-covariance. It more than halves source linear error for a modest increase in paired-sum error. Source-only fitting shows the other extreme: it neglects the midpoint contribution and damages the sum.

This supports testing the separate-role objective next. It does not establish native transfer; these are calibration errors of linear components, not full-model intervention errors. The overall goal still needs accurate conditional behavior, stable semantic units, extraction, composition and broad OOD evidence.

Evidence: `MIDPOINT_ADAPTIVE_SHARED_LINEAR_V1.json`, `MIDPOINT_SHARED_LINEAR_NATIVE_V1.json`, and successor `MIDPOINT_LINEAR_ROLE_METRIC_V1.json` under `direct_tensor_match`.
