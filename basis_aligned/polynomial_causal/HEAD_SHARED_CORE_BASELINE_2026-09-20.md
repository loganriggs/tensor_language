# Joint head-coordinate tensor and shared sparse-core baseline

The exact128-coordinate normalized executor reproduces the native four intervention states. Sparse cores in joint tensor-energy bases do not preserve the mean/remainder cross interaction. Dense-core controls at the same output ranks also fail, so this is not evidence against sparsity alone. Full circuit prediction, selectivity and reuse remain unproven.

## Tensor, execution and accounting

Fold the head writer O into A=L O and B=R O, retaining all128input coordinates. The symmetric tensor is T_aij=sum_k D_ak(A_ki B_kj+A_kj B_ki)/2, shape1152x128x128. We compute output and input mode Gram eigenspaces from this joint tensor. Output ranks32/128 are tested; the input basis is full rank128. No factors are fitted on activations.

For a fixed output frame W and full orthogonal input frame P, symmetric core coefficients are pruned by Frobenius energy, counting unordered pairs once and weighting offdiagonal energy twice. `shared_quadratic_core.py` compiles all retained coefficients into a dictionary of unique input pairs. Each pair product is computed once and feeds its selected output channels. A unit test independently checks dense symmetric execution, signs, empty support and reuse of10products across30coefficients.

The remaining quadratic component is replaced in each normalized corner while background, background-linear terms and denominators remain exact. Both cross-term error and complete normalized interaction error must be<=10% in every family. This prevents the exact background/denominator terms from hiding a bad quadratic approximation.

Floating-value counts below cover only W, P and retained core coefficients. Index arrays are counted separately in the JSON. They are not whole-program prices: original background generators and projections remain necessary. The analysis uses float64; no production speedup or whole-model storage reduction is claimed.

## Measured frontier

Each entry reports the maximum over code, arithmetic, repetition and opened-prose families. The same prompts as v637/v638 are now opened.

| Core | Output rank | Coefficients | Unique products | Branch floating values | Tensor error | Max cross error | Max total interaction error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sparse | 32 | 256 | 120 | 53504 | 0.978 | 0.897 | 0.779 |
| sparse | 32 | 2048 | 616 | 55296 | 0.961 | 0.872 | 0.731 |
| sparse | 32 | 8192 | 1618 | 61440 | 0.943 | 0.856 | 0.702 |
| sparse | 128 | 256 | 113 | 164096 | 0.977 | 0.891 | 0.773 |
| sparse | 128 | 2048 | 413 | 165888 | 0.954 | 0.844 | 0.730 |
| sparse | 128 | 8192 | 1007 | 172032 | 0.923 | 0.800 | 0.660 |
| dense | 32 | 264192 | 8256 | 317440 | 0.911 | 0.784 | 0.672 |
| dense | 128 | 1056768 | 8256 | 1220608 | 0.773 | 0.609 | 0.525 |

The dense controls evaluate the output projection directly, mathematically equal to the full core with a complete input frame. At output rank128, cross errors are52–61%, while sparse8192-coefficient cores have64–80% error. Dense controls therefore isolate a substantial output-subspace bottleneck before pruning. Neither negative result excludes a different output frame, a task-specific observable, a nonorthogonal dictionary, or jointly optimized shared features.

## Validation and limitations

Native coordinate replay is5.10e-7 relative; native normalized interaction replay is3.05e-6. The exact executor and joint polynomial fold therefore pass the1e-4 instrument gate. All sparse and dense approximation gates fail. v639/v640 each use64native forwards,0fits/model updates. Background states are recomputed in the actual native corners, then accounted for in the conditional representation.

This baseline discovers shared linear features and sparse products, but the chosen tensor-energy frames do not isolate the observed interaction sufficiently. It provides executable and priced conditional objects, not semantic feature identification. Future work should change the observable or feature objective rather than merely increase these rank/budget grids. A learned method must be compared with these dense and sparse baselines under identical interfaces and causal checks.

[Sparse receipt](../bilinear_quotient/circuits/followups/head_shared_core_v639_result.json) · [dense control](../bilinear_quotient/circuits/followups/head_shared_core_v640_result.json) · [exact normalized fold](MEAN_REMAINDER_NORMALIZED_FOLD_2026-09-20.md)
