# Existing quartic dictionaries have little useful complementarity in this test

22 September 2026, 02:22 UTC. The native weight-based Gaussian refit of the union of two frozen512-atom CP dictionaries failed its registered small-output improvement screen. This differs from the earlier failed prediction ensemble: all1024quartic atoms receive independently refitted output coefficients, rather than averaging two fixed predictions.

Each atom is a product of four learned linear input forms. We recomputed exact native Gaussian cross moments and feature Gram matrices under the existing calibration mean/covariance, then solved output coordinates4–15 with ridge1e-6. No text labels were fitted. Both individual dictionaries were refitted under the identical objective, preventing an objective change from being mistaken for complementary feature capacity. Dominant coordinates0–3 were held at the seed1001parent's predictions.

| Dictionary | Original small-feature value RMS | Original small-feature response RMS | Larger-panel value RMS | Larger-panel response RMS |
| --- | ---: | ---: | ---: | ---: |
| Single1001, Gaussian refit | 51.73% | 63.20% | 55.82% | 57.57% |
| Single1002, Gaussian refit | 54.18% | 60.29% | 57.71% | 59.98% |
| Union1024, Gaussian refit | 50.61% | 59.11% | 54.49% | 55.93% |

The primary comparison uses the better single-bank score separately for values and responses. Union ratios are approximately0.978 and0.980, far above the required0.85. The original panel contains2048states and the existing30directed same-token pairs; the secondary opened panel contains16384states and2494token-position matched pairs. These are descriptive polynomial-interface responses, not semantic interventions or fresh/OOD validation. The full JSON retains original mixed-objective parent scores as additional controls.

The union doubles the variable product count from1536 to3072. An implementation retaining the dominant seed1001readout and reading all1024atoms into the other12outputs needs4,751,360 floating coefficients including the common1152x16writer, versus2,385,920 for an original CP parent. Zero readout entries need not be stored. The single1002 control with dominant coordinates still supplied by seed1001 also requires both banks for a whole16-output implementation; the table isolates small-output dictionary quality and is not a matched-total-program-price comparison. No larger candidate is adopted or exported from this failed screen.

Five quadrature controls validate analytic moments on complementary random banks, shared factors, repeated powers, signed coefficients and duplicate banks, with relative errors around1e-15. Native regularized normal-equation residuals pass1e-8. Duplicate columns can alter ridge geometry, so additional columns alone are not evidence of distinct computation. Fixed-ridge numerical solves are not certified unregularized lower bounds, and this result does not exclude refitting directions or other input laws.

The conjunction of failures is informative: output balancing alone, cheap low-degree residual corrections, and union of these existing quartic dictionaries each fail to recover the smaller output responses. The next structural hypothesis should learn missing higher-degree products against the native residual, preserving explicit output scope and literal added cost. Repeating combinations of already-tested banks has low priority. Native removal tests for the earlier conditional programs remain queued and separate.

[Protocol](UNION_CP_DICTIONARY_PLAN_V1.md) · [Independent quadrature checks](check_union_cp_dictionary.py) · [Native weight contractions and evaluation](audit_union_cp_dictionary.py) · [Results](UNION_CP_DICTIONARY_V1.json).
