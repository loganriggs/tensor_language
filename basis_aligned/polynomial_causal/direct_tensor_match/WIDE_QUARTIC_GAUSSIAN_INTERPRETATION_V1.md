**Completing the Gaussian functional metric helps modestly, but does not repair native-input transfer.**

The dictionary is fixed at 32 quadratic features, each containing four bilinear products, with all 528 pair products. All readout variants cost 656 products and 903,168 stored coefficients. The target is the pure quartic MLP16→MLP17→unembedding contribution; residual/bias terms and intervening normalization remain outside this polynomial target.

| Readout fit | Calibration functional error | Second-panel functional error |
|---|---:|---:|
| Empirical values | 2.16% | 15.05% |
| Isotropic coefficients | 55.27% | 54.38% |
| Second-moment-weighted coefficients | 20.39% | 21.44% |
| Exact isotropic Gaussian function | 475.84% | 466.31% |
| Exact second-moment Gaussian function | 21.20% | 19.44% |

The final two fits use exact Gaussian functional moments, not sampled Gaussian fitting inputs. Five independent quadrature controls validate the new teacher cross and bank Gram. Native first-root FP32/FP64 cross errors are below8.4e-7; solve residuals below1.5e-15; physical exports below5.9e-7. Both Gaussian objectives improve over the empirical writer in their own regularized geometry. Registered integrity and improvement over weighted coefficient fitting PASS. The20%improvement over empirical transfer plus5%calibration requirement FAILS. Runtime14.26s.

The fourth Hermite degree accounts for91.17%of the empirical writer's output energy under the isotropic Gaussian law, but22.48%under the second-moment law. The registered preliminary prediction that this fraction would be below50%for both laws FAILS. These are student energy fractions, not fractions of teacher reconstruction error. Trace terms are consequential under the weighted law but cannot explain every failure by themselves.

A subsequent CPU audit compares the complete quartic-feature Gram matrices, allowing an optimal scalar multiplier to remove overall energy differences. The weighted Gaussian/calibration shape discrepancy is21.76%, versus8.47%between the two empirical panels. The preregistered ordering prediction PASSES. Isotropic Gaussian/calibration discrepancy is96.90%. The weighted Gaussian mean-direction discrepancy is6.77%after rescaling. This provides evidence that the assumed law misrepresents the learned feature moments, beyond an overall scale mismatch. It does not isolate the cause of prediction error: the teacher-to-feature cross is not audited by this comparison, and both finite panels have already been opened.

Do not begin a larger direction-learning sweep under these Gaussian objectives merely because their algebra is exact. Their native functional ranking remains inadequate. The exact moment operator is useful reusable machinery for controlled comparisons and toy verification; it is not a discovered circuit. No semantic unit, selective intervention, fresh OOD confirmation or adoption is claimed.

[Managed fit](WIDE_QUARTIC_GAUSSIAN_READOUT_V1.json) · [Empirical moment audit](WIDE_GAUSSIAN_FEATURE_MOMENTS_V1.json) · [Quadrature cross controls](PROJECTED_QUARTIC_GAUSSIAN_CONTROLS_V1.json) · [Bank Gram controls](GAUSSIAN_BANK_METRIC_CONTROLS_V1.json) · [Preregistration](WIDE_QUARTIC_GAUSSIAN_READOUT_PLAN_V1.md).
