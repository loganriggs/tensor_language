**Finite-change fitting improves training loss but leaves three native intervention failures.**

Historical source donors are a fixed257-row shift of2048calibration states. Input RMS is recomputed at both endpoints. Midpoint polarization makes the centered-quadratic response calculation exact; it is not a first-order approximation to RMS. The data-informed objective combines covariance coefficient error, .1isotropic coefficient error and finite-response error, each squared and normalized. Both exports include the same mean/tangent affine repair and3686products/14,067,072coefficients.

| Arm | Covariance error | Isotropic error | Fitting finite-response error | FineWeb natural error | Code natural error |
|---|---:|---:|---:|---:|---:|
|Fixed inherited readers; exact writer solve|11.21%|38.31%|4.55%|4.34%|2.09%|
|Learned readers,100Adamsteps|10.56%|38.04%|3.26%|4.20%|2.05%|

The learned objective improves8.90%. All fitting predictions pass; best checkpoint is step100, so convergence is not established. Independent CPU direct endpoint evaluation reproduces the fixed fit error with5.75e-8affine-centering discrepancy. FP32export replay passes for both.

| Primary full-path cohort | Fixed error | Learned error |
|---|---:|---:|
|FineWeb continuation|13.80%|13.83%|
|FineWeb spaced word|11.04%|10.85%|
|Code continuation|11.44%|11.30%|
|Code spaced word|7.58%|7.60%|

Both arms pass isolated-mode and natural-behavior limits but fail the full-path requirement in three of four cohorts. Learning helps two full-path cells and slightly worsens two, despite reducing training response error. Natural CEadded is+.00405/+.00137for fixed and+.00425/+.00080for learned (FineWeb/code; lower is better). These are the same opened32FineWeb/16code documents, not independent OOD confirmation. Native donors are token/cohort matched, unlike arbitrary calibration donors. Neither export is adopted as a faithful circuit.

This result does not establish that finite-response matching is useless or that the budget cannot preserve the circuit. It does show that replacing a tangent response constraint with this particular finite historical response constraint is insufficient. Final normalization/softcap and calibration-to-native intervention transfer remain distinct potential causes. The preregistered successor measures exact centered-quadratic response error on native swaps before final normalization, with both the training denominator and full residual-response denominator. It will distinguish whether substantial transfer error already exists in the polynomial metric; that diagnostic is not new fitting.

[Fits](FULL_QUADRATIC_FINITE_RESPONSE_V1.json) · [Native outcomes](FULL_QUADRATIC_FINITE_RESPONSE_NATIVE_V1.json) · [Independent export audit](FINITE_RESPONSE_EXPORT_AUDIT_V1.json).
