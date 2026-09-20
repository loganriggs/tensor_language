# Native quartic Gaussian trace/radial baseline — 2026-09-20 16:33 UTC

Exact identity for fully symmetric H: E||H x^4||²=24||H||F²+72||TrH||F²+9||Tr²H||² under standard Gaussian x. Independent quadrature verified coefficients/orthogonality and a factor-contracted exact mean to<7e-16.

Compute the native pure MLP16→17 numerator's Gaussian mean exactly from weights using the first-layer quadratic Gram matrix. Evaluate4,096 independent Gaussian inputs in float64 to estimate total function energy and its sampling uncertainty. Compare constant-mean and homogeneous radial quartic baselines; radial coefficient is mean/[d(d+2)]. The radial baseline is a legitimate degree-four polynomial and requires one norm-square squared plus an output vector, along with the output frame. Constant baseline is only a diagnostic, not a homogeneous quartic student.

Also evaluate available final V1/V2 student artifacts on these new inputs, restore teacher scaling, and report exact mean mismatch. Distinguish mean-error contribution from full function error. Use prior Frobenius-energy estimate to estimate fourth-Hermite energy contribution; any remainder is an estimate, not exact trace-energy recovery. No training, checkpoint changes, native forwards, or circuit claims.

Predictions: Monte Carlo teacher mean within5% of exact mean; homogeneous radial baseline function relative error<90%; some coefficient-trained student has mean contribution explaining≥50% of its Gaussian residual energy. Preserve nulls; these test metric structure, not overall model quality.
