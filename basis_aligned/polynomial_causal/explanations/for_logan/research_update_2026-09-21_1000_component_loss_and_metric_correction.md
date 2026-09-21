# Direct component fitting still fails; the coefficient metric needs clearer scope

21 September 2026, 10:00 UTC. **Fitting the actual component product does not solve the generalization gap. A follow-up audit also found that our current product dictionary is much less accurate under a native isotropic coefficient metric than under its calibration-shaped metric.** This changes the next research decision: more output-coefficient fitting in the same dictionary cannot fix that native coefficient error.

**Correction to the recent coefficient-error wording**

The recent local reports called their fixed coefficient objective “original coefficient error.” Original meant the unchanged target and output weighting, but the input coordinates were already shaped by calibration covariance. The reported 7–9% numbers were **not isotropic native-coordinate coefficient errors**. The figures and registered comparisons remain numerically valid; their scope needed to be stated explicitly.

Let $Q$ be one quadratic form in native input coordinates, and let $S$ be the positive square root of the calibration second moment about the fixed mean. The local tensor loss acts on forms proportional to

$$
T=SQS,
$$

with the registered output scales. It therefore constrains the whole polynomial in that geometry, but does not give all native-coordinate coefficient directions equal weight. This distinction matters for the user's requested comparison of weight-only and covariance-informed fitting. Earlier broader experiments compared those cases; the recent compact-dictionary fits were using the latter geometry.

The selected empirical-source direction fit has **7.94% calibration-shaped error**, versus **59.83% native isotropic error**, where the latter averages squared relative errors equally across the three source-read pairs before taking a square root. Its native pair errors are 50.08%, 60.14% and 67.93%. These are two specified metrics on the same exported polynomial, not conflicting measurements.

**The completed direct-component fit**

The existing calibration cache was replayed on the same 232 additional prefixes to retain all three later-state measurements. Replay passed: previous fields agree below $2.2\times10^{-14}$; all six folded source reads and all three component products agree with their native computations below $5\times10^{-7}$. No new evaluation examples were introduced.

We then held the 399-product dictionary fixed and optimized the actual conditional component

$$
\widehat\phi=(A_0+D_a w_a)(B_0+D_b w_b),
$$

plus the existing calibration-shaped coefficient penalty. The fixed arrays include the native later state, explicit RMS division, and exact affine/mean corrections. Holding one readout fixed makes the other update a convex quadratic solve. Alternating these updates is monotone in the objective, but is not a globally optimal method for the joint nonconvex problem.

Three weights, two starts and 20 alternating cycles completed in 51.4 seconds on CPU. The primary weight was preregistered as 0.1. Each setting selects its start by fitting objective. All programs retain 399 products and 896,198 floating coefficients.

| Component-loss weight | Full calibration component-3 error | Opened component-3 error | Calibration-shaped coefficient error |
|---|---:|---:|---:|
| 0, same-direction control | 2.17% | 15.10% | 7.927% |
| 0.1, primary | 1.89% | 14.91% | 7.928% |
| 1, descriptive | 1.59% | 14.63% | 7.935% |

The primary's three opened component errors are **1.52%, 1.85%, 14.91%**. It passes the coefficient guard but fails the third-component relative requirement of 13.14%. No candidate is adopted. Even the stronger-weight comparison fails that requirement.

Five seeded tests verified dense loss equivalence, stationary coordinate updates and monotone fitting. Independent exported execution reproduces calibration and opened errors below $10^{-8}$. This supports retaining the negative result, rather than dismissing it as an unverified optimizer or export bug.

**Why the small calibration error does not imply success**

The primary's third-component errors are 2.02% on the original 1,536 calibration states, 1.87% on the additional 14,848, and 14.91% on the opened 448. Target standard deviations are similar: approximately 232, 235 and 229. Thus the large gap is not a denominator artifact. These historical evaluation states have now been repeatedly inspected; they are diagnostic, not fresh validation.

Input geometry also shifts. In calibration-whitened coordinates, mean squared input radius is 1,152 in calibration and 1,466 in the opened panel. In raw coordinates the total input energy is slightly lower on the opened panel, but energy in the weakest 128 calibration axes is about **2.04 times** its training value. This is consistent with vulnerability in directions calibration underweights; it does not by itself prove the cause of the function-error gap.

An initial diagnostic attempted to rank eigenvectors of the whitened training second moment. That matrix is identity to numerical precision, so those axes are arbitrary. Before reporting grouped-axis conclusions, the diagnostic was corrected to use raw calibration covariance eigenvectors. The final receipt records both geometries.

**Can a better readout rescue the current dictionary?**

We solved the unregularized native-isotropic coefficient projection exactly within the existing dictionary, retaining its private/shared connectivity. Gram condition numbers are 161–206 and normal-equation residuals are below $3.3\times10^{-15}$.

The best such projection still has **59.43% native isotropic error**, only slightly below 59.83%. Its third-component opened error worsens to **17.01%**. This is a numerical projection optimum within the fixed dictionary, not a bound over other feature directions or graph structures.

Consequently, improving native-coordinate weight fidelity requires changing the product dictionary or topology, not another output-readout solve. A new direction-fitting comparison should explicitly retain both native isotropic and calibration-shaped objectives, with the same cost and original behavioral checks. Semantic identification, fresh/OOD manipulation, and standalone extraction remain unproved; all these programs still receive native intermediate states.

Evidence: [component fits](../../direct_tensor_match/COMPONENT_PAIR_READOUT_V1.json), [panel/execution audit](../../direct_tensor_match/COMPONENT_READOUT_PANEL_AUDIT_V1.json), [input geometry](../../direct_tensor_match/CALIBRATION_INPUT_GEOMETRY_V1.json), [both coefficient metrics](../../direct_tensor_match/NATIVE_COEFFICIENT_GEOMETRY_V1.json), and [fixed-dictionary native projection](../../direct_tensor_match/NATIVE_DICTIONARY_PROJECTION_V1.json).
