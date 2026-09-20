# Next input-law diagnostic: two-component Gaussian mixture

Raw and normalized single-Gaussian probes both underestimate native text error;
normalization moves errors farther away. Do not launch the conditional
normalized-probe fitting sweep on this evidence.

The input-only feature audit finds covariance agreement on calibration but
quadratic feature covariance differs from Gaussian prediction by31.81%.
Generalized variance ratios span .401–3.395. Skewness reaches1.356 and
standardized fourth moments5.819 versus Gaussian3. This demonstrates failure
of Gaussian moment closure in the frozen feature directions, not proof of the
cause of the full reconstruction gap.

Next bounded CPU diagnostic: fit exactly two Gaussian components in the 16
frozen learned reader coordinates, split calibration rows at the median of
the calibration direction with largest absolute standardized third moment.
Fit means/covariances/weights on panel0 only. Compare single-Gaussian and
mixture predictions of all 16 third/fourth moments and the four quadratic
product covariances against both panels. Selection of split uses only panel0.
Predict mixture reduces held-out fourth-moment RMS discrepancy and product
covariance discrepancy, each by >=10%; fail if improvement occurs only in
calibration. Include a randomly permuted split with seed2035 as control.
No full-space mixture, student fitting or claim of identified circuits yet.
A successful low-dimensional control could motivate exact weighted contractions
under a small mixture; those require full input-law parameters and higher
moments for quartic residuals, not merely this 16-dimensional covariance.
