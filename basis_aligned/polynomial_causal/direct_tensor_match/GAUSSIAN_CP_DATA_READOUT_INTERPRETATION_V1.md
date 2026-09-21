**The calibration mean makes weight-based Gaussian fitting much better on text, but global and component guards still fail.**

All feature directions are frozen CP400directions, two seeds,1536products/2385920coefficients. Readouts are fitted from exact contractions of native weights under each Gaussian reference law. Only the law's mean/covariance comes from6144calibration input states; empirical output values are not fitted. Numerical integrity and improvement under each law pass; the combined preservation gate fails. Runtime17.79s,peak2.50GB.

|Reference law|Own-law error, seeds1001/1002|Opened text error|Isotropic Gaussian error|Sampled coefficient error|
|---|---:|---:|---:|---:|
|N(0,centered covariance)|27.10/26.27%|54.67/52.93%|95.10/94.82%|108.74/107.23%|
|N(mean,centered covariance)|6.74/6.92%|9.60/10.33%|106.43/97.99%|126.77/126.66%|
|N(0,raw second moment)|2.65/2.77%|15.00/15.23%|156.10/145.72%|219.27/227.82%|

Original coefficient-fit text errors were40.87/39.03%. The shifted-law result is a substantial conditional improvement without fitting output probes. Its own-law6.7%and raw-second-moment2.7%errors use different distributions and denominators; those numbers are not a universal ranking. The original empirical384-product graph still has8.13%aggregate text error at lower cost, so the new fit is not an improved simplicity/fidelity frontier on that metric.

The mean accounts for68.6%of calibration input squared norm. Matching a raw second moment with a zero-mean Gaussian does not match this actual mean, and produces different higher moments. Exact higher-moment integration fixes finite-probe interpolation; it does not make a Gaussian model the true activation distribution. The empirical covariance changes appreciably on the opened evaluation panel.

**Positive-result redteam: component transfer still fails.** Reusing the native-reference same-token newline pairs and independently cached sensitivity weights:

|Frozen program|Root1 same-token response error|Root1 sensitivity-weighted error|
|Empirical384 baseline|15.32%|14.54%|
|Shifted Gaussian CP,1001|13.11%|20.43%|
|Shifted Gaussian CP,1002|15.53%|21.82%|

The first seed improves the same-token response; neither reaches10%, and both worsen sensitivity-weighted error. These are30directed/20unordered pairs, not30independent samples. Cached native pair targets replay within3.55e-6. This remains an opened-panel local/component diagnostic, not finite feature removal, OOD or semantic identification. Improved aggregate values alone did not close the intervention-relevant gap.

Next use an exact coefficient-deterioration budget while fitting the shifted law. This maps a controlled tradeoff before spending on new feature learning; a pure switch of objective has already shown unacceptable global changes. The three-hour review derives a convex fixed-feature solution and CPUcontrols, with the teacher norm cancelling from loss differences.

[Native law comparison](GAUSSIAN_CP_DATA_READOUT_V1.json) · [Component transfer](GAUSSIAN_CP_COMPONENT_TRANSFER_V1.json) · [Metric definitions and counterexample](COVARIANCE_METRIC_SCOPE_V1.md) · [Constrained fitting plan](COEFFICIENT_GUARDED_NATIVE_PLAN_V1.md).
