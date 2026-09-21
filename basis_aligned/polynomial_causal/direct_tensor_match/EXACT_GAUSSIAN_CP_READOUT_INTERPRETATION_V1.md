**An exact Gaussian objective improves its own metric while damaging coefficient and text fidelity.**

The two CP400factor sets are frozen. Only16x512output coefficients are refitted, using exact isotropic Gaussian self/cross contractions from original weights. There are no fitting probes. Candidate architecture and1536product/2385920coefficient cost stay unchanged.

|Start|Metric|Archived coefficient fit|Exact Gaussian readout|
|---|---|---:|---:|
|1001|Gaussian error|98.976%|84.785%|
|1002|Gaussian error|98.878%|84.722%|
|1001|Sampled coefficient error|97.776%|123.273%|
|1002|Sampled coefficient error|97.899%|120.487%|
|1001|Opened text scalar error|40.875%|302.744%|
|1002|Opened text scalar error|39.027%|276.714%|

Integrity and Gaussian-improvement gates PASS; coefficient/text preservation FAIL. This is an objective tradeoff with fixed features, not sampled-probe overfitting. Both candidates remain poor in absolute terms. The earlier33-product projected-radial baseline's87.40%Gaussian error prevents treating84.7%at1536products as a major simplicity result.

The Hermite decomposition diagnoses the tradeoff exactly. The native mean energy is9.8444 and its degree2energy20.1003 in scaled16-output coordinates. For start1001, mean residual energy falls9.0893→.4993 and degree2residual20.0490→17.4402. But degree4explained energy changes+.1244→-1.7674: the fourth-order Hermite component becomes worse than zero prediction. The second start has the same pattern. These component energies are exact; the teacher's degree4norm and total normalized population error are not computed here.

Native trace identity agrees2.44e-15, readout solve residuals<1.8e-15, exporterrors<2.9e-7. Runtime7.23seconds,peak1.94GB. The cached native mean/quadratic projection makes later exact Gaussian feature optimization practical; its availability is an instrument, not evidence that this objective yields semantic circuits.

Next compare centered covariance, raw second moment and a Gaussian with the actual calibration mean and covariance. New CPU audit finds68.6%input energy in the mean; ignoring it is a substantive modeling choice. Noncentral exact controls pass<3.4e-14. Keep isotropic/coefficient diagnostics alongside data-informed improvements rather than silently changing success criteria.

[Native metric comparison](EXACT_GAUSSIAN_CP_READOUT_V1.json) · [Controls](NATIVE_GAUSSIAN_CP_CROSS_CONTROLS_V1.json) · [Covariance scope](COVARIANCE_METRIC_SCOPE_V1.md).
