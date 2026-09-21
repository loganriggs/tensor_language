**Changing only the output weights cannot bring the frozen wide dictionary below5%error on these panels.**

The CPU audit computes exact least-squares projections using SVD of the528 root features. Positive controls recover a known orthogonal residual floor and preserve it after invertible feature mixing. Rank tolerance is1e-10; all native designs have full column rank and minimum relative singular values at least8.6e-4. Ill-conditioned truncation does not explain the result.

| Learned dictionary | Calibration-fitted writer: panel2 | Panel2-only oracle floor | Pooled writer: panel1 / panel2 |
|---|---:|---:|---:|
|4 features, inherited start|12.93%|12.20%|6.52% / 12.40%|
|4 features, random start|14.92%|14.20%|7.02% / 14.42%|
|32 features, inherited start|15.05%|7.58%|5.27% / 9.16%|
|32 features, random start|19.89%|9.44%|7.06% / 11.55%|

For the primary inherited32 bank, the prediction of a more-than-twofold improvement from the panel2 oracle narrowly FAILS:7.575%is above half15.050%. The pooledwriter<5%both prediction also FAILS. A separate panel2-fit cannot get below7.58%, so no writer in this fixed feature span can attain5%there. The pooled fit's individual errors are not separately optimal; the separate-panel floor is the relevant impossibility evidence.

This distinguishes two limitations: readout transfer explains some excess error, but the feature span itself lacks the required responses. Oracles use evaluation targets deliberately as diagnostic lower bounds. They are not export candidates or OOD validation. The features themselves were learned on panel1, so its lower residual does not establish global recoverability.

Successor changes the structural question: each32x4 bank has at most256 independent input readers. Its entire graph is invariant along the orthogonal complement. Measure native quartic Jacobian energy in that complement before expanding products-per-feature or launching another objective sweep. This is a local sensitivity restriction, not a semantic circuit or natural-data error lower bound.

[Primary audit](WIDE_QUARTIC_READOUT_SPAN_V1.json) · [Code and controls](audit_wide_quartic_readout_span.py).
