**Joint value/derivative fitting improves transfer modestly at fixed cost, but misses the primary value gate.**

The same32 quadratic features,528 pair products and903,168stored coefficients are used in every arm. Only the output writer changes. Derivatives are evaluated directly at16 calibration anchors with centered-covariance perturbation weighting; this is not a Gaussian closure of the quartic function.

| Derivative weight | Calibration value error | Second-panel value error | Calibration derivative error | Second-panel derivative error |
|---|---:|---:|---:|---:|
|0|2.16%|15.05%|47.86%|35.60%|
|0.01|2.89%|13.72%|32.70%|31.29%|
|0.1|4.93%|13.02%|23.48%|28.17%|
|1, preregistered primary|7.74%|13.61%|20.70%|27.73%|
|10|10.84%|14.92%|20.17%|29.67%|

Integrity PASS; primary value gate FAIL; primary derivative gate PASS. The primary required calibration<=5%and at least20%second-panel value improvement, neither achieved. Increasing derivative weight reduces fitting derivative error monotonically, but second-panel derivative error worsens at10. Secondary0.1is a diagnostic tradeoff, not a replacement primary selected after looking at the second panel. Narrow four-feature models already reached about12.93%second-panel value error at much lower cost; this broad fit has not established a value/cost win over them, and their derivative comparison remains unmeasured here.

Independent augmented least-squares controls pass on five structures; zero weight reproduces the old empirical fit within2e-11absolute error. Solve residuals are below1.5e-15 and physical export errors below6.2e-7. Native runtime1.01s. All panels are already opened. No full normalized-model intervention, semantic feature identity or adoption is claimed.

The next stage-two candidate changes program structure: compress the output space of the root readout, express each output-shared root as a symmetric quadratic form of the32 learned quadratic features, and diagonalize that form into weighted squares. This retains the same shared leaf features but can reduce stored root coefficients substantially. It is a restricted block-term graph, not arbitrary DAG search. Five fresh-probe compiler controls now pass, including shared output and cancellation cases. Native savings and fidelity have not yet been measured.

[Primary fit](QUARTIC_JOINT_READOUT_V1.json) · [Derivative diagnosis](QUARTIC_DERIVATIVE_SPLIT_INTERPRETATION_V1.md) · [Block compiler controls](SHARED_ROOT_BLOCK_CONTROLS_V1.json).
