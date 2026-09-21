**Most derivative mismatch occurs inside the learned reader span.**

For the empirical32-feature quartic bank, centered-covariance local derivative error is47.86%on calibration anchors and35.60%on second-panel anchors. Of that squared error,94.66%and92.66%lies within the256-reader span. The irreducible unread response at this fixed span is11.05%/9.64%relative to total teacher derivative norm. These errors combine orthogonally in squared norm, not linearly.

| Same-bank output writer | Centered derivative error, panel0 / panel1 | Ordinary value error, panel0 / panel1 |
|---|---:|---:|
| Empirical values | 47.86% / 35.60% | 2.16% / 15.05% |
| Isotropic coefficients | 68.96% / 62.97% | 55.27% / 54.38% |
| Second-moment coefficients | 30.09% / 24.99% | 20.39% / 21.44% |
| Second-moment Gaussian function | 31.24% / 25.83% | 21.20% / 19.44% |

All three registered predictions PASS: numerical integrity, within-span dominance on both centered panels, and Gaussian writer beating empirical derivative error. Native FP32/FP64 agreement is retained; squared partition and student-unread leakage are below1.1e-13. Runtime2.67s. Results do not establish that the entire within-span gap can be repaired by this nonlinear architecture. They do show that unread directions alone cannot explain it.

The same output-writer changes improve derivative fidelity while worsening ordinary values. Before changing the feature architecture, test an exact combined value-plus-centered-derivative readout at the same budget. This is distinct from a Gaussian input-law closure: local derivatives are evaluated directly at calibration anchors and weighted by an explicitly chosen perturbation covariance. It remains data-informed and does not establish natural intervention fidelity.

The successor solver now passes independent augmented-least-squares controls on five bank structures, including cancellation, and exactly reproduces the existing empirical ridge fit at derivative weight zero. No native joint-fit outcome is claimed in this record.

[Derivative split](QUARTIC_DERIVATIVE_SPLIT_V1.json) · [Joint-fit solver controls](QUARTIC_JOINT_READOUT_CONTROLS_V1.json) · [Earlier capacity bounds](QUARTIC_READER_RANK_INTERPRETATION_V1.md).
