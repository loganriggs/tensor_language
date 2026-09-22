# Matching the input norm does not close the Gaussian-to-text gap

2026-09-22 04:11 UTC. Frozen graph edits; no new fitting.

The inputs to the selected MLP16→MLP17 quartic path are RMS-normalized. A Gaussian approximation does not have this fixed-radius property. We tested whether this explains why its edit scores look too optimistic. **It explains little of the observed gap: radial normalization raises relative edit RMS by only 4–8%, while text errors remain about 2.5–3.7 times larger than normalized-probe errors.**

We used 32,768 artificial probes in four independently seeded batches, from the existing calibration mean and covariance. Each probe was rescaled to the calibration RMS radius. Both the parent and edited candidates are homogeneous quartic functions, so rescaling an input by $a$ multiplies their outputs and residual by $a^4$, and squared energies by $a^8$. Direct evaluations on normalized probes verified this shortcut within $2\times10^{-15}$ relative error.

| Seed | Edits | Exact Gaussian error | Sampled Gaussian error | Normalized Gaussian error | Opened text error |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1001 | 64 | 0.106% | 0.105% | 0.110% | 0.305% |
| 1001 | 128 | 0.423% | 0.425% | 0.454% | 1.143% |
| 1002 | 64 | 0.099% | 0.098% | 0.103% | 0.377% |
| 1002 | 128 | 0.347% | 0.351% | 0.378% | 0.938% |

All percentages measure the edited candidate versus its frozen parent, divided by parent RMS under the corresponding measure. They do not measure reconstruction of the native transformer. The four batch estimates are retained in the JSON; sampled unnormalized errors differ from exact Gaussian errors by less than 1%.

The observed text input norms have mean 33.94112529, standard deviation 1.5e-06, and range 33.94111957–33.94113138. The calibration radius is 33.94112529. Small deviations from an exact sphere remain, so the test uses a stated radius rather than assuming exact native normalization.

Radial normalization changes more than the norm distribution: it also changes the mean and covariance. This is a comparison between two explicitly defined artificial measures, not a clean causal decomposition of the transfer error. Nevertheless, it weakens the specific hypothesis that simply respecting RMS normalization is enough to make this Gaussian scoring rule representative of text.

The scientific implication is to retain the distinction between raw input covariance and the higher-order moments governing quartic error. Neither the previous exact Gaussian joint score nor this normalized-probe variant explains the text discrepancy. No new candidate is promoted. The next priority remains the pending native residual-learning and removal-stage experiments rather than further variants of the same graph edit.

[Protocol](GAUSSIAN_RADIAL_EDIT_PLAN_V1.md) · [All batch estimates and norm statistics](GAUSSIAN_RADIAL_EDIT_V1.json) · [Audit code](audit_gaussian_radial_edit.py).
