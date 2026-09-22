# Why local quadratic-sharing scores understated text error

22 September 2026, 04:09 UTC. Descriptive analysis of the frozen 64- and 128-edit candidates; no fitting or new candidate selection.

**The Gaussian metric underestimates relative edit error on text by 2.7–3.8 times. Omitted interactions between edits also matter more on text than under that Gaussian.** Replacing the local score with an exact joint score under the same Gaussian would therefore address only part of the observed discrepancy.

Let $e_i(x)$ be the 16-dimensional change caused by edit $i$, including its output writer and compensating scalar. The full edit is $e(x)=\sum_i e_i(x)$. Under any stated measure $\nu$, define

$$
I_\nu=\frac{\mathbb E_\nu\|\sum_i e_i(x)\|^2}{\sum_i\mathbb E_\nu\|e_i(x)\|^2}.
$$

A ratio of one means cross terms cancel in aggregate. A ratio above one means constructive interference. These are squared-energy ratios, not RMS errors. We evaluate the Gaussian with the existing calibration mean and covariance exactly through eighth moments, and the empirical measure on all 16,384 opened text states.

| Parent seed | Edits | Gaussian interference | Text interference | Gaussian relative edit RMS | Text relative edit RMS |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1001 | 64 | 1.04 | 1.62 | 0.106% | 0.305% |
| 1001 | 128 | 1.04 | 1.56 | 0.423% | 1.143% |
| 1002 | 64 | 1.11 | 2.81 | 0.099% | 0.377% |
| 1002 | 128 | 1.20 | 2.40 | 0.347% | 0.938% |

Relative RMS divides edit RMS by the parent's RMS under the same measure. The text/Gaussian comparison therefore includes differences in both numerator and denominator; it is not an estimate of absolute edit-energy inflation. The empirical reconstruction independently reproduces the earlier saved edit errors within 3.3e-17 absolute error.

The diagnostic criterion of interference above two is met on text for seed 1002, but not seed 1001, and never under the Gaussian. Both seeds nevertheless show more than twofold relative-error transfer gaps. Thus interference alone is not a sufficient explanation of the failure.

This concerns second-stage approximation to the frozen CP programs for the selected MLP16→MLP17 quartic path and its 16 fixed output coordinates. It does not show that a new objective will fix their native reconstruction failures. Nor does it directly decompose the earlier finite-response error: it diagnoses value-edit energy, while finite differences have their own measure and cross terms.

The connection to the covariance discussion is concrete. For a quartic function, squared error depends on moments up to degree eight. Matching raw input mean and covariance does not match that functional metric unless the Gaussian assumption is appropriate. A joint Gaussian edit score can be exact for its assumed measure and still understate text error.

Next priority remains interpreting the queued native residual-learning and downstream removal-geometry experiments. Any later graph search should compare joint edit scores and finite-response fidelity rather than treating low local Gaussian edit cost as sufficient evidence of preservation.

[Protocol](QUADRATIC_EDIT_INTERFERENCE_PLAN_V1.md) · [Numerical results](QUADRATIC_EDIT_INTERFERENCE_V1.json) · [Executable audit](audit_quadratic_edit_interference.py).
