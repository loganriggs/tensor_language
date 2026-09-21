**Reader capacity depends strongly on the perturbation metric; uncentered weighting is dominated by the mean direction.**

For a replacement of the form $\widehat F(x)=g(B^\top x)$, all Jacobian rows lie in the column space of $B$. At fixed anchors, let

$$
G=\sum_a J_F(x_a)^\top J_F(x_a).
$$

For any rank-$r$ reader space, the relative Jacobian error is bounded below by

$$
\left(\frac{\sum_{i>r}\lambda_i(G)}{\operatorname{tr}G}\right)^{1/2}.
$$

This allows the entire reader space to rotate and allows an unrestricted downstream differentiable function. It is a necessary bound; the optimal derivative projection need not be realizable by our constrained quartic graph. The target is the complete pure quartic MLP16→MLP17→QR output, omitting residual/bias terms and intervening normalization. Sixteen anchors from each previously opened panel define the local metric.

| Perturbation geometry | Best possible rank256 error, panel0 / panel1 | Necessary readers for10% error, panel0 / panel1 |
|---|---:|---:|
| Isotropic | 19.13% / 15.45% | 707 / 572 |
| Centered covariance, eigenvalue floor0.01 | 3.76% / 2.92% | 25 / 14 |
| Centered covariance, eigenvalue floor0.10 | 3.87% / 3.01% | 25 / 14 |
| Uncentered second moment, floor0.01 | 0.72% / 0.60% | 2 / 2 |

The native isotropic integrity and rank256-insufficiency predictions PASS. Rank512-sufficiency FAILS: its floors are13.56%/10.91%. Full FP32/FP64 Jacobians agree to4.32e-7; trace/projection checks agree below2e-15. Runtime0.83s. The earlier learned256-reader space missed49.14%of isotropic Jacobian norm, so there is both an optimization gap and an unavoidable rank restriction in this particular metric.

The immediate CPU redteam changes the geometric assumption while retaining the native Jacobians. For perturbations $\delta x=Lz$, replace $G$ by $L^\top G L$ and transform the student reader directions consistently. The prediction that weighted rank256 floors are below10%on both panels PASSES. The current learned readers miss2.14%/1.99%of uncentered-weighted sensitivity; the isotropic49.14%should not be presented as their natural-response error.

A further CPU check separates covariance from the mean. In the unfloored second moment, the mean-direction term explains96.35%/95.81%of weighted derivative energy. The registered>50%prediction PASSES. Centering raises the necessary ranks substantially, although the rank256 floors remain below4%. This is not a numerical bug: $E[xx^\top]=\operatorname{Cov}(x)+\mu\mu^\top$ weights a genuine mean-direction perturbation. It is a poor basis for claiming that ordinary variation is intrinsically two-dimensional.

For the pooled isotropic Gram, at least693 readers are needed for10%local derivative error. Under the specific32-feature architecture with$k$bilinear products per feature, the reader rank is at most$64k$, giving the necessary condition$k\ge11$. Under the pooled uncentered-weighted geometry, that same argument gives only$k\ge1$. Neither is a sufficient recipe for a successful replacement. We should therefore not expand$k$solely because of the isotropic certificate, or call the uncentered rank-two result a circuit.

Frozen principal reader spaces also transfer imperfectly between panels. An isotropic rank256 space from panel0 misses30.61%of panel1 sensitivity, versus panel1's own15.45%floor. Centered-covariance rank256 transfer error is5.57%, versus2.92%for the panel1 optimum. These use opened panels, not independent OOD confirmation.

Next, distinguish reader-space error from errors in the nonlinear computation within that space under a centered, explicitly defined perturbation geometry. The fixed-feature function-value oracle still has7.58%second-panel error; low local rank alone does not repair that failure. No discovered semantic unit, selective manipulation or circuit adoption is claimed.

[Native rank bounds](QUARTIC_READER_RANK_V1.json) · [Weighted bounds](QUARTIC_WEIGHTED_READER_RANK_V1.json) · [Mean/covariance audit](READER_METRIC_MEAN_AUDIT_V1.json) · [Fixed-span sensitivity](QUARTIC_READER_SPAN_INTERPRETATION_V1.md) · [Readout-only floors](WIDE_QUARTIC_READOUT_SPAN_INTERPRETATION_V1.md).
