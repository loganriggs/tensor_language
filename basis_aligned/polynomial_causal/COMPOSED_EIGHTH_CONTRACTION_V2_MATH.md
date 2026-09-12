# Exact repeated-input contraction through the deeper bilinear path

12 September 2026. The degree-balanced formal-source fits now converge reproducibly but fail joint native intervention fidelity. Their bounded rotation audit finds no material escape. This note develops a different mathematical object rather than treating the failed fit as evidence that no structure exists.

## The actual homogeneous path

For normalized MLP15 input $x\in\mathbb R^{1152}$, define the nonconstant producer

$$
m(x)=\lambda D_{15}\big[(L_{15}x)\odot(R_{15}x)\big].
$$

The three frozen forms $A_0,A_3,A_8$ already fold MLP16 into the selected shared-parent and partner computations. Their pure-producer branch numerators are

$$
q_j(x)=m(x)^TA_jm(x),\qquad
f_j(x)=q_0(x)q_j(x),\quad j\in\{3,8\}.
$$

Each $q_j$ has degree four and each $f_j$ degree eight in the **same** input. Residual writers and the full unembedding act on the two branch coefficients afterward. This is an exact description of these two selected homogeneous paths, not a factorization of every native output computation. Producer bias, mixed background paths and RMS denominators remain outside this numerator audit; the prior source tests show mixed paths cannot simply be discarded in circuit execution.

## Contraction without storing an eighth-order tensor

Polarize the producer into a symmetric bilinear map:

$$
M(u,v)=\frac\lambda2D_{15}\big[(L_{15}u)\odot(R_{15}v)
+(L_{15}v)\odot(R_{15}u)\big].
$$

For four vectors, the symmetric reader contraction is

$$
Q_j(a,b,c,d)=\frac13\big[
M(a,b)^TA_jM(c,d)+M(a,c)^TA_jM(b,d)+M(a,d)^TA_jM(b,c)
\big].
$$

For eight input slots, the fully symmetric branch contraction is

$$
T_j(x_1,\ldots,x_8)=\frac1{70}
\sum_{\substack{S\subseteq\{1,\ldots,8\}\\|S|=4}}
Q_0(x_S)Q_j(x_{S^c}).
$$

Order within each four-element subset does not matter. The implementation computes the28distinct producer pairs once, applies all three matrices to these pair values, then assembles70subset contractions. Setting all eight vectors equal gives $T_j(x,\ldots,x)=f_j(x)$. No $1152^8$ tensor is allocated. A projected producer uses the same oracle with its projection folded into $D_{15}$, so gradients can reach the proposed upstream interface.

## Controls and the corrected symmetry assumption

The first implementation passed dense coefficient, diagonal and slot-permutation tests. Its unrestricted reader-matrix gradient comparison failed by24.6%, because the pairing formula assumed symmetric $A_j$ while the ambient gradient test also perturbed antisymmetric entries. That failure is retained in [the V1 receipt](COMPOSED_EIGHTH_CONTRACTION_V1_FAILED_CONTROL.json).

V2 explicitly applies $A_j\leftarrow(A_j+A_j^T)/2$. Its [control](COMPOSED_EIGHTH_CONTRACTION_V2_CONTROL.json) passes dense eighth-order contraction, diagonal, permutation, reader-gradient and producer-gradient tests, with maximum relative error below3.34e-15. Thus the fixed symmetric native computation is unchanged, while the gradient extension now matches the underlying quadratic function. The corrected implementation is frozen separately from V1.

## Measuring the actual coefficient norm from weights

If the eight vectors are independent with identity covariance, multilinearity gives

$$
\mathbb E\left[\left\|\sum_jUw_jT_j(X_1,\ldots,X_8)\right\|^2\right]
=\left\|\sum_jUw_j\otimes T_j\right\|_F^2.
$$

Independent Rademacher coordinates satisfy this condition. They are synthetic coefficient probes, not text inputs or samples from the model's training distribution. Using independent vectors here measures the symmetric tensor's coefficients; it does not replace the model's repeated input. The tensor itself was constructed so its diagonal is the original repeated-input polynomial. Evaluating $f_j(X)$ on one repeated random vector instead would yield a different moment-weighted norm.

The [registered native audit](COMPOSED_EIGHTH_NATIVE_V1_PREREGISTRATION.md) uses2048probe tuples and full-U writer Gram, with paired error/reference energies and uncertainty estimates. It compares the actual fully symmetrized eighth-degree error of each frozen converged interface with the formal independent-producer degree-four prediction. The former contracts through the actual repeated MLP15 inputs; the latter only uses the producer coefficient Gram as an independent-coordinate metric. They need not agree.

A resolved difference would motivate testing the actual deeper metric; agreement would narrow that explanation for this pure path. Neither result settles the mixed paths, the correct weighting of natural inputs, arbitrary decomposition complexity, or the four circuit properties. Inspect the native result receipt for the measured outcome before treating this as a completed metric comparison.


## Native result and uncertainty (08:57)

The [native audit](COMPOSED_EIGHTH_NATIVE_V1_RESULT.json) completes in3.345seconds with all2048probes; A/Cpass,Bfails. Native diagonal/permutation errors are below1.79e-15. The actual eighth-coefficient relative errors are0.857624/0.857606, versus formal predictions0.870132/0.870132. Estimated standard errors are about0.00769. The1.25percentage-point gap does not meet the registered material-difference criterion.

The executed [paired bootstrap and leave-batch-out audit](COMPOSED_EIGHTH_NATIVE_V1_UNCERTAINTY.json) retains this conclusion. The first arm's95%Monte Carlo bootstrap interval is[0.84327,0.87300], and deleting any64-probe batch gives[0.85541,0.86129]. The second arm is nearly identical. These intervals concern synthetic coefficient-probe uncertainty, not OOD document uncertainty. They do not prove that optimizing the two metrics would choose the same frames.

Absolute coefficient scale does change substantially: the actual symmetric squared norm is about0.010738times the paired-coordinate squared norm. There is an exact accounting interpretation. Before full eight-slot symmetrization, the paired coefficient tensor $C$ is symmetric within its four producer pairs and under permutations of those pairs. Full symmetrization averages over105pairing cosets. Since symmetrization is an orthogonal projector,

$$
\frac{\|\operatorname{Sym}_8 C\|^2}{\|C\|^2}
=\frac{1+\sum_{\pi\ne I}\langle C,\pi C\rangle/\|C\|^2}{105}.
$$

The measured norm ratio therefore implies a mean normalized overlap of0.001226across the104other pairing cosets, with probe-bootstrap interval[0.000641,0.001869]. This is an average signed overlap; large individual terms could still cancel. The large scale change by itself does not explain the failed native circuit fidelity, and here the relative approximation error changes only modestly.

The useful deliverable is an exact, differentiable deeper-path contraction with a cheap native coefficient estimator. The current frames remain poor approximations in both coefficient metrics, and their original intervention misses remain. Purepath agreement does not settle the mixed background terms; no circuit promotion or general negative structural claim follows.
