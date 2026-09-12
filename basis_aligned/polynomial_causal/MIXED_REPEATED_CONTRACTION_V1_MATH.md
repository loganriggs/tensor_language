# Exact coefficients after substituting the upstream bilinear producer

The exact repeated-input audit now covers every mixture of the remaining
background and the MLP15 producer. On the two frozen, locally converged common
interfaces, the new metric does not resolve a large error difference from the
formal independent-producer metric. This is a fixed-candidate comparison: it
does not establish that optimizing the two metrics finds the same factors.

## Object and exact contraction

The selected downstream computation has a common quadratic factor and two
private quadratic factors. Its two scalar outputs are

$$
f_j(b,x)=q_0(b+m(x))q_j(b+m(x)),\qquad
q_j(y)=y^T A_j y,\qquad
m(x)=\lambda D[(Lx)\odot(Rx)].
$$

Here $x$ is the normalized MLP15 input, $b$ is a declared remaining-background
port, and $\lambda$ is the actual residual re-entry gain. The scaled producer
bias is absorbed into $b$. The matrices $A_j$ come from the existing exact
packed downstream fold. Their two output writers are measured through the full
unembedding Gram $W^T U^T U W$. Actual normalization and the dependence of $b$
on earlier inputs remain external: this polynomial is not the full model.

Define the symmetric bilinear producer and four-input downstream coefficient:

$$
M(u,v)=\frac{\lambda}{2}D[(Lu)\odot(Rv)+(Lv)\odot(Ru)],
$$

$$
C_j(v_1,v_2,v_3,v_4)=\frac16
\sum_{\substack{S\subset\{1,2,3,4\}\\|S|=2}}
(v_{S_1}^TA_0v_{S_2})(v_{S^c_1}^TA_jv_{S^c_2}).
$$

Producer degree $k$ means using $k$ copies of $m$, hence $2k$ copies of $x$,
and $4-k$ background inputs. Its coefficient tensor is separately symmetric
in these two kinds of input. Let $\mathcal M_{2k}$ be the perfect matchings of
the $2k$ producer-input slots. The exact contraction is

$$
T_{j,k}(b_1,\ldots,b_{4-k};x_1,\ldots,x_{2k})
=\frac{\binom4k}{|\mathcal M_{2k}|}
\sum_{\pi\in\mathcal M_{2k}}
C_j(b_1,\ldots,b_{4-k},M(x_{\pi_1},x_{\pi_2}),\ldots).
$$

The numbers of matchings are $1,1,3,15,105$ for $k=0,1,2,3,4$; the empty
matching counts once. On the diagonal, summing these five contractions gives
$f_j(b,x)$ exactly. Total polynomial degrees run from four to eight.
The implementation reuses all producer pairs and six quadratic splits instead
of materializing any of these high-order tensors.

Independent Rademacher vectors in every tensor slot give an unbiased estimator
of its squared coefficient norm, because their second moments are identity.
They are synthetic probes of the weights, not samples of model activations.
This differs from evaluating a polynomial on one isotropic vector in all slots,
which would introduce higher-moment trace terms.

## Controls and native result

[Dense controls](MIXED_REPEATED_CONTRACTION_V1_CONTROL.json) check every degree,
permutation invariance, summed diagonal equality, and gradients through both
producer and quadratic matrices. Maximum discrepancy is below $5.24\times10^{-15}$.
The [native audit](MIXED_REPEATED_NATIVE_V1_RESULT.json) independently reproduces
the existing five-sector expansion and the earlier eighth-degree oracle within
$1.78\times10^{-15}$. It takes 6.71 seconds with 2048 probes per degree.

The table shows arm0; arm1 is nearly identical. Errors are square roots of
relative squared coefficient error, not behavioral errors.

| Producer degree | Exact repeated-input error | Bootstrap 95% interval | Formal metric error |
|---|---:|---:|---:|
| 1 | 0.7181 | [0.6979, 0.7369] | 0.7255 |
| 2 | 0.8220 | [0.8053, 0.8393] | 0.8290 |
| 3 | 0.8798 | [0.8653, 0.8958] | 0.8747 |
| 4 | 0.8763 | [0.8605, 0.8920] | 0.8701 |

The registered material-gap test for mixed degrees2/3 fails. The one-producer
internal control and precision/runtime criteria pass. [Paired bootstrap and
leave-batch-out analysis](MIXED_REPEATED_NATIVE_V1_UNCERTAINTY.json) do not change
this verdict. These intervals describe probe uncertainty, not uncertainty over
documents or circuit generalization. The earlier pure eighth-degree estimate
0.8576 used different probes; both receipts remain valid noisy estimates.

## Symmetrization changes scale much more than this candidate's relative error

When comparing raw norms, the grouped polynomial coefficient includes
$\binom4k$. Its squared norm before additional producer-input symmetrization
equals $\binom4k$ times the corresponding formal slot-count grade norm.
The factor is essential for raw norms and cancels from relative error ratios.
After this correction, symmetric/paired squared-norm ratios are
1.0401, 0.3244, 0.06761 and 0.009574 for degrees1–4. The first is consistent
with its exact value1 within sampling uncertainty. The others are close to
the inverse numbers of pairings.

For a tensor $C$ already invariant within each producer pair and across pairs,
write $N=(2k-1)!!$. Orthogonal symmetrization gives

$$
\frac{\|\operatorname{Sym}C\|^2}{\|C\|^2}
=\frac{1+(N-1)\bar\rho}{N},
$$

where $\bar\rho$ is the average normalized signed overlap with the other
pairing representatives. Estimated averages are -0.01342, 0.001015 and
0.0000510 for degrees2–4. A small signed average does not imply every pairing
is independent; positive and negative individual overlaps could cancel.

## Decision and next discriminating test

The common-interface family is now locally converged and highly reproducible,
yet fails joint native intervention fidelity. Fixed-frame repeated-input
rescoring has not repaired that failure. It remains possible that the exact
metric has a different useful gradient. The [registered direction pilot](REPEATED_METRIC_DIRECTION_V1_PREREGISTRATION.md)
tests two independent synthetic-probe gradients and validates fixed steps on
separate probes. This directly checks feasibility before committing to a long
new fit. No OOD, extraction, selective removal, or composition criterion is
newly established by these coefficient audits.

## Direction pilot: the naive stochastic gradient is too noisy at this budget

The [native pilot](REPEATED_METRIC_DIRECTION_V1_RESULT.json) completed in14.60
seconds. Directional finite differences agree within $7.45\times10^{-10}$ and
orthogonality error is $1.10\times10^{-14}$. Two independent512-probe-per-degree
gradient estimates have norms0.8545/0.8701 but cosine only0.00877, failing the
registered0.5 threshold. On separate2048-probe-per-degree validation, the three
steps change balanced squared loss as follows; positive improvement is better.

| Tangent step length | Held-out improvement | Paired standard error |
|---|---:|---:|
| 0.01 | -0.0000156 | 0.0000283 |
| 0.05 | -0.0001407 | 0.0001413 |
| 0.15 | -0.0008879 | 0.0004232 |

No step passes the improvement criterion. The initial held-out loss is0.68316.
Reference-energy normalization uses fixed independent probe estimates; these
standard errors are conditional on those normalization weights. This pilot
does not show that the exact objective is stationary: a real small gradient
could be hidden by the estimator noise.

The executed [CPU noise audit](REPEATED_METRIC_DIRECTION_V1_NOISE_AUDIT.json)
uses independent estimates $g_1,g_2$ of a common mean gradient $g$. Under this
sampling design,

$$
\mathbb E\langle g_1,g_2\rangle=\|g\|^2,\qquad
\mathbb E\frac{\|g_1-g_2\|^2}{2}
=\mathbb E\|g_1-g\|^2.
$$

The observed signal/noise squared estimates are0.00652 and0.73706. A crude
inverse-sample-count extrapolation would require about58,000 probes per degree
for an expected-cosine proxy of0.5, or521,000 for0.9. These are unstable plug-in
estimates from only two replicas, not required sample sizes or guarantees.
They explain why merely taking many optimization steps at512 probes is poorly
motivated. The completed pilot changes the next choice to estimator variance
reduction or a deliberate higher-accuracy gradient audit, before a long fit.
It does not rule out alternative sparse/block/DAG representations.

Literal-count clarification: the preregistration calls4096 probe cases “total gradient contractions.” There are4096 candidate-gradient probe cases and4096 reference probe cases, hence8192 contractions when both are counted. Validation has8192 probe cases, each with one reference and four candidate/start evaluations. The executed schedule matches the stated seeds, batch counts and sizes; no body forwards are used.

## Analytic pairing correction: partial noise reduction, still no usable direction

For each degree, retain the separate pairing error contractions $e_\pi$ before
averaging. Write $N$ for the number of pairings, $G=W^TU^TUW$, and $F_k,R_k(P)$
for the exact formal full/retained squared norms. The new unbiased estimator is

$$
\left\|\frac1N\sum_\pi e_\pi\right\|_G^2
-\frac1{N^2}\sum_\pi\|e_\pi\|_G^2
+\frac{\binom4k}{N}\bigl(F_k-R_k(P)\bigr).
$$

It replaces the sampled same-pairing contributions with their exact expected
value, leaving cross-pairing terms sampled. A *control variate* is such a
zero-mean correction to an estimator; unbiasedness alone does not guarantee
lower variance. Degree1 becomes fully analytic. The exact last term assumes
$P^TP=I$: its derivative is valid along the orthonormal-frame manifold, so
native finite differences use QR retraction rather than unconstrained steps.

[Exhaustive toy coefficient and tangent-gradient checks](MIXED_PAIRING_CONTROL_VARIATE_V1_CONTROL.json)
pass within $3.62\times10^{-14}$. The [matched native pilot](PAIRING_VARIANCE_DIRECTION_V1_RESULT.json)
uses the same training probe seeds, frame and validation budget as the naive
pilot. It completes in14.83 seconds; the analytic derivative check is within
$5.78\times10^{-8}$. The squared difference between gradient replicas decreases
to72.46%of the old value, a27.54%observed noise reduction. However, gradient cosine
is only0.00333, and all three independent validation improvements remain
negative (-0.0000154, -0.0001430, -0.0009177). Instrument criteria pass; the
registered reliability and improvement criteria fail. These two replicas do
not establish a population variance-reduction percentage.

The strongest immediate alternative explanation is a poor scalar control
coefficient. The [executed paired CPU audit](PAIRING_VARIANCE_DIRECTION_V1_COEFFICIENT_AUDIT.json)
minimizes the observed replica-difference norm over that scalar. The best value
is0.9883, almost the implemented1. Its residual ratio is0.72455 versus0.72459.
Thus scalar tuning offers negligible repair of this observed discrepancy.
This descriptive minimum is not an unbiased estimate of future variance and
does not justify selecting a coefficient without fresh replication.

The analytic term's tangent gradient norm is only $6.16\times10^{-5}$ here;
the sampled residual dominates. The next meaningful estimator choice concerns
cross-pairing contractions or a substantially larger deliberate probe budget.
Blindly extending this small-batch fit is not supported. Nothing here rules
out shared structure, establishes exact stationarity, or repairs the native
extraction/removal/composition failures.

## One-slot exact integration: controlled alternative, native price still open

While the larger independent-gradient audit runs, a second estimator has passed
[CPU controls](INTEGRATED_COEFFICIENT_SLOT_V1_CONTROL.json). Fix all coefficient
slots except one. Multilinearity makes the two-output error a linear map
$e(z)=Jz$, with $J\in\mathbb R^{2\times1152}$. Therefore

$$
\mathbb E_z[e(z)^TGe(z)\mid\text{other slots}]
=\operatorname{tr}(J^TGJ),\qquad \mathbb E[zz^T]=I.
$$

Two reverse derivatives recover the two rows of $J$ without constructing the
whole high-order tensor. Differentiating the trace through those derivatives
gives a parameter-gradient estimator with this slot integrated exactly. Under
the finite moments here, differentiation and expectation commute. The law of
total covariance then guarantees that exact conditional integration cannot
increase gradient covariance **per independent draw of the remaining slots**.
It does not guarantee lower variance per second: second derivatives through
the contraction cost more than ordinary sampling.

This is closely related to PSD trace estimation. [Meyer et al., Hutch++](https://arxiv.org/abs/2010.09649)
reduce stochastic trace variance by computing a low-rank part explicitly and
sampling a remainder. Our conditional matrix is $J^TGJ$, PSD with rank at most2,
and we can access its two-factor rows directly. Thus we use an exact trace,
not Hutch++ or its matrix-query complexity guarantee. The other polynomial
slots remain random; this does not integrate the entire eighth-degree norm.

Exhausting all four Rademacher choices of a two-dimensional last slot agrees
with the implementation's values and parameter gradients within
$6.10\times10^{-16}$ across all four producer degrees. The finite toy design
retains46–49%of the ordinary gradient variance. Those percentages are not
native-model predictions. Native timing and variance need a separate test;
the running larger-probe audit has not been changed to use this method.

## 65,536-probe audit: no resolved useful direction at this frame

The [larger native audit](ACCURATE_REPEATED_DIRECTION_V1_RESULT.json) completed
in436.28 seconds. Each of two independent replicas uses65,536 probes per
producer degree, organized into16 independent blocks. Degree1 is analytic.
The existing instrument checks pass. Gradient agreement does not improve:

| Probes per degree per replica | Gradient cosine |
|---|---:|
| 4,096 | -0.00266 |
| 16,384 | 0.000215 |
| 65,536 | -0.000844 |

Final gradient norms are0.06539/0.06578. Within-replica block estimates of
squared noise in those means are0.004328/0.004321, accounting for essentially
their entire observed squared norms0.004276/0.004326.

The independent16,384-probe-per-degree validation starts at balanced squared
loss0.684653. Positive improvement would favor the proposed step:

| Tangent step length | Held-out improvement | Paired standard error |
|---|---:|---:|
| 0.05 | -0.0000484 | 0.0000512 |
| 0.15 | -0.0006472 | 0.0001534 |
| 0.5 | -0.0076804 | 0.0005046 |

Both gradient-reliability and improvement criteria fail. The
[executed block audit](ACCURATE_REPEATED_DIRECTION_V1_BLOCK_AUDIT.json) leaves
each64-probe validation batch out in turn; every resulting mean improvement
remains negative. No candidate is adopted.

For independent block gradients $g_i,h_j$, the average cross inner product
is an unbiased estimate of the squared population mean gradient. It is
$-3.63\times10^{-6}$ here; the negative value is retained rather than silently
clipped. A two-way row/column/interaction variance calculation gives an
estimated standard error $2.86\times10^{-5}$. An
[exhaustive finite-distribution control](GRADIENT_BLOCK_VARIANCE_FORMULA_V1_CONTROL.json)
verifies that variance formula in both zero- and nonzero-mean examples. This
does not supply a Gaussian confidence guarantee or prove a zero true gradient.

**Decision:** stop increasing this estimator's budget at the same formal-fit
frame for now. The much larger audit finds no repeatable useful descent and
does not support a long exact-metric fit initialized here. The newly controlled
single-slot integration method remains available but has not earned another
native run solely to refine this same null. The next structural comparison
should change the representation rather than repeat this local search.
Alternative stationary points and different sparse/block/DAG representations
remain open. Existing native intervention failures are unchanged; no new OOD,
extraction, selective-removal, or reuse property is established.
