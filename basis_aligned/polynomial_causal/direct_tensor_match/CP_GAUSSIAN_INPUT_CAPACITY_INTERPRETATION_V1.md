Gaussian functional dimension is different from coefficient-tensor dimension

22 September 2026, 00:48 UTC. Exact implicit moments for the two fitted CP512 parents under the fixed calibration mean/covariance Gaussian. This is not a result for the original native tensor, text distribution, or intervention behavior.

The previous coefficient audit showed a1.14–1.27% coefficient error floor for any256-term quartic CP representation. Under the Gaussian functional metric, the corresponding input-span obstruction is much smaller. More strongly, an unrestricted conditional-mean approximation using just256input directions has a guaranteed Gaussian RMS error below0.83–0.93%. This does not guarantee a64-term CP program or any particular arithmetic cost.

The related primary literature is Zahm, Constantine, Prieur and Marzouk, [Gradient-based dimension reduction of multivariate vector-valued functions](https://arxiv.org/pdf/1801.07922). Their framework uses a Gaussian subspace Poincare inequality to control ridge-approximation error by a gradient matrix. Here the output is our16-dimensional fixed writer-coordinate vector and x=mu+S z with z standard Gaussian. The input covariance is nonsingular and the polynomial has finite moments, so this mapping satisfies the relevant assumptions. No circuit identifiability or semantic guarantee follows from that framework.

**The finite-degree bound used here.** Define

$$
K=\mathbb E[J_zF(z)^\top J_zF(z)],
\qquad F_P=\mathbb E[F(z)\mid Pz],
$$

where P is an orthogonal rank-r projector. Let

$$
V_P=\mathbb E\|F-F_P\|^2,
\qquad D_P=\operatorname{tr}((I-P)K).
$$

For a polynomial of total degree at most four,

$$
\boxed{D_P/4\ \le V_P\ \le D_P.}
$$

Derivation: rotate coordinates so P retains the first r coordinates and expand F in orthonormal multivariate Hermite polynomials. Conditional expectation retains precisely those coefficients with no degree in the discarded coordinates. V_P is the sum of squared norms of the other coefficients. D_P weights each such squared norm by its discarded-coordinate Hermite degree, an integer between1and4. This proves both inequalities directly. The degree-dependent lower inequality is derived here for this audit; we do not attribute it to the cited paper or claim novelty for it.

If lambda_j are the descending eigenvalues of K, every function of r linear input directions has squared error at least

$$
\frac14\sum_{j>r}\lambda_j.
$$

The conditional mean on the leading r eigendirections has squared error at most the full tail. Divide by E||F||^2 and take square roots for the relative errors below. The lower bound applies to all such ridge functions; the upper bound establishes an unrestricted conditional-mean approximation, not a specified CP term count.

| Input directions r | Lower error, start1001 /1002 | Conditional-mean upper error, start1001 /1002 |
|---|---:|---:|
|16|2.938% /2.899%|5.876% /5.798%|
|32|2.258% /2.213%|4.515% /4.426%|
|64|1.632% /1.583%|3.263% /3.165%|
|128|1.038% /0.985%|2.076% /1.969%|
|256|0.462% /0.417%|0.925% /0.833%|
|512|0.107% /0.089%|0.214% /0.179%|
|1024|0.00452% /0.00366%|0.00904% /0.00733%|

A k-term quartic CP student uses at most4k input directions. The lower bound therefore applies at r=4k. It supplies only necessary CP counts:34/32terms for1% Gaussian error, far below the coefficient audit's265/261 necessary terms for1% coefficient error. Neither count is sufficient. The Gaussian upper bound at r=256 does not say64CP terms can implement that conditional mean.

**Computation and checks.** K is formed exactly from the parent's factors: sum over16 derivative-slot pairs, contracting output coefficient inner products and six-factor affine Gaussian moments, leaving the two input directions uncontracted. It is1152by1152. Its trace agrees with the independently computed derivative-feature Gram energy to1.5e-15. Five planted structural families compare the whole K against direct Jacobian quadrature and verify both conditional-variance inequalities by explicit integration over discarded coordinates. The calculations are floating-point evidence, not interval-certified bounds.

**Executable implication and limits.** A useful alternative initialization can project onto the leading Gaussian-sensitive input subspace and construct the conditional polynomial explicitly. For a product of four affine forms, integrating Gaussian discarded coordinates yields the retained four-form product, six covariance-weighted products of two retained forms, and three constant pairings. Thus the conditional expectation remains degree at most four, but generally adds lower-degree terms and additional arithmetic. Finding a small reusable graph for those terms is still a decomposition problem. This construction has not yet been implemented or evaluated on native responses.

Decision: keep both queued fits unchanged. The bound explains why the Gaussian target is not excluded by the earlier coefficient bound; it does not excuse weak intervention fidelity. A subsequent active-subspace/conditional-polynomial experiment must price the projection and correction terms and test the native interface. Do not report the upper bound as a discovered circuit or a CP compression result.

Artifacts: [numeric results and conditional-integration controls](CP_GAUSSIAN_INPUT_CAPACITY_V1.json), [audit](audit_cp_gaussian_input_capacity.py), [exact moment implementation](gaussian_cp_derivative_gram.py), [coefficient comparison](CP_PARENT_INPUT_CAPACITY_INTERPRETATION_V1.md).
