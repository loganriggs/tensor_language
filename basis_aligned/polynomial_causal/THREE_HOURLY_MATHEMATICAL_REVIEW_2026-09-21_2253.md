**Three-hour mathematical review — 21 September 2026, 22:53 UTC**

ACTIVE_TRACK: WEIGHT_FOLDING. Review started at the three-hour boundary after19:53; source checks and executable controls completed during this review. The user's two-day weights-first focus lasts until22September15:10. Full success still requires predictive/OOD, extractable, selectively manipulable, reusable, stable and simpler circuits. No new program meets that standard.

**Object and change of evidence**

The latest target is a homogeneous quartic F_g(x)=H_g:x^4 with x in R^1152 and16fixed vocabulary-metric output readers of the native MLP16→17 pure quartic branch. Each native MLP has4608products. The readouts are data-informed; this target excludes other output directions, residual cross terms, attention and explicit normalization/softcap. All four input slots receive the same x. Coefficient comparisons therefore use full input symmetrization even when the executable representation is asymmetric.

The tested CP512 student has512products of four linear forms,1536variable products and2385920coefficients including its fixed writer. The sparse shared hierarchy has144quadratic features,4products each,512selected feature pairs,1088products/1353728coefficients/1024indices. Pair support is fixed and seeded, not discovered. Both retain scaling/permutation and redundant-feature freedoms; similar fitted functions do not identify unique internal interventions.

Finite native contraction bounds excluded the old384-product global target. Broader models improve, but remain poor globally: CP400 has97.8%sampled coefficient/98.9%Gaussian error; shared hierarchy96.9–97.2%/91.5%. Exact Gaussian readout optimization improves CP's Gaussian error to84.7%but text error worsens to277–303%. With calibration mean and covariance, text error improves to9.6–10.3%, but coefficient error worsens to126.7%. Same-token root1response13.1–15.5%and sensitivity20.4–21.8%still fail10%; old empirical384 has15.3%response/14.5%sensitivity. Every metric and scope must remain explicit.

**Literature-to-object mappings**

The [tensor-similarity paper, sections2.2–2.3](https://arxiv.org/html/2605.15183v1#S2.SS2) places M on lifted tensor coordinates, distinguishes symmetric coefficient and Gaussian metrics, and states structural restrictions on Gram recursion. Our squared difference preserves amplitude, unlike cosine alone. A Gaussian input covariance induces an eighth-moment metric here; it is not itself the lifted M. We use explicit validated quartic contractions rather than assuming arbitrary graph topology admits the paper's recursion. Exact equality and small error under a chosen measure remain different claims.

[Mamis, generalized multivariate Stein lemma](https://arxiv.org/html/2202.00189v3) relates Gaussian expectations of a function times monomials to expectations of derivatives, assuming the derivatives and expectations exist. Our polynomial target satisfies these assumptions. Whitening and shifting x=mu+Sz gives affine CP factors; 764partial pairings compute their exact Gram. Expected derivatives through order four give the native cross. The cached mean, linear and quadratic projections avoid materializing the order-five tensor. Five independent quadrature/gradient controls pass<3.5e-14. This is an executable matching algorithm for a Gaussian reference law, not a guarantee that real activations are Gaussian or that recovered features are unique.

[O'Leary and Rust, variable projection](https://www.cs.umd.edu/users/oleary/software/varpro.pdf) matches our separation between nonlinear feature directions and linear output coefficients. Conditional on directions, the readout is a linear least-squares problem; eliminating it reduces the nonlinear search. Ridge makes our fixed-feature Gram positive definite. This supplies neither global nonlinear convergence nor monosemanticity. Planted random-start recovery improves greatly with redundant features, so optimization remains a distinct explanation for native failure.

[Boyd and Vandenberghe, Convex Optimization, sections5.2.3–5.2.4](https://stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf) maps to a useful next restriction: optimize a quadratic readout loss subject to one convex quadratic bound. The book was downloaded from the authors and its Slater/QCQP sections checked. With positive-definite regularized Grams and a positive budget, the coefficient optimum is strictly feasible; convex duality and KKT conditions apply. The zero-budget endpoint is handled directly. This guarantee holds only for fixed features.

**Exact constrained readout consequence**

Let G0,X0 be the coefficient Gram/cross and G1,X1 the shifted-Gaussian Gram/cross, with declared ridge included in each Gram. Write

$$
L_j(C)=\operatorname{tr}(C G_j C^\top)-2\langle C,X_j\rangle,
\qquad C_0=X_0G_0^{-1}.
$$

The unknown teacher norm cancels from differences. In particular,

$$
L_0(C)-L_0(C_0)=\operatorname{tr}((C-C_0)G_0(C-C_0)^\top).
$$

We can minimize L1 subject to this deterioration being at most b, instead of guessing a loss multiplier. Set G0=LL^T and Y=(C-C0)L. The constraint is ||Y||_F^2<=b. Diagonalize L^-1 G1 L^-T=E diag(nu) E^T. If B=(X1-C0G1)L^-T E, the solution satisfies

$$
YE=B\operatorname{diag}((\nu_i+\lambda)^{-1}).
$$

Choose nonnegative lambda by monotone scalar bisection, or zero if the unconstrained target fit is feasible. One eigendecomposition costs O(p^3), p512; each budget then uses O(op^2) reconstruction plus cheap scalar iterations, o16. The regularized coefficient optimum is unique; semantic decomposition is not. Budgets will be expressed relative to captured coefficient score, explicitly not as relative total tensor error. Report the unregularized coefficient change too.

Actual CPU implementation `coefficient_guarded_readout.py` was compared against independent SciPy SLSQP on25cases including ill-conditioned and redundant designs. Maximum objective difference1.3e-12, coefficient difference2.1e-6; feasibility, monotonicity and stationarity checks pass. This is a concrete consequence of the review, not merely a literature analogy.

**Decision, alternatives and circuit handoff**

Next map the exact fixed-feature tradeoff under a coefficient budget, retaining isotropic Gaussian, text, same-token and sensitivity metrics. This determines whether the metric conflict can be resolved cheaply before spending more on nonlinear feature learning. Primary budget and tests must be registered before native evaluation. If all useful text fits violate the coefficient guard, the current dictionary or its global target needs changing; another readout-only sweep is not enough.

Alternative adaptive graph-pair selection remains relevant after the seeded shared hierarchy, but its poor absolute reconstruction means exact simplification alone cannot repair it. Wider fitting can help optimization but changes price. Sphere identities were checked against existing normalized-probe/source-moment work: for two homogeneous degree-four functions, Gaussian and uniform-sphere squared errors differ only by a radial factor; allowing mixed degrees changes that conclusion. Do not reuse a sphere identity as a new explanation for the present same-degree metric discrepancy.

The circuit-track handoff is the fixed newline-associated output reader, its same-token responses and actual normalization sensitivity. Root descriptions are constituent conditions, not semantic labels. No further causal promotion is justified by9.6%aggregate values while component responses remain above10%.

**Organization and efficiency**

Latest authoritative receipts are SPARSE_QUARTIC_BANK_NATIVE_V1, EXACT_GAUSSIAN_CP_READOUT_V1, GAUSSIAN_CP_DATA_READOUT_V1 and GAUSSIAN_CP_COMPONENT_TRANSFER_V1. The requested22:13for_Logan report remains historical; canonical interpretations will link new results. Repair the explanations index's stale latest-follow-up pointer rather than adding another unsolicited user report.

Measured native runtimes: CP400259.65s/1.50GB, shared hierarchy883.28s/13.73GB, exactGaussianreadout7.23s/1.94GB, three calibration laws17.79s/2.50GB. Fitting the hierarchy is much more expensive than running its smaller program. Cached teacher projections and fixed-feature moments now make readout questions cheap; reuse them rather than rebuilding captures or another compiler. CPU analytical/gradient controls and interpretation ran alongside the long GPU job. All GPU work used the serial managed runner; no duplicate restart after observation timeout occurred. No time-allocation percentages are inferred from incomplete timing records.
