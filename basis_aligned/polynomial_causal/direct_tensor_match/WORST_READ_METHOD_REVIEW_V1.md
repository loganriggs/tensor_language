**Change the question from average error to bounded-input source error**

Review on21September2026 following the consistent-residual screen. This is an additional focused literature/method review; the last scheduled three-hour review remains13:50UTC.

The original paper defines a weight-tensor metric and requires compatible structure for efficient recursion. Its normalized similarity identifies proportional functions, so amplitude-preserving reconstruction still needs a difference norm. Gaussian moments are one geometry, not a guarantee of arbitrary-distribution accuracy. We retain these distinctions in our losses. [Nissen Gonzalez et al., sections2.2–2.3](https://arxiv.org/html/2605.15183v1#S2.SS2).

Prior joint-quartic coefficient fitting improved tensor error while worsening native behavior; matched-Gaussian and empirical-moment follow-ups also failed transfer. The new homogeneous cubic norm does not discriminate the code failure either. Another average-metric sweep is therefore lower priority than a global error-direction test.

For a fixed quadratic read, the full approximation error is

$$
e(z)=z^\top E z+\ell^\top z+b.
$$

RMS-normalized1152-dimensional inputs satisfy $\|z\|_2\le\sqrt{1152}$ in real arithmetic. We compute both extrema over that ball. A minimizer has a nonnegative multiplier $\lambda$ satisfying

$$
(E+\lambda I)z=-\ell/2,\qquad E+\lambda I\succeq0,
\qquad\lambda(\|z\|_2^2-1152)=0.
$$

This maps directly to the trust-region subproblem with $A=2E$, $g=\ell$, $B=I$, radius $\sqrt{1152}$. The necessary-and-sufficient conditions apply to symmetric indefinite matrices and include singular hard cases. We use an eigendecomposition and secular root, not the paper's generalized-eigenvalue implementation. Dense complexity is cubic in input dimension; matrix-free alternatives exist but are unnecessary for this bounded screen. [Adachi et al., Theorem1.1](https://www.opt.mist.i.u-tokyo.ac.jp/~iwata/papers/TRS.pdf).

Six known-optimum cases and rotated equivalents pass: positive-definite interior, singular interior, pure linear, easy indefinite, zero-linear hard case and nonzero-linear hard case. Every native extremum passes stationarity, feasibility, complementarity and numerical dual-gap checks. These are floating-point certificates, not interval proofs.

**Native finding**

Including every affine correction, the graph is within1.10times the covariance baseline on all six worst-read errors, but exceeds1.10times the isotropic baseline on all six. For the second read of component three, the graph's worst absolute error is24.15% of the teacher's maximum absolute output over the same ball; the covariance baseline is26.09% and isotropic baseline17.72%. Thus graph error is1.362times the isotropic baseline. This ratio is not pointwise relative error, and adversarial ball directions need not correspond to realizable text.

This identifies a weight-derived robustness gap hidden by the covariance ranking. It does not prove that the worst direction causes the observed code failures, or bound the final normalized component without additional sensitivity analysis.

**Concrete successor underway**

A fixed graph has32existing correction squares available to its third component. Changing only their second-read coefficients preserves directions, products, sharing and representation size. At each fixed input, read error is affine in those coefficients; squared normalized error is convex quadratic. Original coefficient/value/Jacobian constraints are also convex quadratic with the first read held fixed. Exact ball extrema can therefore supply adversarial cuts to a finite convex minimax fit.

The conditional8-constraint model has been implemented and checked against the original dense evaluator at zero and two perturbations. The registered next optimization uses at most8separation rounds and retains original fidelity limits. Any generated-interface regressions are audited independently; a local success still requires future untouched transfer and intervention evidence. A failure restricts this fixed dictionary and edit, not general shared-DAG expressivity.

[Worst-read results](NATIVE_WORST_READ_V1.json) · [Robust-fit plan](ROBUST_FIXED_READ_PLAN_V1.json) · [Conditional replay controls](ROBUST_FIXED_READ_CONTROL_V1.json).
