# Projected LL1 convergence and group-stability comparison

The two native 64-group, rank-16 LL1 pilots stopped after 120 seconds without convergence. Their whole functions agree more than their individual groups. This run changes the optimization method: eliminate all output vectors exactly at every input-factor step, rather than jointly stepping those outputs with L-BFGS. The family, full-unembedding coefficient objective, and 0.01 whole-group energy penalty remain fixed.

For group Gram matrix G, positive group norms define D = diag(sqrt(diag(G))). Solve the equivalent normalized system

$$
(D^{-1}GD^{-1}+0.01I)Z=D^{-1}H^T,\qquad C=D^{-1}Z.
$$

The first term is a positive-semidefinite correlation matrix with trace 64. Thus the system condition number is at most 6,401 in exact arithmetic. This improves numerical scaling; it does not change the conditional optimum or guarantee a good nonconvex basin. Zero/invalid group norms abort rather than receive an unregistered regularizer.

Two starts: the saved spectral and native-initialized LL1 pilots. Each receives at most 1,200 soft seconds and 10,000 L-BFGS iterations, with history size 20, max line-search steps 40, ftol 0, gtol 1e-9. Reuse cached objective evaluations in callbacks. Local stopping requires both maximum gradient in the documented normalized packed coordinates <=1e-7 and absolute objective change across the last 20 accepted steps divided by coefficient capture <=1e-6. Other solver terminations and time limits are reported separately. No global-minimum claim.

Registered predictions:

- A: native directional finite-difference error <=1e-6; executable, reduced-identity, and normalized output-system residual <=1e-8; accepted objective increases <=1e-10. The initial exact output solve must not increase the previous joint penalized objective by more than 1e-8. The condition bound is checked.
- B: both starts meet the stated local gradient and progress criteria.
- C: final whole-function cosine >=0.95 and at least 16 of 64 signed whole-group Hungarian matches have cosine >=0.8. Group matching is independent of arbitrary input/output factor gauges.

Report coefficient capture, optimization cost, gradient/progress, all missed predictions, and the complete group matching. No data/CE fitting, corpus access, or model-body forwards. A failed convergence/stability test limits this solver and returned representation; it is not evidence that structure is absent. Unlike the uninformative near-identical-square merging preflight, this experiment directly addresses a remaining optimization uncertainty before using the group geometry to propose shared computations.

Literal price remains 1,254,400 learned coefficients for the 64 rank-16 groups, plus the retained 1,152 bias values, unembedding, and native background. Eliminating outputs during fitting does not remove them from the executable price. Save both final factor sets durably. Overall runner alarm is 3,000 seconds. Only managed lane 1 may execute GPU operations, with reviewed source and bound dependencies.

This run can improve or falsify the stability of proposed reusable output groups. It does not supply OOD prediction, behavioral extraction, selective removal, or compositional circuit evidence. CPU work on shared input subspaces can continue while it runs. No automatic repetition of a failed unchanged fit.
