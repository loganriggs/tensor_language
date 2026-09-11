# Shared-reader groups: exact updates and an executed optimization failure

11 September 2026,13:42 UTC. This implements a new multi-group tool using existing one-group partner mathematics. It does not claim a new native discovery. The group reads one shared feature, multiplies it by a low-rank partner map, and writes multiple output directions:

$$
F(x)=\sum_j(a_j^\top x)M_jx,\qquad M_j=U_jV_j^\top,\quad\operatorname{rank}(M_j)\le r.
$$

Here output coordinates already include the full unembedding metric. Groups may overlap; no cross-group orthogonality is imposed. This is a proposed interface for shared computation, not a demonstrated semantic partition. Unit readers fix one scale ambiguity; partner basis changes remain a gauge.

## Exact conditional reader update

For a residual tensor E, fixed partner M and unit a, the variable part of squared coefficient error is

$$
f(a)=\tfrac12a^\top M^\top Ma-2s^\top a,\qquad s_i=\sum_{o,k}E_{oik}M_{ok}.
$$

The omitted constant includes one half of the squared Frobenius norm of M. In the signed product representation of E, s is computed by contractions of existing factors. For Q=M^TM of rank less than d, solve

$$
(Q+\lambda I)a=2s,\quad\|a\|=1,\quad\lambda\ge0.
$$

A thin QR and small eigendecomposition expose the positive eigenspace. In the regular case, bisection solves the scalar norm equation. When the force lies in the range and its pseudoinverse solution has norm at most one, lambda=0 and a deterministic nullspace component fills the remaining unit norm. For any unit z,

$$
f(z)-f(a)=\tfrac12(z-a)^\top(Q+\lambda I)(z-a)\ge0.
$$

This is a global *conditional* certificate. Floating-point tests replay it with the original factors; tiny numerical eigenvalues are truncated, so this is not interval-certified arithmetic. For fixed a, the existing transformed-SVD rank-constrained partner solve is reused. Exact updates to both blocks do not imply joint/global convergence.

[Implementation](shared_reader_conditional_v1.py) and [controls](SHARED_READER_CONDITIONAL_V1_CONTROL.json) agree with independent dense contractions/autograd to3.4e-16. Near-initialized planted one/two-group examples recover to1e-12 in6/42sweeps. Random starts leave14.35%/2.14%error after500sweeps; these descriptive misses remain recorded.

## Initialization and joint optimization are substantive limitations

On an independent two-group planted problem, spectral initialization from the leading eigenvectors of sum_o T_o^2 does not recover: alternating updates leave3.07%error and reduced-objective Riemannian conjugate gradient leaves5.04%. A random start with that same reduced optimizer recovers to9.44e-13error and matches both planted group functions at cosine>.99999999999. Thus the data contain the structure and the family represents it, while a plausible initialization fails. [Comparison](SHARED_READER_INITIALIZATION_V1_AUDIT.json).

A separate joint trust-region least-squares fit reproduces the spectral endpoint exactly and allows all factors to move together. It uses a dense autograd Jacobian only on this small problem; that Jacobian is not proposed for the native model. After1000functionevaluations it remains unconverged at4.823%error. The two physical group energies grow from3.545times target energy in aggregate to**1,242,181.8times**, almost canceled by their cross term. The hypothesis that both initial and final energies exceed10times fails because the initial value does not. The final runaway cancellation is directly observed. [Joint fit](SHARED_READER_JOINT_POLISH_V1_AUDIT.json), [saved-state energy identity](SHARED_READER_TOY_CANCELLATION_V1_AUDIT.json).

## Penalize whole computations, not arbitrary internal coordinates

For an arbitrary a and partner M, a group's exact squared coefficient norm is

$$
\|G\|_F^2=\tfrac12\|a\|^2\|M\|_F^2+\tfrac12\|Ma\|^2.
$$

The new chunked objective uses

$$
\mathcal L=\frac{\|T-\sum_jG_j\|_F^2+\eta\sum_j\|G_j\|_F^2}{\|T\|_F^2}.
$$

It reuses the exact CP loss/gradient kernel, accumulates tied-reader gradients, and adds analytic whole-group gradients. Replacing U,V with U H^{-1},V H^T leaves M unchanged for invertible H; the penalty is invariant. Penalizing each internal product separately generally lacks that invariance. Dense gradient and nonorthogonal partner-gauge controls agree within3.28e-16. [Objective](shared_reader_group_objective_v1.py), [controls](SHARED_READER_GROUP_OBJECTIVE_V1_CONTROL.json).

For positive eta and a nonincreasing objective, summed physical group energy is bounded by initial loss divided by eta. With unit a this also bounds M because group energy is at least one half of its squared norm. This controls physical cancellation even though factor-coordinate gauges can remain poorly conditioned. It does not select the globally best basin or establish uniqueness.

An executed L-BFGS-B fit from the saved stalled *initial* groups, eta=.01, max2000iterations, maxcor20, maxls30, ftol0 and gtol1e-9, reaches gradient-infinity norm1.59e-8 (passes the registered1e-7bar). Its group energy is1.524 and residual5.513%; recovery fails. Its objective.07037 is far above the known feasible planted objective.01034. Bounded cancellation and local stationarity therefore do not establish successful discovery. [Receipt](SHARED_READER_GROUP_PENALTY_V1_AUDIT.json).

The next optimization decision must use genuinely different initializations and select by the weight-only objective, scoring recovery separately on planted cases. No identical stalled polish or native multi-group fit has been queued. The full native-sized coefficient objective is implemented; a robust native optimization schedule is not yet established. No new text fitting, behavioral circuit, extraction, selective-removal or OOD claim follows from these controls.

## Executed restart check — 13:44

Eight independent random initializations at the same eta=.01 give three low-objective recoveries, below the preregistered four-of-eight bar. Selecting solely by penalized weight objective chooses seed1501: relative coefficient error**0.0001604**, matched individual-group cosine**0.999752**, groupenergy1.00077. Numerical/dense replay and selected-group recovery pass; recovery-rate prediction fails. The selected solver terminates ABNORMAL despite a gradient-infinity norm1.11e-8; preserve that termination and do not infer global convergence. Other starts settle at materially worse functions, including stationary ones. This directly demonstrates the value and limits of restarts in the proposed structural family. It supplies a usable candidate initialization strategy, not a native recovery result. [All eight starts and exact optimizer settings in source](shared_reader_restarts_v1_audit.py), [receipt](SHARED_READER_RESTARTS_V1_AUDIT.json).
