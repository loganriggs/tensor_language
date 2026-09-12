# Mathematical review — 12 September 2026, 20:17 UTC

The previous review was recorded around17:12. This review addresses the user's concrete solver question and the unresolved producer source-function fit, following the bilinear reconstruction handoff rather than the superseded better_math_ideas direction.

## Object and circuit decision

For each of27 producer heads in layers8/9/13, preserve the complete joint QK1 × QK2 × value numerator writing into four selected downstream readings. Query dimension1152, concatenated current/first source dimension2304, key/query width128. At a fixed relative position the target has query degree2/source degree3. Native normalizers, source-state generators, position handling and downstream background remain explicit external dependencies. We seek rank16 source cubic features with private query/output consumers, not independent QK task assignments.

A candidate bank has16 ×3 ×2304 =110592 nonlinear reader parameters. Output coefficients are solved exactly through the source Gram matrix and implicit query/output inner products. Fit coefficient Frobenius error, not text activation loss. Output elimination gives a nonlinear projection objective; no full source/query tensor or dense Jacobian is needed to evaluate it. Coefficient similarity alone cannot identify a circuit: the downstream test is frozen complete-function agreement and physical removal/interchange. The producer layer choice was behavior-informed; weights determine fitting.

## Exact coordinate change

For a nearly coincident pair of aligned cubic products write the three readers as b_j ± t d_j, t>0. Let P_±(s) be the resulting products. Their span equals that of

$$
E(s)=b_0b_1b_2+t^2(d_0d_1b_2+d_0b_1d_2+b_0d_1d_2),
$$

$$
O(s)=d_0b_1b_2+b_0d_1b_2+b_0b_1d_2+t^2d_0d_1d_2,
$$

where each b_j or d_j denotes the corresponding linear source reading. Specifically P_±=E±tO. With fixed nonzero t, (b,d) ↔ (a_+,a_−) is invertible. There are still two freely consumed source features, not eight independent factors. This exact parameterization changes conditioning but not the feasible function family. At t=0 the inverse fails; that tangent-limit model is not silently substituted.

The saved native fit's feature Gram condition number fell from8.3million to2.55 under a prior fixed rewrite. QR alone had agreed with the original objective/gradient, so simply replacing the linear solver was not a repair. The new implementation makes the rewrite differentiable and optimizable.

Scale, source-reader permutation/sign and output-feature basis gauges remain. An unconstrained parameter gradient can be artificially small through scaling. For the paired readers,

$$
g_+=(g_b+g_d/t)/2,\qquad g_-=(g_b-g_d/t)/2.
$$

For each raw reader a=nu with ||u||=1, use the unit-reader tangent gradient n(I−uu^T)g_a to score stationarity. This pullback agrees with independently differentiated reconstructed raw atoms to1.63e-15. It does not certify a global optimum or Hessian definiteness.

## Literature mapping and limits

Variable projection exactly eliminates linear consumers conditional on source factors; this is the separable nonlinear least-squares setting of [O'Leary and Rust](https://www.cs.umd.edu/users/oleary/software/varpro.pdf). Our implicit coefficient Gram supplies the needed inner products. Local differentiability requires the feature bank to maintain rank; near singularity remains a real issue. Linear elimination does not remove nonlinear local traps.

[Breiding and Vannieuwenhoven](https://arxiv.org/abs/1709.00033) combine Riemannian Gauss–Newton, trust regions and ill-conditioning-triggered restarts for small dense CP approximation. Their product-of-Segre representation and numerical results motivate the comparison; our repeated-input symmetric source factors, eliminated private consumers and implicit large target differ. We are not implementing their algorithm merely by selecting a SciPy solver, and their recovery performance is not a guarantee here.

Exact matrix-free Newton curvature is an alternative when the full residual Jacobian is too large. For reduced scalar objective f(θ), automatic differentiation computes

$$
H(\theta)v=\nabla_\theta\big[\nabla f(\theta)^Tv\big].
$$

A Krylov trust-region solver uses this product without storing H. The native dense Hessian would contain110592² doubles, about97.8GB; an individual vector is0.885MB. HVPs require derivative graph storage and repeated contractions, which must be measured rather than called free. Exact Newton includes residual curvature and is not Gauss–Newton; indefinite curvature and rank-loss failures remain possible. There is no universal polynomial-time global optimizer for this unrestricted circuit-recovery problem asserted here.

## Executed consequences

1. Ten matched random starts on a dense4D planted rank2 symmetric cubic source-span target, same exact variable-projected residual in all arms. Trust-region least squares recovers to1e-5 relative coefficient error in3/10 starts for both raw and secant coordinates; L-BFGS recovers0/10 in either. Total4.09seconds. Budgets differ by solver's iteration/evaluation convention and are explicit; this is a bounded comparison, not an asymptotic ranking. Most failures are not justified structural negatives because the solution is known to exist. The coordinate rewrite does not universally improve basin discovery.
2. Dense independent Hessian and finite-difference controls validate the curvature adapter. On both saved native endpoints, exact HVP finite-difference discrepancies are6.43e-10 and3.22e-10. CPU gradient costs~1.29seconds and HVP~2.24seconds. Initial exact-coordinate objectives agree within3.66e-9 relative. Thus a native matrix-free comparison is executable without flattening the huge tensor.
3. Registered and implemented a four-arm native warm-start pilot: both saved endpoints × raw/secant coordinates, same exact-Newton trust-krylov solver,64steps/60seconds each,420seconds total wall cap. Fixed objective scale, full source domain, original fit position and separately scored held position. Success requires a10% capture gain AND unit-reader gradient<=1e-6; time-limited misses remain unfinished. Later independent starts and physical circuit validation are separate requirements.

Receipts: [ten-start control](CUBIC_SECANT_SOLVER_V1_CONTROL.json), [curvature control](CUBIC_PROJECTION_CURVATURE_V1_CONTROL.json), [gradient pullback](CUBIC_SECANT_GRADIENT_PULLBACK_V1_CONTROL.json), [native pilot specification](FOLDED_CUBIC_TRUST_PILOT_V1_PREREGISTRATION.md).

## Decision

Take the native curvature pilot before an expensive ten-start native sweep: it tests whether the proposed stronger method even improves the known failed endpoints at reasonable cost. If it remains unstationary, do not call absence of structure. Compare adequate continuation, independent initializations, and representations preserving the full joint operation using the measured failure. If it improves coefficient fits, require complete-function replication and frozen native behavioral tests before promoting any source feature. The already-confirmed regional component is unchanged.
