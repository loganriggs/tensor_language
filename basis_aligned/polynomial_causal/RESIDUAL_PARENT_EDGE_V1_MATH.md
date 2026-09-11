# Changing a shared reader's consumers using the full weight residual

The next arithmetic-graph operation now works: take an existing shared reader and propose using it in another output group, while removing one private reader to keep the group's input rank fixed. The search scores the proposed computation against the **native folded weight tensor**, including interference with all the other groups. This is an actual change to graph incidence, beyond rotating two existing branches.

The tested expansion of spectral parent 1 **does not pass our error allowance**. We tested all 62 groups that did not already consume it. Eleven proposed groups gave the parent at least 1% of their own quadratic energy under named-node deletion. The best qualifying addition was group 11. It saved 1,137 floating coefficients and one linear reader, with the variable-product count unchanged. But its normalized objective increase was about $2.9\times10^{-5}$ after all cores were refitted, exceeding the $10^{-6}$ allowance. No candidate is adopted or promoted as a circuit.

## What is being optimized?

Work in output coordinates that include the complete unembedding metric. The implicit target is the symmetric coefficient tensor of the last bilinear layer followed by linear unembedding, before the native final RMS/tanh tail. Each fitted output group has a unit writer $c_g$, an orthonormal input basis $B_g$, and a symmetric quadratic core $H_g$:

$$
\widehat T=\sum_g c_g\otimes Q_g,\qquad Q_g=B_gH_gB_g^\top.
$$

The fitting objective is

$$
J=\frac{\|T-\widehat T\|_F^2+\eta\sum_g\|Q_g\|_F^2}{N},
\qquad \eta=0.01,\quad N=\|T\|_F^2=99{,}245{,}061{,}353.47293.
$$

Hold all other groups and all writers fixed. Contract their residual with this group's writer:

$$
S_g=T\mathbin{\cdot}_{\rm output}c_g
-\sum_{h\ne g}(c_g^\top c_h)Q_h.
$$

For any proposed orthonormal input basis $Z$, the best core is exactly

$$
H_g^*=\frac{Z^\top S_gZ}{1+\eta}.
$$

This differs from independently reconstructing the old group matrix: it includes the original native target and cross-group overlap. All contractions use thin bases and the existing implicit native product representation; the full vocabulary-by-input-by-input tensor is never materialized. Same-basis conditional core polishing provides the matched comparison.

## The graph change and its first result

For each new consumer, retain all its old shared parents and insert frozen parent 1. Restrict the new input span to the union of its old rank-16 span and that parent. This union has dimension 17 in these cases. Keeping rank 16 requires discarding one direction in the private complement. The first implementation selects private directions using the full quadratic marginal, including shared/private interactions, then solves the retained core exactly. This reuses the earlier marginal-space correction; it is not assumed globally optimal.

[Version 1](residual_parent_edge_v1.py) tested all 62 possible additions. The best qualifying group's conditional objective cost was $2.9480921\times10^{-5}$. Refitting all 64 cores reduced the cost to $2.9011232\times10^{-5}$. The new consumer retained 3.069% named-parent removal energy. Numerical execution and normal-equation checks passed, but both acceptance bars missed. [Receipt](RESIDUAL_PARENT_EDGE_V1.json).

The changed graph has 1,238,656 floating coefficients, 92 integer indices, 1,010 linear reads and 1,041 variable products, versus the starting graph's 1,239,793 coefficients, 90 indices, 1,011 reads and 1,041 products. Native background, output metric, bias and tail remain required. Savings do not establish semantic reuse.

## Red-team: solve the discarded-direction problem with a numerical global bound

The potential explanation for the miss was a poor private-space heuristic. Here that restriction can be solved much more directly.

Let $S$ be the residual quadratic restricted to the 17-dimensional union. If $e$ is the unit direction discarded from the private complement, $P=I-ee^\top$ projects onto the retained span. The lost squared coefficient energy is

$$
\|S\|_F^2-\|PSP\|_F^2
=2\|Se\|^2-(e^\top Se)^2.
$$

Writing $e$ using private coordinates $v$, set $A=(S^2)_{\rm private,private}$ and $B=S_{\rm private,private}$. The problem becomes

$$
\min_{\|v\|=1}\left[2v^\top Av-(v^\top Bv)^2\right]
=
\min_{t\in[\lambda_{\min}(B),\lambda_{\max}(B)]}
\left[t^2+\lambda_{\min}(2A-2tB)\right].
$$

The identity follows by writing $-q^2=\min_t(t^2-2tq)$, interchanging two minimizations, and applying the minimum-Rayleigh-quotient identity. For the optimal pair, $t=v^\top Bv$, which lies inside the stated interval.

The minimum eigenvalue of an affine symmetric matrix is concave: it is the infimum of affine Rayleigh quotients. Therefore its endpoint chord lies below it on each interval. Adding $t^2$ gives an explicitly minimized quadratic lower bound. Eigenvectors at evaluated points give feasible upper bounds in the original vector problem. Repeatedly split the interval with the lowest bound until the gap closes. The concavity principle is standard; this mapping to our discarded-direction objective is the derivation used here. [Boyd and Vandenberghe, Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/).

[Solver](private_direction_bound_v1.py) uses FP64 eigensolves with a stated $10^{-12}$ scaled roundoff margin. Its bounds are **numerical optimality bounds, not formal interval-arithmetic certificates**. A diagonal known optimum and an independent 8,193-angle mixed two-dimensional control both pass. [Control](PRIVATE_DIRECTION_BOUND_V1_CONTROL.json).

[Version 2](residual_parent_edge_v2.py) reran all 62 additions. Every bound closed to at most $9.71\times10^{-10}$ in normalized fitting-objective units, using at most 10 interval splits; summed small bound-solver time was 0.0244 seconds, excluding residual contractions and other work. Independent projected-energy identities were within $5.10\times10^{-16}$.

The best qualifying group remained **11**. Its conditional cost improved only to $2.9473288\times10^{-5}$; all-core refitting gave $2.9004738\times10^{-5}$. Its named-parent removal energy remained 3.070%. The new graph's core solve converged in 66 iterations with true relative normal residual $9.93\times10^{-12}$; the unchanged baseline already met its core tolerance. The original private-space heuristic explains almost none of this miss. [Receipt](RESIDUAL_PARENT_EDGE_V2.json).

Even the cheapest unconstrained addition, group 18, costs $4.33627\times10^{-6}$ before joint all-core refitting and gives the new parent only 0.151% of group energy. Thus none of the conditional private-space minima meets the $10^{-6}$ allowance, even before requiring a meaningful parent role. We did not refit all 64 cores separately for every losing edge, so the conditional lower bounds do not certify that every jointly refitted graph must fail.

## What this changes

We now have residual-aware graph incidence proposals, exact conditional core fitting, and a numerically bounded alternative to the private-space heuristic. These tools address shared computation and allow a failed graph move to be distinguished from a failed local solver.

The negative result is restricted to **this frozen parent, existing writers, fixed group rank, and each old-span-plus-parent union**. It does not rule out moving the parent, learning a different shared sum, changing several connections together, or changing the input/output subspaces. The overall weight fit remains unconverged. A useful next graph search must change one of those restrictions; simply spending more steps on this discarded-direction subproblem is not justified. No corpus was used for discovery, and no new OOD, extraction or behavioral selectivity claim is made.


## Releasing the reader and input-space restrictions: matched local refits

The next experiment lets the readers and private input spaces move after the connection changes. It includes every affected consumer: starting from groups 11,16,25 and following all shared-reader connections gives groups **10,11,16,19,25** and parents **1,8,9**. Moving those three readers changes no other group. The other59groups can remain a literal frozen background.

Let $T_{\rm outside}$ be that background and let the columns of $C$ be an orthonormal basis of the five original group writers. For any permitted local tensor $\widehat T_{\rm local}$ whose output is in this span,

$$
\|T-T_{\rm outside}-\widehat T_{\rm local}\|_F^2
=
\|C^\top(T-T_{\rm outside})-C^\top\widehat T_{\rm local}\|_F^2
+
\|(I-CC^\top)(T-T_{\rm outside})\|_F^2.
$$

The last term is fixed. This reduces the output dimension from1152metric coordinates to5without changing any allowed objective difference. Writers may vary within that five-dimensional span; all private input spans and all three shared readers may move. The five cores are still solved jointly. This restriction is explicit: it does not allow writers to leave the original span or change the other59groups.

[CPU preflight](CLOSED_COMPONENT_REFIT_V1_PREFLIGHT.json) passes independent full/local objective differences and graph injection at errors below4.15e-16; directional gradient relative error is at most8.42e-8. The legacy objective's leading1creates a harmless constant shift, **-0.09978352475059615**, which the runner adds back before reporting full objectives. Its shifted capture is never called whole-model capture.

Two separately managed20-minute fits are now queued: original connections versus parent1additionally used by group11. Both use the same frozen outside graph/outputspan and the established bounded solver. Each may stop early only when fresh gradient and recent progress bars hold. The comparison asks whether reader/private-space movement brings the new graph within1e-6of the equally optimized original, saves at least1000floats, and keeps at least1%parent1removal energy in consumers11,16,25. These are weight/interface criteria; native behavioral validation remains separate. [Registered protocol](CLOSED_COMPONENT_REFIT_V1_PREREGISTRATION.md).
