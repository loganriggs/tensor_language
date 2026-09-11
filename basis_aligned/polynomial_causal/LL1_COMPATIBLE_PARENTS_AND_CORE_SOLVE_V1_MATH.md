# Compatible shared parents and converged interaction coefficients

The joint graph is executable, but its approximation error still exceeds the registered limit. Two further checks narrow the reason. Selecting jointly compatible parents fixes the measured span incompatibility but removes only 4–7% of the error. Solving every group's interaction coefficients together to convergence improves the fit somewhat more, but most of the gap remains. The next numerical search must change shared readers and private input spaces, not only their interaction coefficients.

These checks use weights only. They do not fit corpus activations or establish the four behavioral circuit properties. The input groups still come from the frozen, unconverged LL1 pilots. The independent longer projected LL1 run is separate.

## 1. Individual compatibility does not imply joint compatibility

For each original group, let $A_g$ be an orthonormal basis of its input space. If proposed shared parents have orthonormal joint basis $U_g$, their worst squared membership is

$$
\mu_g=\sigma_{\min}(A_g^TU_g)^2.
$$

This tests the entire parent span, including differences between nearly parallel readers. We enumerated every subset of each group's at-most-four proposed parents. We selected the largest subset with $\mu_g\geq0.90$, breaking ties by the sum of already-measured parent mixed energies. Parent nodes with fewer than two remaining consumers were removed. Removing more readers from a compatible span cannot lower this minimum membership.

The surviving graphs have 12/16 shared parents, with 6/7 groups still consuming multiple parents. Actual minimum membership is 0.9514/0.9751. Thus multiple-parent structure survives the check.

However, the capture loss drops only from 0.0012125 to 0.0011255 in the spectral start and from 0.0015298 to 0.0014698 in the native start. Both fail the registered 50% loss-reduction prediction and the final 0.001 limit. The graphs retain 81.2%/94.1% of the preceding graph's float savings and save 1.1645%/1.4318% against the original LL1 parameter bill. Execution errors are below $2.2\times10^{-15}$.

This is a correction to the earlier interpretation: spectacularly bad worst-case joint membership was real, but fixing it explains little of the aggregate loss. It is not sufficient to attribute the remaining error to that confound.

## 2. Solve all cores jointly, including their cancellation

At fixed group input spans and fixed output directions, the whole interaction problem is linear. Write

$$
\widehat T=\sum_{g=1}^{m}\widehat c_g\otimes(B_gH_gB_g^T),
\qquad B_g^TB_g=I,\quad \|\widehat c_g\|=1,\quad H_g=H_g^T.
$$

There are $m=64$ groups and $16\times16$ symmetric cores $H_g$. Output magnitudes are absorbed into the cores. The group input basis combines that group's shared-parent span and private span. It is only a coordinate system for solving the coefficients; installation converts back to the original shared readers and diagonalizes the private block, preserving the graph's storage and operation counts.

The objective is

$$
\frac{\left\|T-\sum_g\widehat c_g\otimes(B_gH_gB_g^T)\right\|_F^2
+\eta\sum_g\|H_g\|_F^2}{\|T\|_F^2},\qquad\eta=0.01.
$$

Because the bases are orthonormal and writers unit norm, the penalty is exactly the existing whole-group tensor-energy penalty. It is not a new coefficient-dependent regularizer.

Let $M_{gh}=B_g^TB_h$. The normal operator is

$$
(\mathcal A H)_g=
\sum_h(\widehat c_g^T\widehat c_h)
M_{gh}H_hM_{gh}^T+\eta H_g.
$$

The right-hand side is the target tensor contracted with writer $\widehat c_g$ and both input bases $B_g$. Our implicit native product factors compute it without constructing the vocabulary-by-input-by-input tensor. Cross-group terms account for cancellation and overlap; these are not independent local core fits.

The operator is positive definite because $\eta>0$. Each group's embedding is an isometry on symmetric cores, so the stacked embedding has squared operator norm at most $m$. Therefore

$$
\kappa(\mathcal A)\leq\frac{m+\eta}{\eta}=6401.
$$

Conjugate gradients applies this operator without constructing the full core-by-core normal matrix. The implementation verifies its final true normal residual against the right-hand-side norm, rather than relying solely on the recursively updated residual. A small dense solve, coefficient reconstruction, objective identity and executable graph all agree below $9\times10^{-16}$.

## 3. A matched optimization control

We solved both the proposed graph and the original LL1 input spaces, with the same fixed original output directions and penalty. Otherwise, a gain from extra optimization could be misattributed to the shared graph.

| Measurement | Spectral start | Native-product start |
|---|---:|---:|
| Original-space solve iterations | 45 | 38 |
| Shared-graph solve iterations | 57 | 57 |
| Original-space solve seconds | 1.435 | 0.494 |
| Shared-graph solve seconds | 1.142 | 0.754 |
| Original optimized capture | 11.6608% | 11.6397% |
| Shared-graph optimized capture | 11.5526% | 11.4965% |
| Matched capture gap | 0.0010820 | 0.0014321 |
| Shared-graph penalized objective gain | 0.00008437 | 0.00007484 |

All four true relative normal residuals are below $9\times10^{-9}$, passing the $10^{-8}$ convergence bar. Independent full coefficient-objective checks agree within $1.8\times10^{-15}$; executable graph checks agree within $1.4\times10^{-15}$. CPU timing covers the iterative solve, not all target contractions, artifact writing or validation.

Both graph fits pass the registered $10^{-5}$ objective-improvement bar. Both still fail the matched 0.001 capture-gap bar. Original-space improvements are much smaller, at $1.61\times10^{-6}$ and $4.83\times10^{-7}$ in penalized objective. This rules out inadequate optimization of this particular fixed-space, fixed-writer linear subproblem as the main remaining obstacle. It does not establish convergence of the full reader/writer/graph problem or rule out much better graph structures.

## 4. What to optimize next

The shared-parent and private input spans have so far been proposed geometrically. Keeping them fixed, even while optimizing all core coefficients, retains most of the approximation gap. A useful next search changes those spans jointly while preserving shared reader identities across consumers. The conditional core solve can be reused inside that search, so every reader proposal is judged after optimizing its interactions.

That next objective must retain the complete folded weight metric, the same group-energy penalty, explicit graph price, and matched original-space controls. Local convergence of a conditional linear solve is not global recovery of an arithmetic DAG. Frozen FineWeb and the four circuit tests follow candidate discovery; they are not substituted for this weights-first search.

Implementations: [compatible incidence selection](ll1_compatible_parents_v1.py), [selection audit](ll1_compatible_parents_v1_audit.py), [joint core solver](ll1_joint_core_solve_v1.py), [matched solver audit](ll1_joint_core_solve_v1_audit.py). Receipts: [compatible graphs](LL1_COMPATIBLE_PARENTS_V1_AUDIT.json), [converged core solves](LL1_JOINT_CORE_SOLVE_V1_AUDIT.json). The final graph artifact is `LL1_JOINT_CORE_SOLVE_V1_GRAPHS.pt`; it preserves the shared-parent executor and includes the output whitener and bias needed for the physical interface.
