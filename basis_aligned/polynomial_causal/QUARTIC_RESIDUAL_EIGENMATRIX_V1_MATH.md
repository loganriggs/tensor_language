# Residual-derived quadratic intermediates

12 September 2026, 01:43 UTC. A planted recovery control and native operator validation are completed. No native spectral factorization has run.

The known stationary miss can be repaired by constructing a new quadratic from its missing coefficient function. This is more directed than random replacement or selecting from an old spectral bank. It remains an initialization method, not a global optimization or circuit-identification theorem.

## Spectral relaxation

After removing a weak node and refitting retained output coefficients, let $R_m$ be the residual fully symmetric quartic tensors. In the small control we choose the output combination with largest residual coefficient energy. For that scalar residual $R$, define a linear operator on symmetric matrices:

$$
[\mathcal R(Q)]_{ij}=\sum_{kl}R_{ijkl}Q_{kl}.
$$

It is self-adjoint under the Frobenius inner product. The quadratic-square feature associated with $Q$ is $F_Q=\operatorname{sym}(Q\otimes Q)$, and

$$
\langle R,F_Q\rangle=\langle Q,\mathcal R(Q)\rangle_F.
$$

An eigenmatrix of largest absolute eigenvalue maximizes the absolute expression on the right over unrestricted unit-Frobenius symmetric matrices. However, the actual atom has normalization

$$
\|F_Q\|_F^2=\frac{\|Q\|_F^4+2\operatorname{tr}(Q^4)}{3},
$$

and a rank constraint on $Q$ changes the optimization problem. Projection against retained quartic features changes the candidate denominator again. Thus an eigenmatrix followed by rank truncation is a **relaxation and initialization**, not the exact best rank-limited atom. We score its actual conditional coefficient gain before accepting it.

## Executed recovery test

The [control](QUARTIC_RESIDUAL_EIGENMATRIX_V1_CONTROL.json) uses the previously saved two-node planted fit stuck at66.05% error. It removes the least conditionally contributing node, computes the residual from the target tensor and retained program, takes four largest-absolute eigenmatrices, and truncates each to rank2. It includes the original node as a fallback. No candidate reads the planted true factors; those define the target and final recovery check only.

The selected candidate reduces error to23.23% before nonlinear optimization. Joint refitting reaches2.28e-7 error in19 recorded steps, with gradient3.87e-7. All three registered predictions hold. Twelve earlier local perturbations failed, and random weak-node replacement recovered2/4 times. These are results on one small failed instance, not comparative success-rate estimates for the native problem.

## Native operator without a fourth-order array

Write the producer quadratics as $A_a=\operatorname{sym}(l_a r_a^T)$ and its downstream scalar form as

$$
H=\lambda_{17,0}^2D_{16}^T S D_{16},\qquad
T=\operatorname{sym}\!\left(\sum_{ab}H_{ab}A_a\otimes A_b\right).
$$

Contraction with a symmetric matrix $Q$ gives

$$
\mathcal T(Q)=\frac{1}{3}\left[
\sum_a A_a\sum_b H_{ab}\operatorname{tr}(A_bQ)
+2\sum_{ab}H_{ab}A_aQA_b\right].
$$

The [implementation](quartic_weighted_trace_v1.py) evaluates this using $LQL^T$, $LQR^T$, $RQR^T$ and native matrix contractions. The earlier partial trace is precisely $Q=I$. A fitted quadratic square contributes

$$
\mathcal F_j(Q)=\frac{\operatorname{tr}(Q_jQ)Q_j+2Q_jQQ_j}{3},
$$

which can be subtracted in low-rank form to apply the residual operator. This acts on a matrix of size1152-by1152 instead of materializing a1152-to-the-fourth coefficient array. The precomputed producer core is4608-by4608, about170MB in float64. A future eigensolver must also charge its vectors, workspaces and all native producer dependencies.

[Dense checks](QUARTIC_WEIGHTED_TRACE_V1_CONTROL.json) compare against the independently implemented composed four-linear oracle, the previous identity-trace formula, self-adjointness and fitted-square subtraction. Maximum error is2.75e-16.

The [CPU native check](QUARTIC_WEIGHTED_TRACE_NATIVE_V1_PRICE.json) uses the actual first selected output form, without language data. Identity contraction agrees with the saved native trace to1.42e-12; a signed rank2 contraction agrees with independent four-slot oracle evaluations to1.58e-15. Two operator actions take3.70 and3.70 seconds; total setup/check time is9.55 seconds. This is CPU timing only, not GPU pricing or eigensolver convergence.

## Decision boundary

### Why a large flattening rank does not rule out simple intermediates

The following is an analytic counterexample, checked independently of the native fit. Let a single quadratic be $q(x)=x^TQx$, with $Q=I_r/\sqrt r$ on an $r$-dimensional subspace. Its square has just one quadratic intermediate followed by one squaring operation. Nevertheless its symmetric flattening acts as

$$
\mathcal F(X)=\frac{\operatorname{tr}(X)I_r+2X}{3r}.
$$

The identity direction has eigenvalue $(r+2)/(3r)$; every traceless symmetric direction has eigenvalue $2/(3r)$. Thus its matrix rank is $r(r+1)/2$, despite having one quadratic-square node. At our permitted inner rank16 this can be136. The [executed control](QUARTIC_FLATTENING_COMPLEXITY_V1_RESULT.json) verifies ranks1,3,10,36 for inner ranks1,2,4,8 and spectral error at most1.67e-16. Reader storage still grows with inner rank: this is not a constant-cost program claim.

Consequently neither a broad matrix spectrum nor the number of required eigenmatrices counts reusable arithmetic nodes. Spectral initialization can help search; preserving eigencomponents individually can miss the cheaper quadratic representation. This is a concrete reason to red-team negative spectral results before concluding that the composition lacks structure.

Related algorithms solve different objects. [Hopkins, Schramm and Shi (2019)](https://proceedings.mlr.press/v99/hopkins19b.html) give robust fourth-order decomposition under algebraic nondegeneracy assumptions, with runtime $\widetilde O(n^2d^3)$ up to conditioning factors. Those assumptions have not been checked here, and their component count is not our count of rank16 quadratic intermediates. [Nie and Wang](https://arxiv.org/abs/1308.6562) formulate best symmetric rank-one approximation as polynomial optimization on a sphere and propose semidefinite relaxations. That is relevant to fourth powers of linear forms; replacing them with squares of signed, rank-limited quadratics changes the feasible set. Neither source supplies a verified global solution to our retained-span conditional atom problem.

The tool is ready for a possible residual-informed initialization after the pending comparisons. No native eigenmatrix has yet been found, rank-truncated, optimized or behaviorally scored. A future native test must verify eigen residuals, retain exact conditional-gain selection, and compare frozen native effects at the same final program cost. Estimating a multi-output residual direction would require a declared method; the small control's exact residual Gram is not available automatically at native scale. Preserve the distinction between finding a useful starting point, fitting coefficients, and obtaining circuits with the four requested properties.

## Matrix-free eigensolver and GPU pricing, 12 September 02:20

The prior square, replacement and learned mixed-core comparisons have completed; their [terminal interpretation](COUPLED_QUARTIC_WRITER_V1_MATH.md#terminal-comparison-12-september-0154) motivates testing new initialization as an alternative to further local continuation. The joint mixed-graph fit is currently running independently. No native residual eigensolve has run yet.

The [new adapter](quartic_matrixfree_eigen_v2.py) uses the symmetric matrix's diagonal entries and $\sqrt2$ times each upper off-diagonal entry as coordinates. This preserves the Frobenius inner product in $d(d+1)/2$ dimensions:664128 at $d=1152$, approximately5.31MB per float64 vector. Twelve Lanczos vectors alone take about63.8MB, before operator cores, packed-coordinate indices, workspaces and device transfers. We do not materialize the664128-by664128 flattening matrix.

The adapter calls [SciPy's symmetric ARPACK interface](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html) with a fixed random starting vector and largest-magnitude eigenvalue selection. It verifies each returned eigenpair with a fresh operator action. Partial convergence and an explicit action-budget limit are separate statuses; the latter returns no invented eigenpairs. These checks verify numerical eigenpairs, not that rank-truncated matrices solve the conditional atom problem.

The [CPU control](QUARTIC_MATRIXFREE_EIGEN_V2_CONTROL.json) uses an independently assembled dense signed quartic flattening. Four returned eigenvalues agree to5.01e-16 relative error, individual eigen residuals are below1.02e-15, and the norm/roundtrip identities hold. A one-action cap terminates explicitly. V2 constructs coordinate scales in float64 even if the caller's default tensor dtype is float32; the earlier V1 control used a float64 default and did not cover that boundary. Previously executed V1 files are preserved, and V2 is the selected implementation.

The managed [GPU operator-price job](../bilinear_quotient/ops/run_quartic_eigen_operator_price_v1.py) is queued behind the mixed fit. It checks the same native identity and signed rank2 contractions as the earlier CPU receipt, then measures a third action including packed CPU/GPU transfer. Numerical agreement at1e-8, speed relative to the recorded CPU actions, and allocation below28GiB are separate predictions. An eigensolve budget will use that measured cost. This preparation introduces neither a new native factorization result nor a new circuit claim.

## Four output directions without estimating a residual covariance

The [candidate implementation](../bilinear_quotient/ops/run_quartic_residual_eigen_candidate_v1.py) is drafted, not queued or executed. Its action/time configuration and input binding will be fixed after the GPU price arrives and the running mixed-fit result is interpreted. It uses the frozen LBFGS V1 square program so the comparison remains directly matched to the earlier node-replacement experiment.

Remove the weakest conditionally contributing node, refit the31 retained nodes, and let $R_0,R_1$ be their two-output coefficient residual. For a unit-Frobenius quadratic matrix $Q$, write

$$
c(Q)=\big(\langle R_0,F_Q\rangle,\langle R_1,F_Q\rangle\big).
$$

Use the four fixed directions

$$
\mathcal D=\left\{(1,0),(0,1),\frac{(1,1)}{\sqrt2},\frac{(1,-1)}{\sqrt2}\right\}.
$$

For every two-dimensional vector $c$, one of these unoriented axes is within $\pi/8$ of it. Hence

$$
\max_{w\in\mathcal D}|w^Tc|\ge\cos(\pi/8)\,\|c\|_2.
$$

This avoids needing an estimated residual output Gram merely to choose a direction. If each scalar residual operator were solved exactly over unrestricted unit-Frobenius matrices, the best sampled correlation would also be within this factor of the unrestricted two-output optimum: maximize the pointwise inequality over $Q$. Our numerical solver verifies eigen residuals rather than certifying spectral extrema. More importantly, neither rank16 truncation nor the Schur-complement feature normalization preserves this guarantee. It is a rationale for covering four directions, not a guarantee on native candidate quality.

The draft requests two largest-magnitude eigenmatrices per direction, producing at most eight proposed rank16 quadratics. It adds them to the old bank, scores actual conditional gains, includes the deleted node as a fallback, and refits the selected32-node output map. The resulting program remains592704 floats. It reuses the diagonal target correlations from the already bound full mixed-core receipt; the source readers, quadratic eigenvalues and output writers must match the square baseline exactly, and the reconstructed original objective must agree within1e-8. Newly proposed correlations still use the native four-slot contraction oracle.

Registered comparisons are separate: all eigen/replay checks and nonincrease; at least1% capture improvement; at least20% lower native write error. The latter does not enter selection. An incomplete eigensolve records an explicit failed numerical stage with no candidate program, rather than silently returning an old node as successful new evidence. Joint nonlinear refitting, more output directions, and the four circuit properties remain further questions.
