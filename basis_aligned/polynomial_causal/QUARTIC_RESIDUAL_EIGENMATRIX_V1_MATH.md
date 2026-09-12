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

**Execution update,02:34:** the [GPU price receipt](QUARTIC_EIGEN_OPERATOR_PRICE_V1_RESULT.json) passes all bars: actions take0.259/0.255seconds versus3.70seconds on CPU, packed roundtrip0.258seconds, peak allocation1.24GiB. Identity/scalar/packed errors are1.42e-12,3.37e-15,1.50e-15. After interpreting the completed mixed fit's failed native effects, the candidate is now audited and queued with [bound configuration](QUARTIC_RESIDUAL_EIGEN_CANDIDATE_V1_BINDING.json):200actions per direction plus eigenpair verification, at most808actions, and a600-second outer alarm. Base action timing suggests about208seconds before fitted-residual subtraction and eigensolver overhead. This supersedes the draft-only status above; no native candidate result yet.

**Terminal candidate,02:36:** all four native eigensolves converged in22,22,22,13 operator actions, plus eight verification actions total. Maximum eigen residual is1.66e-7. The entire [candidate experiment](QUARTIC_RESIDUAL_EIGEN_CANDIDATE_V1_RESULT.json) takes26.89seconds and peaks at1.57GiB. Candidate33—the rank16 truncation of the positive eigenmatrix for the first output direction—replaces node13. Coefficient capture gains0.348%, and native write error improves26.65→26.12%. Numerical A passes; the1% gain and20% write-improvement bars both fail. The [native-effect scorer](QUARTIC_RESIDUAL_EIGEN_NATIVE_EFFECTS_V1.json) also passes replay and fails swap/removal fidelity. No candidate has been identified as a circuit.

This is a valid but weak initialization result. The small planted example required joint refitting after its spectral proposal, so the next [managed refit](../bilinear_quotient/ops/run_quartic_residual_eigen_refit_v1.py) allows180updates or900fit seconds from the new node. It reuses the existing LBFGS implementation and compares coefficient capture with the saved180-update prefix of the original continuation: objective−1.829830191668918 after815.17seconds. Registered B requires at least1% more captured energy than that reference; C remains gradient at most1e-6, separate from gain. A checks gradient finite differences, starting replay and descent. Wall times and line-search work may differ; no exact time-matched claim is made. The old180-update prefix has no saved native program, so we will not invent a behavioral comparison at that prefix. The frozen final program will instead be scored against the exact selected component with the existing native background. This is the intended initialization-plus-refinement test, not an automatic unchanged continuation.

## Refitting does not repair fidelity; rank truncation has a measurable cost

The [180-update refit](QUARTIC_RESIDUAL_EIGEN_REFIT_V1_RESULT.json) finishes with objective−1.841864, only0.658% more capture than the old180-update reference, missing the1% bar. It takes814.79fit seconds versus815.17for that reference. Projected gradient0.0927 misses stationarity; native write error worsens26.12→28.80%. [Swap/removal scoring](QUARTIC_RESIDUAL_EIGEN_REFIT_NATIVE_EFFECTS_V1.json) passes replay but fails fidelity: swap errors50.88/38.47/35.62/33.07%, removal-effect CE disagreements0.0539/0.0988/0.0337/0.0609nats. The changed initialization has not yielded a circuit or a substantially better matched fit.

A mathematical implication of the **already computed untruncated eigenvalues** identifies a different restriction. For a unit-Frobenius matrix $Q$,

$$
\|F_Q\|_F^2=\frac{1+2\operatorname{tr}(Q^4)}3\le1.
$$

Let $v$ be its coefficient norm squared after projection away from retained features. Then $v\le\|F_Q\|_F^2\le1$. If $Q$ is an exact eigenmatrix of a unit output-direction residual operator, its residual correlation in that direction equals its eigenvalue $\lambda$. Consequently its two-output conditional addition gain satisfies

$$
\Delta=\frac{\|c_{\mathrm{res}}(Q)\|_2^2}{v}\ge\lambda^2.
$$

Subtract the cost of removing the old node to obtain a net-improvement lower bound. The [evaluated receipt](QUARTIC_EIGEN_TRUNCATION_BOUND_V1.json) uses maximum absolute eigenvalue445960792, old captured energy5.33326e17 and removed-node cost2.45429e14. It implies **at least37.24% extra capture for an untruncated candidate**, under exact eigenpair/operator assumptions, versus the observed0.348% from the rank16 candidates. This is a mathematical bound evaluated with numerical eigenpairs, not an interval-certified floating-point bound or a behavioral guarantee. It does not mean37.24% of the whole target or model is explained.

The next [native capacity comparison](../bilinear_quotient/ops/run_quartic_full_quadratic_candidate_v1.py) therefore tests full symmetric $Q$ against rank16 truncation using exact matrix-free target correlations, the same retained31-node span, and coefficient-only selection. It charges the larger full matrix explicitly and records its overlap with the identity matrix to diagnose possible radial structure. Rank16 replay and eigensolver accuracy are numerical A; full net capture gain at least30% is B; full native write error at most20% is C. A higher-capacity weight fit still needs functional validation. This test changes the rank restriction instead of repeating the same local optimizer.


## Full quadratic result: the capacity restriction matters, but extraction still fails

The [full-matrix comparison](QUARTIC_FULL_QUADRATIC_CANDIDATE_V1_RESULT.json) completed in34.70seconds with numerical and coefficient-gain bars held. One untruncated quadratic gives **121.77% additional captured coefficient energy** over the original32-node program, versus0.348% for rank16 replacement. This is relative captured energy, not fraction of the whole target. Its native write error is25.02%, missing the20% bar. It stores1,238,384 fitted floats, compared with592,704 originally; the validation-write cache is additional and not part of execution.

The winning full quadratic is the negative eigenmatrix of the second output-direction residual. Rank16 keeps only14.99% of its squared matrix norm. Its squared overlap with the normalized identity is3.33%, so a predominantly identity/radial matrix does not explain the result. A high-rank intermediate can still be a useful arithmetic node; low matrix rank was an extra assumption in the earlier implementation.

The extended [shared scorer](quartic_native_effects_v3.py) exactly reproduces the prior square-program receipt in its [regression check](QUARTIC_NATIVE_EFFECTS_V3_CONTROL.json). [Frozen full-quadratic effects](QUARTIC_FULL_QUADRATIC_NATIVE_EFFECTS_V1.json) reproduce native write accounting to2.8e-16 and the exact-reference effects bit-for-bit. Swap errors improve in every family:

| Family | Original square program | One full quadratic | Removal CE disagreement, full quadratic |
|---|---:|---:|---:|
| Agreement verbs |65.51%|25.13%|0.0298 nats|
| Count nouns |55.57%|31.98%|0.0879 nats|
| Past |35.64%|21.37%|0.0289 nats|
| Progressive |34.06%|18.78%|0.0503 nats|

Nevertheless, every family misses the10% swap-error and0.02-nat removal-disagreement bars. Count-noun swap sign agreement is87.5%, below90%. These are reused developmental examples with the native background retained, not OOD evidence or standalone extraction. No circuits are promoted.

### Next discriminating test: several full quadratic intermediates

The [eight-node bank experiment](../bilinear_quotient/ops/run_quartic_full_quadratic_bank_v2.py) recomputes the same eight coefficient-derived eigenmatrices and refits their outputs jointly with the31 retained low-rank intermediates. This changes the number of unrestricted intermediates while preserving weight-only discovery. It asks whether the behavioral fidelity limit comes partly from keeping just one such intermediate. The cost rises to about5.89million fitted floats; any gain must be interpreted at that price.

For symmetric full matrices $Q_i,Q_j$, their squared-quadratic features have exact coefficient Gram

$$
K_{ij}=\frac{\operatorname{tr}(Q_iQ_j)^2+2\operatorname{tr}(Q_iQ_jQ_iQ_j)}3.
$$

Combine this block with the retained-feature Gram and their cross terms, then solve $KA=C$ for the output coefficients. The existing scaled eigensolver discards eigenvalues below $10^{-12}$ of the maximum and reports its normal-equation residual. This is a converged linear output solve when the residual passes, not joint nonlinear convergence or a globally optimal discovery algorithm. Dense explicit-quartic and low-rank/full cross checks agree within2.3e-13 in [the focused control](QUARTIC_FULL_BANK_V1_CONTROL.json).

Registered A requires single-full replay and normal-equation residual at most1e-8, with each eigen residual at most1e-5. B requires10% extra coefficient capture over the one-full-node program; C requires native write error at most20%. Native data do not choose the bank, weights or iterate. Shared swap/removal scoring follows the frozen result. A miss would not show that sparse arithmetic structure is absent: the eight matrices come from only four residual directions, their readers are frozen, and cross-products between distinct full quadratics are excluded.

### Eight-full bank completed: substantial improvement, intervention thresholds still missed

The [bank receipt](QUARTIC_FULL_QUADRATIC_BANK_V1_RESULT.json) passes all three registered bars in38.82seconds. Captured coefficient energy is2.04777e18, **73.13% above the one-full-node program**. The39-feature linear solve has relative normal residual1.35e-15 and full numerical rank; this establishes its fixed-bank output solve only. Native write error falls to**10.89%**. Literal fitted storage is5,887,294 floats, plus separately recorded validation/Gram caches. The static enqueue checker rejected the first runner's prediction-key syntax before execution; the [second runner](../bilinear_quotient/ops/run_quartic_full_quadratic_bank_v2.py) preserves the experiment and completed normally.

[Native effects](QUARTIC_FULL_QUADRATIC_BANK_NATIVE_EFFECTS_V1.json), computed by the shared scorer's bank extension, reproduce the saved write and exact-reference effects. All64 live swap signs agree. Swap relative errors are15.85/26.16/19.41/19.71% for agreement/count/past/progressive; all still miss10%. Removal CE disagreements are0.0132/0.0456/0.0117/0.0208nats: agreement and past pass individually, count and progressive fail, so the registered all-family removal verdict remains failed. This is a better approximation of the selected two-output composed component; it is not full-unembedding recovery or an identified circuit.

The executed [paired-difference diagnosis](QUARTIC_FULL_BANK_DIAGNOSIS_V1.json) explains why11% endpoint accuracy can coexist with larger swap errors. Before the final native normalization/readout, reference donor-minus-base writes have norms only3.90–23.69% of the combined endpoint norms. Approximation errors in those differences are15.0–31.8%, or1.39–2.74times the relative endpoint errors. Exact common/difference accounting holds within3.7e-16. This supports sensitivity to small differences, without claiming it completely explains the nonlinear margin errors.

For endpoint errors $e_b,e_d$, define $e_+=(e_b+e_d)/2$ and $e_-=e_d-e_b$. Then

$$
\|e_b\|^2+\|e_d\|^2=2\|e_+\|^2+\frac12\|e_-\|^2.
$$

The same CPU receipt measures redundancy in coefficient space. Deleting any one of the eight full nodes and optimally refitting the remaining output coefficients loses only0.20–2.93% of total captured energy, whereas individual unfitted node energies range2.99–125.37% of that capture. These are different interventions: conditional importance permits compensation, direct node energy does not. Their disparity shows correlated/cancelling features and cautions against treating eigenmatrices as independent circuits. No behavioral data were used to repair the factors.

The remaining question is how to discover stable shared quadratic subcomputations and preserve their small input-dependent differences. This result justifies relaxing the rank restriction; it does not justify unlimited bank growth or a claim that eigendecomposition alone has found the circuit DAG.

## Reusing the native producer, and recovering the stronger baseline

The [producer-span test](QUARTIC_BANK_PRODUCER_SPAN_V1_RESULT.json) asks whether the learned full matrices can read already existing MLP16 computations. Write its neuron product matrices as

$$
A_a=\operatorname{sym}(l_a r_a^T),\qquad z_a(x)=x^TA_ax.
$$

Their Frobenius Gram is computed without materializing4608 dense matrices:

$$
G_{ab}=\frac{(l_a^Tl_b)(r_a^Tr_b)+(l_a^Tr_b)(r_a^Tl_b)}2.
$$

For each learned quadratic $Q$, solve $G\beta=c$, with $c_a=l_a^TQr_a$, to project it onto the neuron-product span. To restrict it further to readers of the actual MLP16 output $D z$, solve $(DGD^T)a=Dc$ and set $\beta=D^Ta$. These are different shared interfaces. Bias is outside the homogeneous path; the learned residual scale can be absorbed in the reader coefficients.

Both exact linear projections and native execution checks agree within3.2e-15. Neuron-span matrix errors are7.69–16.78%, missing the all-below10% prediction. Their composed write changes5.63%, missing the5% preservation bar, but reference error improves10.89→6.74%. Output-reader projection gives8.41% reference error. This is compatible with the known canonical-flattening caveat: symmetrization mixes matrix slots, so canonical eigenmatrices need not be native producer variables. The earlier [flattening counterexample](QUARTIC_FLATTENING_COMPLEXITY_V1_RESULT.json) already demonstrates this phenomenon; no new toy is needed.

Neuron-product reuse needs36,864 new coefficients for eight readers; native-output reuse needs9,216. The native parent weights are **not free**: the shared product interface requires10,616,832 L/R values, and the shared output interface15,925,248 L/R/D values. Retained31 low-rank terms and downstream writers/mixing also remain. Small marginal reader storage is not a smaller standalone program.

The [neuron effects](QUARTIC_BANK_NEURON_NATIVE_EFFECTS_V1.json) and [output-reader effects](QUARTIC_BANK_OUTPUT_NATIVE_EFFECTS_V1.json) both fail the all-family swap/removal criteria. Neuron swaps are39.60/16.62/14.64/19.43%; output-reader swaps22.61/12.81/10.16/4.98%. Improved level reconstruction does not uniformly improve interventions.

### The old outer16 baseline was missing its intervention screen

The earlier exact-producer outer16-per-output baseline already had5.74% write error. Its [newly completed intervention screen](QUARTIC_OUTER_BASELINE_NATIVE_EFFECTS_V1.json) passes every swap family:3.68/8.31/7.17/1.24%, with all signs agreeing. Removal CE disagreement is0.0158/0.0338/0.0058/0.0099nats; only count nouns fail. Reconstructing the old weight formula reproduces its write error within1.3e-12. The extracted [shared numerical scorer](quartic_frozen_native_score_v1.py) exactly reproduces the previous initial-program effect report, avoiding further format-specific scoring copies.

This is a material comparison correction: the new full-matrix fits were not the best developmental intervention approximation available. The earlier spectral result had been screened for write accuracy but not fully compared on interventions. Completing that missing comparison changes the next action.

A separately registered, fixed **32 outer terms per output** then tests that single unresolved extraction limit. Its [receipt](QUARTIC_OUTER32_NATIVE_EFFECTS_V1.json) passes replay, swaps and removals in2.50CPU seconds. Write error is1.69%; swaps are2.01/7.44/2.53/0.80%, all64 signs agree, and removal disagreements0.0018/0.0118/0.0016/0.0024nats all pass. This uses76,096 fitted values plus the15,925,248-value shared native producer and its declared normalization/background dependencies. Eigen truncation solves its individual symmetric-matrix approximation; it does not establish global optimality for the composed quartic or semantic uniqueness.

The result supports a faithful **partial-component interface on developmental rows**, not an isolated linguistic circuit, full-unembedding decomposition, or the four-property goal. The next step is a frozen fresh lexical/construction validation before naming or adopting factors. Continued growth of the dense eigenmatrix bank is demoted in light of this stronger simple baseline.

## Frozen fresh validation: removal generalizes, progressive swaps narrowly fail

[The fresh receipt](QUARTIC_OUTER32_FRESH_V1_RESULT.json) completes18body forwards/144sequences in1.96seconds excluding binding checks. All128 native answer-versus-foil contrasts are correct. Actual model-hook versus manual-tail relative error is below8.3e-7; exact composed numerator identities are within3.8e-15. The previously frozen program is unchanged.

| Fresh family | Write error | Swap error | Swap sign agreement | Removal CE disagreement |
|---|---:|---:|---:|---:|
| Agreement |2.38%|5.11%|93.75%|0.00801 nats|
| Count |2.75%|2.40%|100%|0.01135 nats|
| Past |1.37%|9.17%|100%|0.00278 nats|
| Progressive |1.96%|10.54%|100%|0.00545 nats|

All families have16live swap pairs. Registered A/C/D/E pass; **B fails** because progressive exceeds10%. This is a narrow but real failed prediction. There is no threshold adjustment, row filtering or claim that the complete fresh screen passed. It is lexical/construction generalization, not corpus OOD or proof of a selective standalone circuit.

The [executed error diagnosis](QUARTIC_FRESH_ERROR_DIAGNOSIS_V1.json) finds progressive error distributed across examples: the largest row contributes15.9% of squared error and the largest four47.2%. First-order propagation of the swap-state discrepancy through native RMS and capped logits predicts the margin error within0.079%relative for progressive, and0.086–0.162%for the other families. This is consistent with ordinary approximation sensitivity; nonlinear curvature and a single bad example do not explain away the miss.

Independently, a weight-only test finds the omitted outer matrix spectrum holds55.9/56.7% of its two matrices' squared Frobenius norms, despite the small native write error. A scalar identity on the omitted subspace explains only0.405/0.104% of that tail energy. A predominantly isotropic omitted tail is therefore not supported in this matrix metric; a small-energy correction could still matter behaviorally, but no data-driven correction is fitted.

The next [producer-metric derivation](PRODUCER_METRIC_SPECTRAL_V1_MATH.md) changes the weight objective: compose the native producer's exact quadratic Gram into the outer matrix before truncation. Its explicit paired-tensor control passes. Native execution is still pending; no improvement is inferred from the derivation.
