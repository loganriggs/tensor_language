# Exact coupled output solve for the composed quartic

The nested spectral baseline independently approximated each inner quadratic and then retained its old outer weight. It failed native write fidelity, despite an adequate outer approximation. This experiment asks whether jointly sharing those fixed quadratic features across both output directions can repair extraction. It does not yet move their input readers.

Let each inner feature be a symmetric quadratic with its own input directions:

$$
Q_j=B_j\operatorname{diag}(\nu_j)B_j^\top,
\qquad q_j(x)=x^\top Q_jx,
\qquad f_j(x)=q_j(x)^2.
$$

Its fully symmetric quartic coefficient tensor is

$$
(F_j)_{abcd}=\frac{(Q_j)_{ab}(Q_j)_{cd}+(Q_j)_{ac}(Q_j)_{bd}+(Q_j)_{ad}(Q_j)_{bc}}{3}.
$$

The tensor Frobenius inner product counts all ordered input slots. This is the standard coefficient inner product discussed in [Kolda and Bader's tensor review](https://www.kolda.net/publication/TensorReview.pdf); the following contraction formulas are our application to these quadratic squares, not a recovery theorem from that reference.

The exact Gram matrix is

$$
K_{ij}=\langle F_i,F_j\rangle
=\frac{\operatorname{tr}(Q_iQ_j)^2+2\operatorname{tr}(Q_iQ_jQ_iQ_j)}{3}.
$$

Evaluate the traces using the small cross-reader matrices $B_i^\top B_j$. There is no need to materialize a fourth-order tensor. Negative quadratic eigenvalues are allowed.

For target scalar quartic $T_m$ (the full two-MLP numerator folded into frozen centered-output reader $m$), the exact right-hand side is

$$
C_{jm}=\langle F_j,T_m\rangle
=\sum_{a,b}\nu_{ja}\nu_{jb}\,T_m(b_{ja},b_{ja},b_{jb},b_{jb}).
$$

This uses the existing fully symmetric native contraction oracle. Each rank16 quadratic needs256 basis-pair evaluations. Both native seeds require16,384 contractions in total. These are deterministic weight contractions, not synthetic training examples or activation fitting.

For mixing matrix $A$, the approximating scalar outputs are $\widehat T_m=\sum_j A_{jm}F_j$. The fixed physical output writers are orthonormal under the centered-unembedding metric, so the objective is

$$
\|T-\widehat T\|_F^2
=\|T\|_F^2-2\operatorname{tr}(A^\top C)+\operatorname{tr}(A^\top K A).
$$

Thus $KA=C$ gives the global linear optimum in this fixed feature span. We scale each coefficient feature to unit norm before solving and report the retained Gram rank and normal-equation residual. An eigenvalue cutoff of1e-12 relative to the largest eigenvalue handles numerical rank; if it excludes a meaningful direction, the residual gate can fail. No claim of a global optimum over the input readers follows.

The old assignment uses16 features exclusively for each of two outputs. The new solve lets all32 features serve both outputs, adding32 mixing floats:592,704 fitted floats per seed. Native input normalization, the last-layer denominator, other computational paths and final readout remain outside the candidate boundary.

[Dense controls](COUPLED_QUARTIC_WRITER_V1_CONTROL.json) compare the Gram and target contractions against explicit small tensors and recover a planted two-output mixture, all within1.8e-15. The native experiment requires: (A) normal residual at most1e-8 and nonincreasing exact coefficient objective; (B) at least50% reduction in native write error for both seeds; (C) native write error at most10% for both. All native checks use the previously reused developmental cache, not fresh/OOD data. A passed A with failed B/C means that output optimization in this fixed feature span is insufficient for native extraction. It does not prove that no other mixing would fit this cache, because cache fitting is deliberately not the objective.

Native results belong in `COUPLED_QUARTIC_WRITER_V1_RESULT.json` when execution completes. No circuit is identified by a successful linear solve alone.

## Native result and next nonlinear step

The [native exact solve](COUPLED_QUARTIC_WRITER_V1_RESULT.json) passes its numerical criterion: both Grams retain all32 directions, minimum unit-scaled eigenvalues are0.216 and0.226, and normal residuals are below1.2e-15. The coefficient objective decreases by about7.1e15 for each seed. The reported2.4% is relative to the objective with its target-norm constant omitted; it is **not** a percentage reduction in total coefficient error.

Native write errors increase from55.44/54.89% to56.91/56.31%, failing both improvement and10% fidelity bars. Poor linear-system conditioning or an unfinished output solve does not explain this failure. The fixed input features must change for this weight-objective approach to have another chance at extraction. We have not proven that arbitrary activation-fitted mixing in the same span would fail.

[Gradient controls](COUPLED_QUARTIC_GRADIENT_V1_CONTROL.json) now validate the next objective. For moving reader/eigenweight parameters $\theta$, solve the mixing $A_*(\theta)$ at every step. The envelope differential is

$$
d\mathcal J=\operatorname{tr}(A_*^\top(dK)A_*)-2\operatorname{tr}(A_*^\top dC).
$$

Derivatives of the optimal mixing cancel by the normal equations. On a nontrivial random symmetric target, envelope and fully differentiated solves agree within4.1e-16; finite-difference errors for readers and inner eigenweights are below1.4e-10. This validates the local derivative, not recovery or convergence.

The next one-seed pilot moves32 orthonormal input banks and32 unit-norm signed eigenweight vectors. These constraints remove divergent scale directions while allowing each rank16 quadratic its own subspace. A QR retraction updates input frames and normalization retracts eigenweights to their spheres. Every trial point recomputes exact coefficient contractions and the optimal two-output mixture. Armijo backtracking requires descent; the objective is divided by its fixed initial captured energy. The12-step budget is a pilot, with a separate projected-gradient threshold1e-6. A missed stationarity threshold requires further optimization or diagnosis, not an absent-structure claim. Native directional derivative checks precede fitting, and cached behavioral states are used only for fixed before/after reporting.

The [12-step native pilot](COUPLED_QUARTIC_NONLINEAR_V1_RESULT.json) completed in61seconds, using11.5GB allocated GPU memory. Native directional-gradient error is1.0e-6 and every accepted step decreases the coefficient objective. Captured coefficient energy increases2.21%, below its10% prediction. Native write error falls from56.91% to53.89%. The projected gradient remains0.033, far above1e-6: **this is unfinished optimization**, not evidence that the representation has reached its limit.

The continuation retains the saved factors, fixed initial objective scale, native target and ranks. It uses Polak–Ribière-plus conjugate directions transported by tangent projection, descent restarts and Armijo backtracking on the same Stiefel/sphere product. Retractions, tangent gradients and line searches are standard manifold-optimization tools; see [Boumal's textbook](https://www.nicolasboumal.net/book/IntroOptimManifolds_Boumal_2023.pdf), sections3.6,3.8,4.5 and7.2–7.3. This implementation has no global recovery guarantee. Its [near-start planted control](QUARTIC_MANIFOLD_CG_V1_CONTROL.json) reaches2.6e-7 coefficient error and the1e-6 gradient threshold while maintaining its constraints and monotone objective. The native continuation allows180steps or1000seconds plus finalization. Stationarity, improvement and native fidelity remain separate outcomes; see `COUPLED_QUARTIC_NONLINEAR_V2_RESULT.json` when available.

## A mixed interaction core over shared quadratic intermediates

The square-only model has another restriction: its outer core contains only $q_i^2$. With the same32 inner quadratics, allow the528 products $q_iq_j$ for $i\le j$. For symmetric matrices $A,B,C,D$, direct symmetrization gives

$$
\left\langle\operatorname{sym}(A\otimes B),\operatorname{sym}(C\otimes D)\right\rangle
=\frac{\operatorname{tr}(AC)\operatorname{tr}(BD)+\operatorname{tr}(AD)\operatorname{tr}(BC)+4\operatorname{tr}(ACBD)}{6}.
$$

Here symmetrization averages the six ways of assigning two input slots to each symmetric matrix. The trace formula follows by contracting those six terms; transpose and cyclic trace identities equate the four cycle terms. It reduces to the previous square Gram when $A=B$ and $C=D$. Low-rank reader contractions evaluate it without materializing ambient quartics.

The target contraction similarly becomes

$$
\langle\operatorname{sym}(Q_i\otimes Q_j),T_m\rangle
=\sum_{a,b}\nu_{ia}\nu_{jb}\,T_m(b_{ia},b_{ia},b_{jb},b_{jb}).
$$

[Exact controls](QUADRATIC_PRODUCT_CORE_V1_CONTROL.json) pass at7.4e-16. A planted $q_0q_1$ target is recovered by the mixed core, while these same fixed features restricted to squares give95.6% relative coefficient error. This is a representation counterexample, not evidence about the native model. Squares of *new* linear combinations of quadratics could represent it too; retaining products explicitly preserves reuse of the existing intermediates and may have a different cost under individual rank restrictions.

A shared-node removal zeros all incident products. Joint removal of nodes$i,j$ has a correction equal to their shared cross-edge contribution; adding their separate removal effects would count that edge twice. The control verifies this exact output-space identity. Native denominator and background remain external to these algebraic node interventions.

The native comparison uses the frozen initial11511 bank, independently of the still-running nonlinear fit. The full528-feature solve diagnoses the best coefficient projection in that bank. A sparse32-edge model uses greedy orthogonal pursuit: at each step, choose the edge with largest exact joint-output residual reduction after projecting out previously chosen features. This is a greedy method, not a globally optimal support search. Its [single-edge planted control](QUADRATIC_CORE_GREEDY_V1_CONTROL.json) recovers the correct interaction exactly.

The sparse candidate retains32 outer products and592,704 fitted floats, plus64 edge-index integers; the full core has528 products,593,696 floats and1056 indices. Thus the dense arm is a capacity diagnostic, while the sparse arm is the relevant matched-product comparison. The native predictions require accurate normal equations/nonworsening full coefficient objective; at least10% sparse captured-energy improvement **and** halved native write error; and full-core native error at most10%. None uses text to fit or select edges. See `QUADRATIC_PRODUCT_CORE_NATIVE_V1_RESULT.json` after execution.

### Pilot intervention fidelity and its amplitude red-team

The [frozen pilot native-effects check](QUARTIC_PILOT_NATIVE_EFFECTS_V1.json) reproduces the prior exact-reference scorer bit-for-bit, but fails both swap and removal fidelity. Swap relative errors are1.318,0.736,0.951 and0.801 for agreement verbs, count nouns, past and progressive forms. Removal-CE mean absolute disagreements range0.058–0.226, all above0.02. The lower physical-write error did not establish faithful extraction.

An [executed amplitude/direction check](QUARTIC_PILOT_AMPLITUDE_DIRECTION_V1.json) explains why aggregate write similarity can mislead. Full writes have cosine0.982 with the reference but only0.477 of its norm. Even a diagnostic best scalar multiplier leaves18.7% relative residual. More importantly, paired write changes have cosine−0.102 for verbs,0.824 for nouns,0.875 for past and0.990 for progressive. The corresponding best-scalar residuals are99.5%,56.7%,48.3% and14.0%. None meets10%. This is not just one missing gain: some input-dependent changes point in different directions. These scalar projections are post-hoc geometric diagnostics, not fitted candidates or revised success criteria. The longer weight-only optimization remains independent of these behavioral observations.

## Longer optimization and structural comparisons: 12 September

The [V2 continuation](COUPLED_QUARTIC_NONLINEAR_V2_RESULT.json) ends at its1000-second optimization budget, with170 recorded steps,57.50% more captured coefficient energy than the original exact-writer baseline, and native write error22.81% versus53.89% at continuation start. The derivative/descent and improvement predicates pass; the projected gradient0.02356 misses1e-6. It has **not converged**.

The [same native-effect scorer](QUARTIC_CONTINUED_NATIVE_EFFECTS_V1.json) shows substantial but insufficient progress: swap errors are51.4%,41.9%,36.7%,24.3%; removal-CE disagreements are0.0317,0.0657,0.0274,0.0498. All still miss the original fidelity thresholds. Reference-effect replay is exact. [Conditioning diagnosis](QUARTIC_NONLINEAR_V2_CONDITIONING.json) finds normalized Gram condition88.4 and mixing amplification1.84, versus9.4 and0.84 after the short pilot. There is increased coupling, but not catastrophic near-singularity in this linear subproblem.

The [fixed-bank mixed-core comparison](QUADRATIC_PRODUCT_CORE_NATIVE_V1_RESULT.json) is also complete. The full528-edge solve captures9.08% more coefficient energy than the32-diagonal baseline; greedy32-edge selection captures5.63% more and selects17 cross-intermediate edges. Both numerical solves are sound. Native write errors remain56.19% for the full core and55.62% for sparse32, versus56.91% baseline. The improvement and fidelity predictions fail. This is a narrow negative for the frozen initial bank, not for mixed cores after learning better quadratics. In particular, the dense core closes most of the coefficient gain available to the greedy selector yet still fails native extraction, making support-selection optimization alone a weak immediate next step.

V3 continues the successful-but-unfinished nonlinear route from V2 with the same target, ranks and objective scale. It allows180 more conjugate-gradient steps or1000seconds and separately predicts5% additional capture and gradient norm at most1e-6. No cached native outcomes determine the fitting objective or accepted iterate.

## Apparent reuse and output coordinates, 12 September 00:49

Nonzero weights into both output coordinates do not establish two independent consumers: a common output rotation changes that pattern. The [executed diagnostic](QUARTIC_OUTPUT_REUSE_V1_RESULT.json) searches a dense angle grid and refines its best neighborhood to minimize

$$
\frac{\sum_j K_{jj}\min_{m\in\{1,2\}}(AR)_{jm}^{2}}{\sum_j K_{jj}\|A_{j:}\|^2},\qquad R\in O(2).
$$

Here $K$ is the feature Gram and $A$ the mixing. The numerator discards each node's weaker output branch. Reflections and column exchange leave this loss unchanged. This numerical angle search is not a certified global solve. The nodewise penalty ignores cancellation; we therefore separately refit allowed branch coefficients using the full Gram and measure actual coefficient error against the fitted program. No native data enters the fit.

The planted axis-separable control recovers 1.2e-13 error; rotating the native output frame changes the optimized diagnostic by at most 1.4e-17. V1/V2/V3 minority node-energy fractions are 10.63%, 2.60%, and **1.47%**. The V3 prediction of at least 5% fails. Nevertheless, exclusive branch refits leave 18.85%, 23.27%, and **22.33%** coefficient error. Summed node energies divided by program energy rise from 0.70 to 3.39 to **5.87**: node energies are not additive shares of the program.

The immediate [assignment red-team](QUARTIC_OUTPUT_ASSIGNMENT_V1_RESULT.json) tests every single-node branch flip with an exact refit at the frozen angle. One improving flip lowers V3 error to **21.93%**, after which no single flip improves it. The below-10% repair prediction fails. This excludes that cheap repair, not joint angle/support optimization, simultaneous flips, nonorthogonal coordinates, or different learned quadratics.

Raw two-column support therefore overstates evidence of shared computation, while small secondary weights can still matter through cancellation. These output branches remain coordinates, not identified tasks. No semantic reuse claim follows. The matched nonlinear optimizer remains live, with terminal native extraction scoring next.

## Continued optimization and absolute coverage, 12 September 00:45

The [V3 continuation](COUPLED_QUARTIC_NONLINEAR_V3_RESULT.json) reached its time limit after 1006 fit seconds. Captured coefficient energy increased another **4.15% relative to V2**, missing the registered 5% bar. Its final projected gradient is 0.0923 versus the 1e-6 stationarity bar. Numerical directional checks pass; convergence does not. The run made 169 accepted updates, with 38 descent restarts and 343 line-search objective evaluations.

The [replayed native effects](QUARTIC_NONLINEAR_V3_NATIVE_EFFECTS_V1.json) pass the execution checks but fail swap and removal fidelity. Native write error changes from 22.81% to 24.13%. Swap relative RMS errors are 55.15%, 46.90%, 30.46%, and 24.36% for agreement verbs, count nouns, past, and progressive respectively. Removal-effect mean absolute CE disagreements are 0.0391, 0.0747, 0.0289, and 0.0518 nats. These compare the approximate component with the exact selected component under native background, on the reused developmental cache; they are not OOD or selective-removal evidence. Lower weight error has not monotonically improved these native effects.

The [independent target-norm check](QUARTIC_TARGET_COVERAGE_V1_RESULT.json) resolves an important denominator ambiguity. Gaussian and Rademacher four-slot coefficient probes estimate V2 coverage at **10.86% and 10.77%** of the full-input, frozen **two-output target**, with target-norm relative standard errors of 1.32% and 1.29%. Both distributions pass the projection-identity check and agree within estimated uncertainty. Earlier 57.5% capture gain was relative to the starting projection, not 57.5% of this target and not whole-model coverage.

Using the unchanged objective scale and target, the [executed V3 calculation](QUARTIC_NONLINEAR_V3_COVERAGE_INTERPRETATION.json) gives coverage of **11.31% and 11.21%**, about 94.2% relative Frobenius coefficient error. Estimated coverage standard errors are about 0.15 percentage points; these are descriptive sampling uncertainties, not rigorous bounds. Coefficient error and native write error measure different domains, so their differing magnitudes are not a contradiction.

The strongest immediate methodological alternative is unfinished or inefficient optimization, not absence of structure. The matched limited-memory quasi-Newton run starts from exactly V2 with the same target, rank, scale and budget. It is already live through the managed runner and has exceeded the terminal V3 coefficient objective during fitting; terminal stationarity and native effects remain to be scored. No intermediate cache result selects its accepted iterate. The reusable [effect scorer](quartic_native_effects_v1.py) reproduces the prior V2 receipt exactly in its [replay control](QUARTIC_NATIVE_SCORER_V1_CONTROL.json), so both optimizers will receive the same assessment.

## Independent starts expose a stationary miss, 12 September 00:53

The [four independent-start control](QUARTIC_LBFGS_INDEPENDENT_V1_CONTROL.json) fits a realizable two-output program with two rank2 quadratic squares in8 dimensions, using exact Gram contractions and eliminated linear mixing. Three starts recover relative coefficient error below6e-7. The fourth terminates with projected gradient3.50e-7 but **66.05% error**. Constraint/descent and stationarity predictions hold; all-start recovery fails. Thus even a correct numerical stationarity test is not a global-recovery certificate for this representation.

The immediate [escape control](QUARTIC_LBFGS_ESCAPE_V1_CONTROL.json) applies four independently sampled unit tangent directions at each perturbation magnitude0.01,0.1,0.5 and repeats fitting. All12 return to approximately the same66.05% error, failing the registered recovery prediction. The failed endpoint is saved separately. [Gram diagnosis](QUARTIC_LBFGS_STATIONARY_MISS_DIAGNOSIS.json) gives condition1.045 and fitted-feature cosine0.022, excluding a badly conditioned output solve in this small case. One feature matches a true feature at0.9998 coefficient cosine; the other is a wrong signed approximation. This is evidence of an optimization basin problem, not a proof classifying the stationary point.

Independent restarts succeeded where these local perturbations failed. Native fits therefore need more than long local optimization before a negative result can be read as absent structure. The live native comparison has not yet reached stationarity; its terminal result still comes first. These controls use synthetic weights only and add no language-data fitting.

## Matched optimizer endpoint, 12 September 00:56

The [limited-memory quasi-Newton run](COUPLED_QUARTIC_LBFGS_V1_RESULT.json) completed180 updates in814.6 total seconds. Relative to the same saved V2 starting point, coefficient capture increases **11.83%**, versus4.15% for conjugate gradient under the same180-step/1000-second caps. The numerical and5% improvement bars hold. The final gradient is0.190, missing1e-6 stationarity; this is iteration-limited, not converged.

The [identical frozen native scorer](QUARTIC_LBFGS_V1_NATIVE_EFFECTS_V1.json) passes reference and write replay, but fails swap/removal fidelity. Native write error rises from22.81% to **26.65%**. Swap errors are65.51%,55.57%,35.64%,34.06% across A1/A2/past/progressive, and removal-effect CE disagreements are0.0427,0.0925,0.0315,0.0553nats. This is improved coefficient optimization without an extraction improvement. Multiplying V2 target-coverage estimates by1.118343 gives approximately12.14%/12.04% coverage of the same two-output coefficient target; it remains far from a whole-model explanation.

The next managed continuation starts this saved endpoint, retains the same ranks, target and objective scale, and allows720 steps or3600 fit seconds. Quasi-Newton memory is explicitly reset because V1 did not serialize it. Predicted numerical descent,5% additional capture and1e-6 stationarity remain separate bars. No cache score selects a fitting iterate. Longer optimization addresses an unresolved confound; independent starts and other structures remain necessary before a negative structural conclusion. This run adds no data-guided objective and does not repair the failed causal checks by definition.
