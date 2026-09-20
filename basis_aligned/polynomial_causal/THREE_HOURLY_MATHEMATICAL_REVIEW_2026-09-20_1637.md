# Three-hour mathematical, literature, organization, and efficiency review — 2026-09-20 16:37 UTC

Review interval: since2026-09-20 13:36; research focus explicitly redirected at15:10. The user's two-day decomposition-science instruction continues through September22,15:10UTC and supersedes switching back to causal screens. Goal remains a predictive, extractable, selectively manipulable, composable and stable simple circuit representation. None of the current weight fits satisfies that full goal.

## Decision

Continue the three discriminating native studies already queued: full-input shared-bilinear quartic fitting with paired parameter scaling; matrix-free symmetric quadratic ALS; and a quartic trace/radial baseline. Do not launch another broad rank sweep before interpreting them. Optimization, structural capacity and metric weighting are now separately measurable. The main new mathematical consequence is an exact decomposition of Gaussian quartic error into trace-related components, with a large dimension-dependent metric condition number.

## Current mathematical objects and prices

The folded quadratic numerator is

$$
f(z)=C[(Az)\odot(Bz)],\quad C=UD,\quad A=LE,\quad B=RE.
$$

Here the original concatenated input has6912 coordinates; exact thin QR frames reduce relevant input and output coordinates to1152 each for the isotropic study. Frame equivalence is Euclidean; changing covariance or physical coordinate scaling changes the metric. RMS normalization, attention normalization and final logit softcap remain explicit outside the polynomial identity. Dense width-k factors store3·1152·k reduced values; expanded output writers or shared frame storage must also be counted.

The new full-input quartic target is the pure MLP16→MLP17→unembedding composition, including the carry multiplier. All four input slots receive the same1152-dimensional vector. It excludes other intervening residual/attention terms and is not the complete two-block function. Its fully input-symmetric coefficient tensor H has one1152-dimensional reduced output index and four input indices. Querying H uses the three pairings of symmetric first-layer coefficients, not merely the original tree grouping.

The student is

$$
q_a(x)=(A_ax)(B_ax),\qquad
\widehat f_v(x)=\sum_{k=1}^{128}C_{vk}(L_kq)(R_kq).
$$

Banks of128/512 quadratics cost475,136/1,458,176 reduced scalar parameters. Expanded vocabulary writers make these6,766,592/7,749,632 values if the QR frame is folded into C instead of stored. Neither accounting makes output frames free. Cores are dense; fixed widths do not constitute sparse interaction recovery. Gauge freedoms include channel scaling, permutations and basis changes when the chosen family permits them. The local shared-quadratic-bank experiments have additional arbitrary mixing freedoms.

## Evidence since the previous review

Five planted families established functional recovery controls and exposed restart/rate dependence. A sparse-basis stage recovered planted interaction structure, but individual features remained nonunique. Native local quartic sparsification replicated across16 independently fitted contexts; common-factor approximations reached4.0% median error at75values. Frozen transfer to176other rows failed. An interface audit showed input ports and output readers change with the row, so that failure concerns the fixed amplitude interface, not impossibility of global residual-space reuse.

Full quadratic random fits retained high Frobenius error. Width1024 teacher-channel selection with exact output refit reached82.02% versus95.48% for the earlier random fit, demonstrating an optimization gap. Native covariance-informed objectives improved separate actual-input diagnostics while worsening isotropic error; including measured means did not guarantee better evaluation. Variable projection failed to beat the width512 teacher-channel baseline at its tested settings. Exact toy ALS blocks were monotone but sometimes stalled at9–13% error despite representability.

The full-input quartic coefficient oracle passed native polynomial replay6.19e-7 and float64 comparison5.12e-7. Uniform and collision-stratified energy estimates agreed within1.05 estimated SE. All-distinct indices contributed98.74% of estimated coefficient energy. Sparse counterexamples showed uniform sampling can miss all mass; equal stratification helps diagonal spikes but hurts rare distinct spikes.36 matched toy fitting controls showed sampling noise can both help and obstruct convergence.

## Primary literature: exact mappings and limits

Variable projection: [O'Leary and Rust](https://www.cs.umd.edu/users/oleary/software/varpro.pdf) eliminate linear coefficients conditional on nonlinear parameters. Our output factor is that linear block. The reduction is exact without the numerical ridge, but does not certify global recovery by Adam/Muon. Input-feature nonlinear optimization remains unresolved.

Stable ALS: [Minster et al.](https://arxiv.org/abs/2112.10855) use QR/SVD to improve numerical stability of ordinary CP block solves. We implemented SVD on small symmetric coefficient designs; native symmetric blocks use a derived matrix-free normal operator instead. This difference is explicit. Native CG residuals and rejected sweeps are recorded because normal-equation conditioning still matters.

Convergence assumptions: [Hu et al.](https://arxiv.org/abs/2505.14037) analyze ordinary CP-ALS for orthogonal/incoherent decompositions. Our native factors have not been shown to satisfy those hypotheses, and our repeated-input symmetrization changes the parameter redundancy. No convergence rate or identifiability guarantee is imported.

Gaussian polynomial structure: [Tóth's Gaussian Hilbert-space notes](https://math.bme.hu/~balint/prague_11/ghs_090317.pdf) describe orthogonal Wick polynomial spaces and their symmetric-tensor norm relation. Applied to our finite standard-Gaussian quartic, this yields the independently derived and checked trace identities below. Native activation distributions need not obey Gaussian moments.

Hankel/Waring alternative: [Brachat et al.](https://arxiv.org/abs/0901.3706) characterize symmetric decomposition into powers of linear forms using Hankel matrices. This maps to each homogeneous scalar output polynomial, but its atom family differs from shared quadratic DAG features. Low Waring rank is not established. A straightforward degree-two monomial space in1152 variables has664,128 homogeneous coordinates; dense Hankel constructions are already prohibitive. We do not claim a general efficient solver or uniqueness for our native shared circuit problem from this connection. Small planted power-sum controls remain a possible later comparison, below the current higher-information studies.

## Executable mathematical consequence: separate quartic trace effects

For symmetric H, define

$$
J_{vkl}=\sum_i H_{viikl},\qquad t_v=\sum_kJ_{vkk}.
$$

For standard Gaussian x, direct Wick expansion gives

$$
\mathbb E\|f(x)\|^2
=24\|H\|_F^2+72\|J\|_F^2+9\|t\|^2.
$$

The orthogonal function components are

$$
f_0=3t,\quad f_2=6(x^\top Jx-t),\quad
f_4=f-6x^\top Jx+3t.
$$

Vector-output notation applies separately to each J_v. Thus coefficient Frobenius matching measures the highest Wick-degree component; Gaussian function error also weights contractions. The same identity applies to H minus a student coefficient tensor, so it diagnoses approximation error as well as teacher energy.

For a fully symmetric quartic coefficient, the Gaussian/Frobenius metric eigenvalues on harmonic quartics, radius-squared times harmonic quadratics, and radial quartics are respectively

$$
24,\qquad12(d+6),\qquad3(d+4)(d+6).
$$

At d=1152 these are24,13,896,4,015,944: squared-metric condition167,331, norm condition409.06. These are exact isotropic homogeneous-polynomial relations, not native data covariance bounds. Relative errors additionally depend on the target's direction in this space. A metric discrepancy can be large without a bug or a change in representational capacity.

Independent exact-degree Gaussian quadrature verified the trace identity, orthogonality and implicit factor-contracted mean to<7e-16. A full small-dimensional metric eigendecomposition verified the spectrum to1.14e-13 absolute error.

For the native quartic, the mean can be contracted from first-layer quadratic Gram matrices without H. The rotation-invariant homogeneous baseline is

$$
f_{\mathrm{rad}}(x)=\frac{\mathbb E f(x)}{d(d+2)}\|x\|^4.
$$

This supplies a compact degree-four baseline, rather than a constant-only comparison outside the homogeneous family. The native diagnostic is implemented and queued: exact mean,4,096 new Gaussian inputs, radial error and final student mean mismatch. It will determine how much apparent function behavior is explained by traces.

## Complexity and identifiability conclusions

Exact coefficient-query cost scales with queried batches and native factor contractions rather than expanded tensor size. It does not remove sample-variance concerns or guarantee small-error certification. The quadratic matrix-free input normal operator uses width² and width×1152 matrices rather than a full tensor design; its application still carries iterative-solver and conditioning costs. Neither method chooses a semantic basis automatically.

A bounded-rank HT tree can represent compact computations, but tree grouping and repeated-input identities matter; shared DAG nodes admit additional reuse. The earlier sparse-core and common-factor results distinguish dictionary width from interaction count. The executed interface counterexamples show that restrictions to moving low-dimensional ports cannot uniquely identify a global circuit.

## Ranked next actions and falsifiers

1. Interpret global quartic V1/V2 fits jointly with their finite-query output-rank bound and the trace/radial diagnostic. Falsifier for useful compression at these settings: heldout coefficient error stays near1 under both parameterizations, with no gain beyond architecture bounds. This would reject the tested settings/family budget, not all hierarchical circuits.
2. Interpret native symmetric ALS against the same-width teacher-channel baseline. Large CG residuals or repeated rejected sweeps indicate solver limitations; accurate solves with poor error leave nonlinear basin/capacity questions. Do not count guarded monotonicity as a discovery.
3. If metric components explain the discrepancy, test an explicitly weighted coefficient-plus-trace objective with both isotropic and covariance versions. Do not choose weights solely to produce a favorable headline. If traces do not explain it, revisit implementation/estimation before more training.
4. Only after a repeatable global component appears, return to its functional specification, extraction and selective interventions. The user has deferred that data-based stage during this two-day focus, so no new causal screen is launched now.

## Organization, scheduling and efficiency audit

Canonical artifacts remain in `direct_tensor_match/README.md`; timed user reports are linked from explanations/README.md; the computation-path registry records the relevant scope and failures. The interface clarification was appended to the original failed-transfer report. Raw JSON, runnable scripts and exported factors were committed and pushed, with larger tensors in LFS. Unrelated shared-worktree changes and GPU jobs were preserved.

Measured recent compute: full-quadratic V2~114seconds; native moment sweep~66seconds; teacher-channel baseline3.8seconds; variable projection10.5seconds; full quartic query1.9seconds;36 sampled-loss toys21seconds;27 ALS controls11seconds. CPU derivation/testing continued during the shared GPU queue. The current bottleneck is waiting behind a verified live external job, not a dead process; no duplicate native runs were launched. Validation targeted actual metric, gradient, frame and numerical failure modes rather than deployment ceremony. No additional refactor currently outranks interpreting the queued results.

Continuation receipt: quartic trace identities and spectrum checks executed; native radial/mean diagnostic queued. Keep the full goal active. The next mathematical review is due three hours after this completed review; the next hourly strategic review remains separately scheduled, with the user's decomposition-focus override retained.

## Results arriving at the review boundary and revised continuation

At16:37–16:38 the queue advanced. Both global quartic sweeps completed: coefficient errors remained approximately100%, under both parameter scales. The finite-query output-rank bound was78.44%; this is only a relaxation, not a guarantee the shared-DAG family can attain it. Matrix-free quadratic ALS reached94.54% at width128 and89.89% at width512, without rejected sweeps. It improved substantially on random gradient fits but narrowly failed the89.80% teacher-channel comparison at width512.

The Gaussian diagnostic completed: estimated teacher energy fractions were19.2% constant,43.4% second-Wick-degree,37.4% fourth-Wick-degree. The radial baseline error90.04% narrowly missed its preregistered90% bar; that tiny margin is not a robust separation given sampling uncertainty. Mean mismatch explained about19% of failed students' residuals, not the predicted majority. These results replace the earlier pending status; the shared GPU wait did not represent a terminal blocker.

The next executed discriminating control was gradient noise. On an exactly representable rank-one quartic at d1152,256ordered entries produced relative gradient RMSE4.26million for one specified random orientation. Removing self-norm sampling reduced but did not resolve the noise. This is a constructive warning, not proof of the native failure's cause.

Priority now shifts to exact-gradient quartic CP contraction before any larger stochastic sweep. Implemented `quartic_cp.py` computes the exact student self norm and native two-layer teacher cross term; independent dense24-permutation checks passed values9.19e-16 and gradients1.27e-15. The teacher norm is an additive constant, so full teacher-norm evaluation is not required for exact optimization gradients. This is a different student family with different reuse/capacity and storage, which must be stated in comparisons. Plan: `direct_tensor_match/EXACT_QUARTIC_CP_PLAN_V1.md`.
