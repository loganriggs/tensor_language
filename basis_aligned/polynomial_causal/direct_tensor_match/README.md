# Direct weight-tensor decomposition study

[Single-layer quadratic baseline and Gaussian mean-energy correction](ORTHOGONAL_INTERACTION_INTERPRETATION_V1.md), 22 September 03:07 UTC.

[Independent output-local spectral baseline](OUTPUT_LOCAL_BILINEAR_INTERPRETATION_V1.md): exact restricted optimum; adaptive product allocation exposes pooled versus per-output tradeoff.

[Shared output blocks and residual-aware channel baseline](OUTPUT_ROTATED_BLOCK_INTERPRETATION_V1.md), 22 September 03:17 UTC: output basis changes improve coefficient reconstruction; absolute fidelity and cost remain inadequate.

[Gaussian paired-response objective](PAIRED_GAUSSIAN_CP_INTERPRETATION_V1.md), 22 September 03:25 UTC: exact checked kernels; native fixed-bank readout improvements fail.

[Calibration-moment diagnostic](MIXED_CP_CALIBRATION_MOMENTS_INTERPRETATION_V1.md), 22 September 03:31 UTC: same features, improved data-based readout; component-response gate still fails.

[Learned response-loss toy comparison](PAIRED_RESIDUAL_LEARNING_INTERPRETATION_V1.md), 22 September03:42; [duration caveat for pending native fit](LOCAL_QUARTIC_RESIDUAL_DURATION_CAVEAT_V1.md). No new native sweep justified by these toys.

[Exact product reassociation and exported DAGs](QUARTIC_PRODUCT_REASSOCIATION_INTERPRETATION_V1.md), 22 September03:50:9/10additional products saved after reader sharing, unchanged candidate function and native limitations.

[Approximate quadratic reuse](CP_QUADRATIC_REUSE_INTERPRETATION_V1.md), 22 September03:59: larger edit fails finite-response/per-output preservation; smaller edit exposes a computation–fidelity tradeoff.

## Current direction — 22 September 01:59 UTC

The current pure-quartic MLP16→MLP17 branch compares fixed 16-output programs, not a full-model replacement. [Larger evaluation](../explanations/for_logan/research_update_2026-09-22_0112_larger_panel_results.md) confirms good pooled CP error but poor smaller-output fidelity. [Conditional programs and graph simplification](../explanations/for_logan/research_update_2026-09-22_0134_conditional_program_baseline.md) reduce stored coefficients and measured CPU time; native finite-removal tests failed absolute fidelity and parent retention. [Result](../explanations/for_logan/research_update_2026-09-22_0241_conditional_removal_results.md). Balanced producer fitting completed: integrity passed, small-feature improvement and response-retention criteria failed. [Result](../explanations/for_logan/research_update_2026-09-22_0211_balanced_features_failed.md). No semantic, OOD or composition adoption follows yet.

[Current mathematical review and harmonic-truncation failure](../THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-22_0159.md). The broader midpoint branch below remains a separate target and baseline.

## Historical direction — 21 September 00:18 UTC

The broader midpoint target retains all previous-MLP-polynomial-dependent self/cross terms at the last MLP. The 16-product program `MIDPOINT_EXTRACTED_PROGRAM_V1.pt` passed new-document removal and same-token swap confirmation. It remains an operational feature program: upstream state/normalization are explicit, semantic selectivity is untested, and four-feature discovery failed split-stability criteria.

`MIDPOINT_JOINT_CORE_REFIT_V1.pt` shares input dictionaries, reducing reader coefficients from 36,864 to 18,688 at higher native swap error; it lacks fresh confirmation. Joint eight-product CP fits and exact paired-input moment refitting improve a smaller candidate's transfer but fail the registered preservation threshold. Product identities are nonunique in an exact baseline control and unstable across CP restarts. Keep these failures separate from successful function matching.

[Confirmed program](../explanations/for_logan/research_update_2026-09-20_2356_midpoint_confirmation.md) · [Shared graph](../explanations/for_logan/research_update_2026-09-21_0003_shared_midpoint_graph.md) · [Identity limits](../explanations/for_logan/research_update_2026-09-21_0009_midpoint_identity_limits.md) · [Joint products and moments](../explanations/for_logan/research_update_2026-09-21_0018_joint_products_and_moments.md). Earlier “next” entries below are historical.

The signed tied-reader alternative `MIDPOINT_TIED4_PROGRAM_V1.pt` stores 18,432 input coefficients with 16 products and passes reused-panel swap preservation. It does not halve projection arithmetic. Conditional product rediscovery fails, partly because magnitude truncation changes the positive/negative mode inventory. [Tied-reader report](../explanations/for_logan/research_update_2026-09-21_0027_tied_readers_and_signed_cutoffs.md).

Primary research focus: **20 September 15:10 UTC to 22 September 15:10 UTC**, per user direction. [Scope and questions](FOCUS_2026-09-20_TO_2026-09-22.md). This replaces the prior head/branch omission agenda for this period.

We optimize randomly initialized student computations directly against the folded teacher's weights. Exact coefficient losses include symmetric Frobenius and Gaussian function norms; correlated-Gaussian metrics can incorporate covariance estimated from calibration data. No sampled activation targets are used to train the isotropic decompositions.

[Initial scientific findings](INITIAL_FINDINGS_2026-09-20_1522.md) · [Plots](INITIAL_RESULTS_V1.png) · [Original preregistration](PLAN.md) · [Native pilot plan](NATIVE_PREREGISTRATION_V1.md).

| Artifact | Status and scope |
|---|---|
| CORE_CHECK_V1.json | Exact symmetrized/Frobenius, Gaussian quadrature, gradients, repeated-input cancellation passed |
| IMPLICIT_QUADRATIC_CHECK_V1.json | Implicit factor inner products/gradients agree with dense reference |
| COVARIANCE_CHECK_V1.json | Correlated Gaussian quadrature and singular-covariance blindspot passed |
| TOY_SWEEP_V1.json | 120 fits, five structures, two optimizers, two rates, three restarts, matched/wide-sparse variants |
| PLANTED_STRUCTURE_AUDIT_V1.json | Separates functional recovery, feature-space recovery and core support |
| TOY_COVARIANCE_SWEEP_V1.json | 80 fits, isotropic versus synthetic-calibration covariance, matched/wide unpenalized variants |
| NATIVE_QUARTIC_PILOT_V1.json | 96 fits on one native quartic target, two objectives; initial plan prose mistakenly said192 |
| NATIVE_QUARTIC_AUDIT_V1.json | Independent dense polynomial replay and same-target spectral baseline |
| NATIVE_QUARTIC_CONTEXTS_V1.json | Complete:384fits across16contexts; independent per-context fits |
| TREE_RESTARTS_V1.json | Complete:48fits;10,000step matched-rank tree optimization controls |
| NATIVE_FULL_QUADRATIC_V1.json | Complete:12fits; full last-layer joint numerator, CP widths32/128/512 |

JSON snapshots may be partial while a run is live. Read record counts and process/queue state before reporting a completed sweep. Thresholded weights do not establish deployable sparse storage. Native quartic generators and normalized model boundaries remain external dependencies.

Next questions: matched-rank optimization traps, explicit support pruning/refit, sparse-basis selection within a recovered subspace, covariance conditioning versus objective choice, and native activation-covariance metrics. Do not replace these with further data-based circuit screens during the focus period.

Next managed job: NATIVE_FULL_QUADRATIC_V2,24fits comparing objective/rate/width with an explicit radial baseline.

Latest: [Sparse basis recovery and paired covariance conditioning](BASIS_AND_CONDITIONING_2026-09-20_1536.md). Hard four-entry Tucker program improves11.985%to0.00629%error after function-preserving basis search. Paired-coordinate whitening helps somequadratic cases but hurtsquartics at thetestedrate. Nativecovariance capture is queued.

## Sparse quartic bases and full-tensor metric controls — 15:49 UTC

[Timed report for Logan](../explanations/for_logan/research_update_2026-09-20_1548_sparse_bases_and_metric_controls.md) explains the joint tensor, shared computations, baselines and metric distinctions with equations and a flowchart.

- `QUARTIC_BASIS_PLAN_V1.md`, `quartic_sparse_basis.py`, `quartic_basis_sweep.py`, `QUARTIC_BASIS_SWEEP_V1.json`:24 fixed-span gauge searches; planted3 products error9.55e-5; native4 products4.81%.
- `export_quartic_programs.py`, `SPARSE_QUARTIC_PROGRAMS_V1.json`: independently executable exports and exact-degree Gaussian quadrature validation.
- `NATIVE_FULL_QUADRATIC_V2.json/.pt`:24 full quadratic fits; radial Gaussian baseline58.95% error; width1024 Gaussian fit56.47%, best coefficient Frobenius fit95.48%.
- `NATIVE_COVARIANCE_V1.json/.pt`: native calibration/evaluation rows and moments in a precisely saved coordinate frame; nonzero means retained.
- `noncentral_quadratic.py`, `check_noncentral.py`, `NONCENTRAL_CHECK_V1.json`, `NATIVE_METRIC_PLAN_V1.md`: next16 native moment-metric fits, including both isotropic and measured covariance/mean, with heldout empirical diagnostics.

## Replication and native metric results — 15:57 UTC

[Timed scientific report](../explanations/for_logan/research_update_2026-09-20_1557_replication_covariance_and_optimization.md), with [figure](REPLICATION_AND_METRICS_V1.png) and [PDF](REPLICATION_AND_METRICS_V1.pdf).

- `QUARTIC_CONTEXT_BASIS_PLAN_V1.md`, `quartic_context_basis.py`, `QUARTIC_CONTEXT_BASIS_V1.json`:32 gauge fits,16 independent contexts; four-product error median31.98%→4.82%,15/16 improve≥2×. All-product error invariant.
- `NATIVE_METRIC_SWEEP_V1.json/.pt`:16 native fits; width512 evaluation error35.76% isotropic versus15.76% second-moment Gaussian. Noncentral Gaussian improves its own objective with width but worsens evaluation; isotropic errors explicitly retained.
- `NATIVE_CHANNEL_BASELINE_PLAN_V1.md`, `NATIVE_CHANNEL_BASELINE_V1.json`:18 teacher-channel selection/output-refit controls; width1024 Frobenius82.02% versus random-start95.48%. Exact Gram/implicit agreement and least-squares residual controls passed.
- `VARIABLE_PROJECTION_PLAN_V1.md`:8 next random-start fits eliminate output weights through differentiable ridge solves, testing the demonstrated optimization gap.

## Stability, common factors, and failed row transfer — 16:03 UTC

[Timed report](../explanations/for_logan/research_update_2026-09-20_1603_common_factors_and_transfer_failure.md).

- `QUARTIC_STABILITY_PLAN_V1.md`, `audit_quartic_stability.py`, `QUARTIC_STABILITY_AUDIT_V1.json`:32 exports pass float32 and perturbation bars; individual feature agreement fails.
- `audit_shared_structure.py`, `SHARED_STRUCTURE_AUDIT_V1.json`: planted common factor correlation≥0.999999997 across12 searches; native support graphs agree12/16.
- `COMMON_FACTOR_PLAN_V1.md`, `common_factor_sweep.py`, `COMMON_FACTOR_SWEEP_V1.json`:136 random-start common-factor/quotient fits; native median4.00%,worst14.55%,75values.
- `audit_common_factor_baselines.py`, `COMMON_FACTOR_OUTPUT_BASELINES_V1.json`:74-value rank-one output control loses to common-factor model on all16 targets.
- `COMMON_FACTOR_TRANSFER_PLAN_V1.md`, `common_factor_transfer.py`, `COMMON_FACTOR_TRANSFER_V1.json`: all selected programs independently validated by quadrature; frozen transfer to176other rows fails (median100.10%). Oracle quotient refit46.96% versus random-factor81.78% is reuse-capacity evidence only. Exports include factors and original scale.

## Interface audit and global quartic queries — 16:11 UTC

[Mathematical note](INTERFACE_AND_GLOBAL_QUARTIC_2026-09-20_1611.md) bounds the row-transfer failure: native input ports and output readers both change. `TRANSFER_INTERFACE_AUDIT_V1.json` gives a same-circuit coordinate-transport positive control and a rectangular-port nonidentifiability counterexample.

`implicit_quartic.py` queries exact fully symmetrized two-layer coefficients without expanded tensors. `check_implicit_quartic.py` / `IMPLICIT_QUARTIC_CHECK_V1.json` independently validate values, gradients and Frobenius enumeration; a sparse sampling adversary exposes high variance. `NATIVE_QUARTIC_QUERY_PLAN_V1.md` preregisters the queued full1152-input native estimator pilot. No native stochastic fit yet.

## Native global query results and sampled optimization — 16:20 UTC

[Timed report with equations and flowchart](../explanations/for_logan/research_update_2026-09-20_1620_global_quartic_and_sampling.md).

- `VARIABLE_PROJECTION_V1.json/.pt`:8 full quadratic fits; width512 best95.21% Frobenius error, failed89.80% teacher-channel bar; numerical replay/conditioning held.
- `NATIVE_QUARTIC_QUERY_V1.json`:full1152-input native pure quartic oracle passed replay6.19e-7/precision5.12e-7. Uniform and stratified energy within1.05 estimated SE;98.74% energy in all-distinct tuples.
- `check_stratified_quartic_sampling.py`, `STRATIFIED_QUARTIC_CHECK_V1.json`:equal-budget dense/diagonal/paired/distinct sampling controls; equal strata helps diagonal and hurts distinct spikes.
- `STOCHASTIC_QUARTIC_TOY_PLAN_V1.md`, `stochastic_quartic_toys.py`, `STOCHASTIC_QUARTIC_TOYS_V1.json`:36 paired-initialization coefficient-loss fits; noise can both help escape and obstruct recovery.
- `NATIVE_STOCHASTIC_QUARTIC_PLAN_V1.md`:queued V1/V2 full-input random shared-bilinear students with paired output-parameter scaling, independent coefficient and function diagnostics. Dense cores, no sparsity or circuit identity claim.

## Parameter geometry and symmetric ALS — 16:28 UTC

[Literature-to-algorithm note](ALS_AND_PARAMETERIZATION_2026-09-20_1628.md) maps primary variable-projection/ALS sources to our actual symmetric polynomial objective, without transferring unverified convergence assumptions.

- `check_quartic_parameterization.py`, `QUARTIC_PARAMETERIZATION_CHECK_V1.json`: paired initial function/gradient chain rule and finite-query rank-bound controls.
- `symmetric_quadratic_als.py`, `quadratic_als_toys.py`, `QUADRATIC_ALS_PLAN_V1.md`, `QUADRATIC_ALS_TOYS_V1.json`:27 paired ALS/Adam/Muon toy fits. Monotone exact blocks can still stall; no universal optimizer winner.
- `matrix_free_quadratic.py`, `check_matrix_free_quadratic.py`, `MATRIX_FREE_QUADRATIC_CHECK_V1.json`: exact symmetric input normal operator validated against dense design, proximal CG solution error1.66e-11.
- `MATRIX_FREE_NATIVE_ALS_PLAN_V1.md`: queued native block-solve pilot, explicitly reports residuals and rejected sweeps.

## Three-hour mathematical review — 16:37 UTC

[Review](../THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-20_1637.md) separates capacity, optimization, metric and interface assumptions. `quartic_gaussian.py`, `check_quartic_gaussian.py`, `QUARTIC_GAUSSIAN_CHECK_V1.json` validate the exact quartic trace-energy identity and implicit mean. `check_quartic_metric_spectrum.py` / `QUARTIC_METRIC_SPECTRUM_V1.json` verify the Gaussian/Frobenius eigenvalues. `NATIVE_QUARTIC_GAUSSIAN_PLAN_V1.md` preregisters the queued radial/mean diagnostic.

## Review-boundary outcomes and exact-gradient continuation

- `NATIVE_STOCHASTIC_QUARTIC_V1.json/.pt`, `NATIVE_STOCHASTIC_QUARTIC_V2.json/.pt`:all16 global sampled fits stay near100% coefficient error. Finite-query output-rank relaxation78.44% is not an attainability claim.
- `MATRIX_FREE_NATIVE_ALS_V1.json/.pt`:width51289.89% coefficient error, no rejected sweeps; solver residual bar held, teacher-channel89.80% bar narrowly failed.
- `NATIVE_QUARTIC_GAUSSIAN_V1.json`:estimated native energy fractions19.2/43.4/37.4% in Wick degrees0/2/4; radial error90.04%; mean mismatch is not the majority of failed-student residual.
- `check_quartic_gradient_noise.py`, `QUARTIC_GRADIENT_NOISE_V1.json`:representable rank-one toy exposes huge sampled-gradient error despite tolerable scalar-loss estimation. Native cause remains a hypothesis.
- `quartic_cp.py`, `check_quartic_cp.py`, `QUARTIC_CP_CHECK_V1.json`, `EXACT_QUARTIC_CP_PLAN_V1.md`:exact symmetric CP quartic self/cross contractions and gradients independently verified; next route removes coefficient-gradient sampling.

## Exact-gradient component competition — 17:00 UTC

[Timed report](../explanations/for_logan/research_update_2026-09-20_1700_exact_gradients_and_component_competition.md) defines the exact symmetric CP objective and separates rank capacity from optimizer failure.

- `EXACT_QUARTIC_CP_TOY_PLAN_V1.md`, `exact_quartic_cp_toys.py`, `EXACT_QUARTIC_CP_TOYS_V1.json`:16 fits, d6/32/128/1152; exact gradients still stall.
- `exact_cp_fit.py`, `EXACT_CP_RESIDUAL_PLAN_V1.md`, `exact_cp_residual_controls.py`, `EXACT_CP_RESIDUAL_CONTROLS_V1.json`:same rank2/1500steps,8 fits; joint duplicates strong atom (~34%error), residual Adam<4e-7, Muon<1.6e-4.
- `EXACT_CP_SCALE_CONTROL_PLAN_V1.md`, `exact_cp_scale_control.py`, `EXACT_CP_SCALE_CONTROL_V1.json`:four same-direction/unit-raw-scale joint controls remain33.743%; scale alone does not explain success.
- `EXACT_CP_SEPARATION_PLAN_V1.md`, `exact_cp_separation.py`, `EXACT_CP_SEPARATION_V1.json`:18 fits on amplitude/overlap grid; residual wins separated cases but loses badly on some overlaps. One initialization per cell; no universal strategy ranking.
- `NATIVE_EXACT_CP_GREEDY_PLAN_V1.md`:queued native eight-atom exact-gradient residual pilot; heldout coefficient/Gaussian checks, estimated teacher norm only for scaling.

## Refinement-rate correction — 17:05 UTC

[Timed correction](../explanations/for_logan/research_update_2026-09-20_1705_refinement_overshoot.md): the two amplitude0.5 overlapping failures are repaired at the same300-step budget by reducing refinement rate0.05 to0.005. Initial errors0.65%/11.78% had been worsened to44.65%/24.08%; smaller rate reaches numerical precision. Historical endpoints remain intact; do not infer structural impossibility or a universal joint advantage.

- `RESIDUAL_OBSTRUCTION_PLAN_V1.md`, `residual_obstruction.py`, `RESIDUAL_OBSTRUCTION_V1.json`: exact residual output-rank relaxation,12 refinements across longer, perturbed and oracle initializations; longer joint runs recover both.
- `REFINEMENT_RATE_PLAN_V1.md`, `refinement_rate.py`, `REFINEMENT_RATE_V1.json`:nine same-budget refinement rates, initial/best/final training losses. One teacher orientation and initialization; no heldout selection or native result.

## Native exact CP and quartic covariance — 17:14 UTC

[Timed report](../explanations/for_logan/research_update_2026-09-20_1714_native_cp_null_and_covariance.md).

- `NATIVE_EXACT_CP_GREEDY_V1.json/.pt`:8atoms capture0.0863% estimated coefficient energy;99.9356% independent coefficient error,100.1773% Gaussian error. Both improvement predictions failed; finite check passed.
- `audit_native_cp_structure.py`, `NATIVE_CP_STRUCTURE_V1.json`:Gram condition3.58; no cross-atom linear factor correlation>.99. Descriptive, not a no-sharing theorem.
- `cp_dictionary.py`, `check_cp_dictionary.py`, `CP_DICTIONARY_CHECK_V1.json`:conditional exact writer-refit selection matches exhaustive small-dictionary evaluation.
- `NATIVE_CP_DICTIONARY_PLAN_V1.md`:queued native-channel vs random1024candidate dictionaries, retainedwidths1/2/4/8.
- `gaussian_cp.py`, `check_gaussian_cp.py`, `GAUSSIAN_CP_CHECK_V1.json`:105pairing exact quartic Gaussian Gram, full covariance; independent quadrature/gradient/whitening checks below3e-15.
- `QUARTIC_METRIC_TRADEOFF_PLAN_V1.md`, `quartic_metric_tradeoff.py`, `QUARTIC_METRIC_TRADEOFF_V1.json`:12known-undercapacity fits, three metrics, two rates/starts. Covariance2.07% vs coefficient89.45%; fixed teacher-atom covariance baseline1.992%. No empirical-eighth-moment or native covariance claim.

## Hierarchy capacity control — 17:21 UTC

[Timed report](../explanations/for_logan/research_update_2026-09-20_1721_hierarchy_capacity_and_native_baseline.md).

- `HIERARCHY_CAPACITY_CONTROL_PLAN_V1.md`, `hierarchy_capacity_control.py`, `HIERARCHY_CAPACITY_CONTROL_V1.json`: radial quartic symmetric unfolding rank d(d+1)/2; each symmetric flat CP atom rank≤6. At d1152,8atom coefficient error≥81.61% despite exact cheap shared square. Dense spectrum/replay checks pass.
- `SHARED_SQUARE_FIT_PLAN_V1.md`, `shared_square_fit.py`, `SHARED_SQUARE_FIT_V1.json`:12 direct exact-loss fits of supplied diagonal shared-square hypothesis; rate0.05 recovers coefficients modulo sign to≤1.5e-9, rate0.01 often remains at few-percent error.
- `quartic_root_features.py`, `check_quartic_root_features.py`, `QUARTIC_ROOT_FEATURE_CHECK_V1.json`: native quadratic-product coefficient queries validated by independent polynomial replay<5e-16.
- `NATIVE_HIERARCHICAL_ROOT_PLAN_V1.md`:queued sampled-coefficient refits of fixed native whole-quadratic roots, widths1/8/32/128/256; complete4608sharedbank charged. Structural-capacity comparison, not equal-price CP comparison.

## Signed quadratic width and writer basins — 17:29 UTC

[Timed report](../explanations/for_logan/research_update_2026-09-20_1729_signed_width_and_writer_basins.md) includes the inertia-based minimum-width derivation and source background.

- `quadratic_square.py`, `SIGNED_QUADRATIC_WIDTH_PLAN_V1.md`, `signed_quadratic_width.py`, `SIGNED_QUADRATIC_WIDTH_V1.json`:120 rotated dense signed-square fits; exact minimum quadratic width max(n+,n-), constructive baselines. Minimum-width signed fits can stall95% with negative analytic writers.
- `SIGNED_WRITER_BASIN_PLAN_V1.md`, `signed_writer_basin.py`, `SIGNED_WRITER_BASIN_V1.json`:24 paired controls; fixed/learned positive output or exact Gaussian matching recover both problematic families atlr0.05/same600steps. Gaussian contraction validated by quadrature/gradients1.4e-16. Positivity does not transfer to arbitrary native vector writers.
- `NATIVE_CP_DICTIONARY_V1.json/.pt`:native channel-pair8atoms0.01963% energy versus random2e-10 fraction; optimizedCP0.08630% (~4.4x native dictionary). All dictionary predictions passed, but no useful global approximation.

## Native hierarchy and independent spectral cost — 17:38 UTC

[Timed report](../explanations/for_logan/research_update_2026-09-20_1738_native_hierarchy_and_spectral_cost.md).

- `NATIVE_HIERARCHICAL_ROOT_V1.json/.pt`:8roots98.816% heldout coefficienterror beatsCP butuses10.70Mvalues;256roots96.294%,Gaussian91.622%,train93.256%.128root<95%predictionfailed; finite/8rootcomparisonheld.
- `NATIVE_ROOT_SPECTRAL_PLAN_V1.md`, `native_root_spectral.py`, `NATIVE_ROOT_SPECTRAL_V1.json`:16quadratics fullnumericalrank at1e-6, signwidth579–629. Independentwidth256truncation saves12% but40.7%coefficienterror vs exportedprogram; width5124.23%error costs18.88M vs10.70Mshared. Fullmatrixreplay1.9e-15.
- `NATIVE_ROOT_GAUSSIAN_PLAN_V1.md`:queued fixed-feature outputwriter sweep, syntheticGaussian vs coefficient metrics;12cells, pairedbaseline/radial checks.
- `quadratic_product_gram.py`, `check_quadratic_product_gram.py`, `QUADRATIC_PRODUCT_GRAM_CHECK_V1.json`:exact hierarchical coefficient Gram formula, independently dense-validated values/gradients<3e-16. Fullnative application not yet executed.

## Exact native refit and coefficient-sampling control — 17:46 UTC

[Timed report](../explanations/for_logan/research_update_2026-09-20_1746_exact_refit_and_sampling_control.md).

- `NATIVE_EXACT_ROOT_PLAN_V1.md`:queued exact full-teacher cross/student-self contractions for fixed8native roots, streaming4608teacherroots. Teacher self norm still estimated; runtime/precision/heldout comparisons preregistered.
- `COEFFICIENT_REFIT_SAMPLING_PLAN_V1.md`, `coefficient_refit_sampling.py`, `COEFFICIENT_REFIT_SAMPLING_V1.json`:540 fixed-feature SVD writer fits, three teacher structures/three widths/five sample counts/12seeds, vs exact4096tuple enumeration. Gramcondition2.484 witness has84.54%train/112.38%fullerror. Dense/shared cases converge withsamplecount; diagonal concentration remains difficult. Not a native samplecomplexity claim.

## Native exact/ Gaussian results and learned sharing — 18:02 UTC

[Consolidated report](../explanations/for_logan/research_update_2026-09-20_1802_exact_native_metrics_and_learned_sharing.md).

- `NATIVE_EXACT_ROOT_V1.json/.pt`:scan23.49s,fp32/64 error2.5e-7;8rootheldout98.816%→98.765%, allpredictionsheld. Exact studentself/fullteachercross only; receipt scope wording does not imply exactteacher selfnorm.
- `NATIVE_ROOT_GAUSSIAN_V1.json/.pt`:12writerfits;256rootGaussian75.055% (ridge0) /74.985%(ridge.01), versusoriginal90.635% andradial89.940%; coefficienterror101.027% /100.856%. Allpredictionsheld; syntheticGaussian, noactivationcovariance.
- `shared_quadratic_bank.py`, `check_shared_quadratic_bank.py`, `SHARED_QUADRATIC_BANK_CHECK_V1.json`:exactlowrank sharedbank Gram/nativecross andcoefficientqueries, densecheckedvalues/gradients<7e-16.
- `SHARED_BANK_TOY_PLAN_V1.md`, `shared_bank_toys.py`, `SHARED_BANK_TOYS_V1.json`:40plantedfits,5families; exactcapacitydoesnotguaranteerandomrecovery.
- `shared_bank_fit.py`, `SHARED_BANK_INITIALIZATION_PLAN_V1.md`, `shared_bank_initialization.py`, `SHARED_BANK_INITIALIZATION_V1.json`:18followups; widerbank rescuesonehardcase, leadingunfoldingcanmissthetruequadraticspan.
- `SHARED_BANK_TRACE_INITIALIZATION_PLAN_V1.md`, `shared_bank_trace_initialization.py`, `SHARED_BANK_TRACE_INITIALIZATION_V1.json`:10fits; traceinitialization exactlyrecoverscoordinate/rotatedsigned cases butisworseonotherfamilies.
- `NATIVE_LEARNED_SHARED_BANK_PLAN_V1.md`:queued8exact-gradientnative fits,bank4,k4,48,384reducedvalues; compareCP8 at46,080. Dense rootcore, nosparsity/circuitclaim.

## Learned sharing, stability and input capacity — 18:25 UTC

[Timed report](../explanations/for_logan/research_update_2026-09-20_1825_shared_features_and_input_capacity.md).

- `NATIVE_LEARNED_SHARED_BANK_V1.json/.pt`: eight fits; best training-selected coefficient gain0.2015% vs CP8's0.0863%;48,384 vs46,080 reduced scalars. Root conditioning and independent diagnostics retained.
- `audit_learned_bank_stability.py`, `LEARNED_BANK_STABILITY_V1.json`:28 exact function/feature-space comparisons; high-rate function cosines0.830–0.934, minimum quadratic-space cosines0.00049–0.643. Approximation stability is not feature identification.
- `quartic_input_probes.py`, `check_quartic_input_probes.py`, `QUARTIC_INPUT_PROBE_CHECK_V1.json`: dense adjoint and exhaustive Gram validation.
- `NATIVE_INPUT_MODE_PLAN_V1.md`, `NATIVE_INPUT_MODE_V1.json/.pt`: two4096-probe spectra plus independent projected-energy estimates. Learned32-span retains about0.2334%; finite-probe spectra are not global certificates.
- `input_mode_calibration.py`, `INPUT_MODE_CALIBRATION_PLAN_V1.md`, `INPUT_MODE_CALIBRATION_V1.json`: radial input-span bound99.9591% atd1152/r32 and finite-probe spectral-bias calibration.
- `NATIVE_SHARED_BANK_PRUNE_PLAN_V1.md`: queued45-support exact eight-of-ten root selection, matchingCP8 scalar count plus16supportintegers.
- `projected_quartic_energy.py`, `NATIVE_PROJECTED_ENERGY_PLAN_V1.md`: CPU exhaustive weighted-coordinate check passed; native exact conditional input-capacity audit queued.

## Root sparsity and structural allocation — 18:29 UTC

[Timed report](../explanations/for_logan/research_update_2026-09-20_1829_sparsity_and_structural_assumptions.md).

- `NATIVE_SHARED_BANK_PRUNE_V1.json/.pt`: best gain0.14363% atCP8 scalar count+16support integers; strongest ten-root banks retain only44–46% when pruned. Allregistered predictions pass, but90%-retention prediction applies to a weakfit.
- `audit_root_output_rank.py`, `ROOT_OUTPUT_RANK_V1.json`: exact outputrank8 approximation preserves≥99.9586% of exportedstudent energy; keeps10root products andadds80mixing scalars. Notnative teacherrefit or8-rootprogram.
- `SHARED_BANK_ALLOCATION_PLAN_V1.md`, `shared_bank_allocation.py`, `SHARED_BANK_ALLOCATION_V1.json`:80new+40prior fits,2×3/3×2/6×1, same72reader scalars butdifferentwriter costs. All6×1oracle witnesses<2.5e-8; randomcoordinatebest31.1%, contradictingallfamilyrecoveryprediction.
- `square_leaf_control.py`, `SQUARE_LEAF_CONTROL_V1.json`:16known-capacity controls; squareleavesrescuecoordinate butrandomrotatedsignedbest31.1%. Witnessespass; coordinatepredictionholds, signedrecoverypredictionfails.
- `NATIVE_QUARTIC_COVARIANCE_PLAN_V1.md`: queued16forward MLP16input capture, distinctfromearlierMLP17coordinates; centeredcovariance/mean/secondmoment/rows saved forisotropic-vsweightedfollowup.

## Change feature basis before pruning — 18:36 UTC

[Report addendum](../explanations/for_logan/research_update_2026-09-20_1829_sparsity_and_structural_assumptions.md).

- `ROOT_BASIS_SEARCH_PLAN_V1.md`, `root_basis_search.py`, `ROOT_BASIS_SEARCH_V1.json`:130bases×45supports×8students. HighrateAdam eightrootstudent-errors73/75%→6.30/7.39%;<5%predictionfails. Includes16featuremixing scalars.
- `ROOT_BASIS_REFINE_PLAN_V1.md`, `root_basis_refine.py`, `ROOT_BASIS_REFINE_V1.json/.pt`:8rotationrefinements yield2.52/2.43% errorsvsfullstudents, analytic/directreplay andfixedbasisconditionspass. Allrefinementpredictionshold. Eightrootproducts,46,096scalars+16supportintegers; nativeteacherresultpending.
- `NATIVE_MIXED_ROOT_PLAN_V1.md`: queuedfullnativecross/writerrefit andcoefficient/Gaussiandiagnostics for two selectedmixed-rootstudents. Noidentityclaim frombasisoptimization.

## Covariance metric instrument — 18:40 UTC

- [Metric definitions and derivation](QUARTIC_METRIC_GEOMETRY.md): four-independent-slot covariance loss, repeated-input Gaussian trace terms, empirical eighth moments, and eigenvalue-floor sensitivity.
- `check_quartic_covariance_metric.py`, `QUARTIC_COVARIANCE_METRIC_CHECK_V1.json`: independentdense/gradient/quadrature checks<5e-16; equalcovariance doesnotdeterminequarticfunctionerror, floor.01canweightquarticenergy1e-8.
- `NATIVE_WEIGHTED_BANK_PLAN_V1.md`: queued16original-coordinate bankfits, isotropic/centeredfloors.01,.1/secondmomentfloor.01. Matchedisotropicreruns, originalreaderparameterization, empiricalandGaussian diagnostics; depends onMLP16capture, notMLP17statistics.

## Native exact capacity and covariance capture — 18:42 UTC

[Timed report](../explanations/for_logan/research_update_2026-09-20_1842_input_capacity_and_covariance.md).

- `NATIVE_PROJECTED_ENERGY_V1.json`: exact32-span enumeration, learnedavailable0.2280%/utilization88.37%; CPavailable0.16937%/utilization50.95%. Allpredictionshold; conditionalspanonly, totalteachernormestimated.
- `NATIVE_QUARTIC_COVARIANCE_V1.json/.pt`:16forwards,2048actualMLP16inputrows/panel, means/covariances/secondmoments/hashes. Meanenergy68.15/67.21%, covariancepanelshift93.2%; allpredictionshold.
- `audit_quartic_input_statistics.py`, `QUARTIC_INPUT_STATISTICS_AUDIT_V1.json`: inputcoverage andliveflooraudit. Centeredfloor.01inactive, .1changes305eigenvalues; secondmoment.01changes85. Inputcoverage≠quarticfunctioncoverage.

## Mean-centered native baselines — 18:48 UTC

- `centered_quartic.py`, `NATIVE_CENTERED_DEGREE_PLAN_V1.md`: independentpolarization/replaychecks<1e-15; queueddegree0..4 census onactualpanels andnoncentralGaussian, calibrationmeanonly. Componentsneednotbeorthogonal.
- [Exact joint third-order quadratic piece](CENTERED_QUADRATIC_FOLD.md), `centered_quadratic_factors.py`, `CENTERED_QUADRATIC_FOLD_CHECK_V1.json`: foldedconstant/linear/9216-channelquadratic factors, independentlycheckedagainstautogradHessian.31.85Mfactorprice; usefulness/compressionnotyetdemonstrated.

## Native weighted fit results — 18:52 UTC

[Timed report](../explanations/for_logan/research_update_2026-09-20_1852_weighted_native_results.md).

- `NATIVE_WEIGHTED_BANK_V1.json/.pt`:16fits, isotropicpanel2errors58–63%,centered51–57%,secondmoment25–31%; training-selectedsecondmoment25.28%. Allpredictionshold, normestimateshave6–8%relativeSE. Sameoriginalparameters; no denseMdeploymentcost. Meanbaselinepending.
- `COVARIANCE_DOCUMENT_AUDIT_PLAN_V1.md`, `audit_covariance_documents.py`, `COVARIANCE_DOCUMENT_AUDIT_V1.json`: descriptivehalf-docsplitcovshiftmedian.914; calibrationbottomeigendirections quarticpenaltyratios1e6–1e7onpanel2. Allpredictionshold; noCI/studenterrorclaim.
- `NATIVE_MIXED_ROOT_V1_FAILURE.md`: V1invalidinstrumentCPU/CUDAmaskmismatch, no scientificresult. Device-awarehelperfix; freshV2queuedwithCPU/CUDAreplaytripwire, originalscientificpreregunchanged.

## Mean recovery versus variation — 18:57 UTC

- `audit_weighted_readers.py`, `WEIGHTED_READER_AUDIT_V1.json`: secondmomentMuonreader spanscapture96.68–97.56%meanenergy vsAdam32.98–34.02%; centeredpredictioncosines.966–.998. No teachercorrectness orfeatureidentityclaim.
- `NATIVE_VARIATION_AUDIT_PLAN_V1.md`: queued frozen16student mean/centeredresidual decomposition, calibrationconstant andweightconstant baselines, teacheroutputcache. Input-dependent recovery remainsunproven untilthisreceipt.

## Weighted norm precision — 18:59 UTC

- `check_projected_norm_estimator.py`, `PROJECTED_NORM_ESTIMATOR_CHECK_V1.json`: exhaustive independent-sign identity validates exactprojectednorm+sampledorthogonalresidualnorm, errors2.2e-16. Variancereductiontarget-dependent.
- `NATIVE_WEIGHTED_NORM_PLAN_V1.md`: queuedrank4exactprojection+4096residualprobes foreachweightedmetric; refines6–8%SEteachercoverageestimates withoutchangingfit/checkpointselection.
- [19:01 strategic checkpoint](../HOURLY_STRATEGIC_REVIEW_2026-09-20_1901.md): prioritizependingmean/variationcontrols, sparsebasisnativevalidation andmetricprecision overanotherarchitecturegrid. Nextmath/literature review19:37.

## Two-stage clarification and latest controls — 19:09 UTC

[Candidate discovery versus general DAG search](TWO_STAGE_DISCOVERY_AND_DAG_SEARCH.md): userclarificationadopted. Currentrestrictedsharing/basis/pruningisnotgeneralgraphoptimization. Countdistinctreachablecomputationsonce; preserveconstituentconditions; paperM/generalmetricandscoringtractabilitydistinguished.

- `NATIVE_CENTERED_DEGREE_V1.json`: f≤2panel2error18.73%, f≤1 41.94%, f0 66.58%; allpredictionspass, fullreplay3.2e-7. Nativequadraticprice31.85M, notcompactwin.
- `NATIVE_MIXED_ROOT_V2.json/.pt`: devicefixpasses, eightmixedrootsretain99.93–99.94%nativebankgain; allpredictionspass.46,096floats+16supportintegers.
- `NATIVE_VARIATION_AUDIT_V1.json/.pt`: secondmomenttrainingwinner25.28%total/27.81%centerederror, constantbaseline65.51%; allpredictionspass. Cachedteacheroutputs/perdocresiduals.
- `NATIVE_WEIGHTED_NORM_V1.json`: secondmomentprojectedresidualnormSE12.53vsraw454.42; estimatedcoverage99.52%underweightedmetriconly. Allpredictionspass.
- `CENTERED_COMPACT_PLAN_V1.md`, `quadratic_student_fit.py`, `QUADRATIC_STUDENT_FIT_CHECK_V1.json`: matched48,384pricecenteredlinear/quadraticallocation proposal, helperdensevalues/gradients<5e-16; finalcross-allocationselectionrulemustbefrozenbeforeGPUrun.

Latest graph stage: [Timed report](../explanations/for_logan/research_update_2026-09-20_1923_two_stage_decomposition_and_graphs.md). Five exact graph controls pass; native DAG export and replay pass. Rank2 output sharing fails native-error bars; predeclared rank4 saves14.2% coefficients with25.28%→26.28% error,26 products unchanged. Residual audit separates metric mismatch and adverse residual alignment. Full arbitrary graph search remains unimplemented.

Implementation target: [Arithmetic-program search specification](ARITHMETIC_PROGRAM_SEARCH_SPEC.md), incorporating the user’s concrete two-stage proposal, degree constraints, artificial-probe objective, edit/refit loop, and baseline accounting.

Graph optimization controls: [fixed-topology plan](TRAINABLE_DAG_PLAN_V1.md), [square optimizer diagnostic](DAG_SQUARE_OPTIMIZER_PLAN_V1.md), [first approximate edit/refit loop](DAG_EDIT_REFIT_PLAN_V1.md). Constants and degree limits implemented;4/5 planted families recovered initially, analytic-writer square follow-up8/8 below1%; redundant product removed, independent products retained. Native centered compact runner queued with fixed cross-allocation Gaussian objective.

Native centered26arm run landed:8products/48,384coefficients,totalerror23.12% vsquartic25.28%,butcenteredvariation34.99% vs27.81%. [Report](../explanations/for_logan/research_update_2026-09-20_1944_centered_programs_and_mean_control.md). [Three-hour review](../THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-20_1937.md) leads to [frozen quartic Gaussian mean control](QUARTIC_MEAN_CORRECTION_PLAN_V1.md), currentlyqueued.

Native stage2 refactor: [report](../explanations/for_logan/research_update_2026-09-20_1955_native_graph_refactor.md). Four learnedproducts96.82%quadraticenergy vsbestdeletion85.35%;34,560coefficients and24.28%nativeerror vs48,384/23.12%. Exact8×16×16core,24fits,actualscalarDAGreplay. Deletionpredictionsfailed; continuousrefactorpredictionspassed. No semanticidentity or globaloptimality claim.

Latest: [Mean correction, cancellation and feature stability](../explanations/for_logan/research_update_2026-09-20_2015_mean_correction_and_feature_stability.md). QuarticGaussianbias18.38%error at47,312coefficients;4productrefits unstable with63–1367×componentenergy. Penalty.001cutsmedianratio3.02 butworstfeaturematchstillfails. Outputsharing4features8products99.30%quadraticenergy. Next fullquarticGaussianprojectedquadraticcross implemented/dense-gradientchecked; nativeprofilingpending.

Native Gaussian quadratic fit V1: science bars failed; centered 24.16%, spherical 28.05% on panel2. See NATIVE_GAUSSIAN_QUADRATIC_FIT_V1.json and GAUSSIAN_QUADRATIC_ARCHIVE_AUDIT_V1.json. Next: GAUSSIAN_LINEAR_CONTROL_PLAN_V1.md; exact directional helper implemented and toy checked.

Gaussian linear control: 22.00% panel2 error at 34560 coefficients/four products; all bars passed. See NATIVE_GAUSSIAN_LINEAR_CONTROL_V1.json and GAUSSIAN_LINEAR_PROGRAM_AUDIT_V1.json. Input-radius mismatch motivates NORMALIZED_PROBE_PLAN_V1.md; planted sphere identity check executed.

Normalized-probe diagnostic fails gap explanation: NATIVE_NORMALIZED_PROBE_V1.json. FEATURE_MOMENT_AUDIT_V1.json shows non-Gaussian higher moments. Next bounded input-only control: MIXTURE_MOMENT_DIAGNOSTIC_PLAN_V1.md.

Fresh frozen validation passed all bars: quadratic20.89/20.94%,quartic17.44/17.90% at64/256 contexts on32newdocuments. FROZEN_FRESH_VALIDATION_V1.json and FRESH_TRANSFER_AUDIT_V1.json. Mixture moment diagnostic passes product covariance but fails fourthmoment improvement; no larger mixture sweep.

DIRECTIONAL_RESPONSE_V1.json: learned-direction bars pass, random directions47–52% response error. Exact QUARTIC_BANK_CORE_V1 now supports next joint shared-product refactor, registered in QUARTIC_BANK_REFACTOR_PLAN_V1.md.

QUARTIC_BANK_REFACTOR_V1: shared12 bank products yields38144coefficients22products,fresh64error17.95%; width8 composition fails despite99.77% bank energy. BANK_FUNCTION_METRIC_AUDIT_V1 exposes noncentral mean mismatch; next BANK_NONCENTRAL_REFIT_PLAN_V1.md.

NONCENTRAL_BANK_REFIT_V1 recovers shared8:28912coefficients18products,fresh17.54/17.95%,directional response passes. SHARED_BANK_IDENTITY_AUDIT_V1 still shows centered primitive minimumcos.44. BANK_FUNCTION_CAPACITY_V1 records rank ceilings; next BANK_WIDTH_FRONTIER_PLAN_V1.md.

BANK_WIDTH_FRONTIER_V1: shared6 yields24296coefficients16products,18.15/18.55%; shared4 19680/14,22.05/22.11%. WIDTH_FRONTIER_IDENTITY_V1 and FROZEN_BANK_FEATURE_ATLAS_V1 retain identity limitations. Next ROOT_FUNCTION_METRIC_PLAN_V1.md.

ROOT_FUNCTION_METRIC_V1 and ROOT_ARCHIVE_METRIC_V1 compute exact Gaussian degree8 root covariance; independentoracle1.1e-15. ROOT_PRODUCT_REFACTOR_PLAN_V1 registers4/6/8root widths; root_product_fit.py gradientchecked. Hourlyreview2101 preserves identification and validation-reuse limitations.

ROOT_PRODUCT_REFACTOR_V1 passes root4 bars; FUSED_ROOT_PROGRAM_V1 gives19632coefficients10products,19.06/19.43% diagnostic errors. Fusedresponse passes learned-direction bars, random48–52%. ROOT_IDENTITY_AUDIT_V1 pairedrootfeatureminimum.904; replication registered with model frozen.

ROOT_IDENTITY_REPLICATION_V1: functionbar passes, all3identitybars fail. ROOT_CANONICAL_FEATURE_AUDIT_V1 posthoc output-shared featuremin.98355/outputmin.99959. CANONICAL_ROOT_FEATURES_V1 exports exact interpretation, not new deployedgraph. Cross-bank replication registered separately.

CROSS_BANK_CANONICAL_V1 all-feature bars fail; bounded CROSS_BANK_OPTIMIZATION_CONTROL_V1 recovers worstfourthmode.977 without changing selectedmodel. CANONICAL_VOCABULARY_V1 common-shift barfails; CANONICAL_CONTRAST_VIEW_V1 separates22.75%commonenergy. Next contrast-error diagnostic registered; softcap caveat explicit.

Cross-output shared-product edit (Sept21 03:13): [report](../explanations/for_logan/research_update_2026-09-21_0313_shared_product_merge.md). Zero of2048 screened pairs met half-deletion squared-error criterion. All130247 pair unfolding lower bounds exceed.5 (best.51631), best fit.53743; no graph edits adopted. Rejects this local threshold on frozen private512, not arbitrary DAGs. Fresh private confirmation bootstrap now available; context-only gap remains.

Joint multi-output graph refactor (Sept21 03:20): [report](../explanations/for_logan/research_update_2026-09-21_0320_joint_graph_refactor.md). Four eight-to-six neighborhoods,48Adam/Muon fits, exhaustive subset-plus-write-refit baseline. Strong improvement gate fails;3neighborhood rank bounds rule it out. Group0 ALS yields.858baseline squared error. Apparent near-best component instability(.907cosine) disappears after extra convergence(>.99999999); fixed-target repeatability is not split stability or semantics. No edited graph adopted.

Stable operational groups (Sept21 03:32): [report](../explanations/for_logan/research_update_2026-09-21_0332_stable_groups_native.md). Convergedhalf-metric functions transferwithin1.3%, componentcos.896FAIL. Exploratoryfivegroup partition mergespairwhoseindividualchanges83/72%but sum6.45%. Frozennew16doc native removal: pair6.34% vsindividual103/109%, twoothergroups1.47/1.53%; allgroups10%gateFAIL(group0=11.06%). This is decomposition-intervention agreement, not a ground-truth native unit or semantic identification.

Continuation behavioral candidate (Sept21 03:46): [report](../explanations/for_logan/research_update_2026-09-21_0346_continuation_candidate.md), [candidate dossier](circuits/letter_continuation_group2.md). Frozen group2 new32FW confirmation: continuation removalCE+.12555 vs spaced-.00021; normmatchedwritercontrol+.05249continuation. All registered gatesPASS; same-current-token contrast1.598logodds,bootstrap[.858,2.021]. Staticwriter localizationFAIL; discoverybyteannotation corrected for one mid-codepoint rowstart. Nativeoriginaloperator grounding, upstream extraction, broad OOD/composition/selectivity remain open.

Original-weight continuation grounding (Sept21 03:56): [report](../explanations/for_logan/research_update_2026-09-21_0356_original_weight_grounding.md). Frozen productcos.999953 withoptimalnative rank1 of fixedoutputobserver;31.47%error vsoptimal31.45%,90.1%energy firstmode. Exactnativeprojection hascontinuationCE+.1329/space+.00017. Rank3effecterror5.72%continuation/33.44%space: allcohortgateFAIL. Exactupstreamscalarfoldreplays<4e-15butisotropic90%energyneeds480/485directions. No standalone extraction/adoption.

[Quadratic edit interference diagnostic](QUADRATIC_EDIT_INTERFERENCE_INTERPRETATION_V1.md): joint versus local edit energy and Gaussian-to-text transfer.

[Gaussian radial-normalization diagnostic](GAUSSIAN_RADIAL_EDIT_INTERPRETATION_V1.md): matching input norm does not close the graph-edit transfer gap.
