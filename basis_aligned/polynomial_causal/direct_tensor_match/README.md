# Direct weight-tensor decomposition study

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
