# Baselines for simpler bilinear programs

A candidate must beat an executable baseline for the **same output and input ports**, at a stated error tolerance. Comparing a scalar component to the entire model is invalid. Compression alone does not establish a circuit.

## Required comparison ladder

| Baseline | Purpose | Current evidence |
| --- | --- | --- |
| Original native factors, exact normalization/attention/bias | Fidelity reference and original program price | v628 native replay |
| Frozen extracted dense recent path | Same h16/x0/v1 ports and same scalar target; independent execution | v628: 24,235,715 tensor values; parent has 256 signed squares |
| Zero / calibration-only constant / isotropic quadratic | Detect trivial prediction and normalization shortcuts | v623 rejects constant/isotropic explanations; rerun when target changes |
| Native bilinear-channel pruning, optionally scalar refit | Does simply retaining original units suffice? | v617/v618; global tensor scope only |
| Joint scalar quadratic eigendecomposition | Strong conventional baseline for a fixed output reader | v619–v624; 256-term parent, not yet a simple semantic circuit |
| Joint symmetric Tucker / HOSVD | Shared input/output spaces of the contracted third-order tensor | v615 family-specific bounds; not a universal circuit bound |
| HT on fixed trees, then alternative trees | Recursive quartic baseline with explicit slots and ranks | Native two-MLP homogeneous branch measured below; full normalized path and alternative-tree comparisons remain missing |
| Sparse shared bilinear DAG | Candidate method: reuse, adaptive widths, sparse interactions | Reference quotient evaluator exists; automatic native discovery incomplete |
| Individual-matrix SVD, balanced-gauge SVD, matched-price random controls | Sanity/control families; not substitutes for joint decomposition | v629 raw SVD fails; native balanced-gauge and matched-price random arms not yet run |

Do not silently omit a missing baseline or describe planned runs as measured.

## Literal accounting

For one native MLP with d=1152 and h=4608: three dense matrices contain
3dh=15,925,248 scalar weights, plus 1152 output biases. It computes h=4608
bilinear channel products per token, in addition to linear projection arithmetic.
These are local costs; attention, residual sources and readout remain chargeable.

For a scalar reader c, form
Q = sym(L^T diag(D^T c) R), with scalar bias c^T b.
The exact eigendecomposition gives x^T Q x = sum_j lambda_j (p_j^T x)^2,
at most d signed squares. Dense eigendecomposition costs O(d^3) after forming Q;
forming Q directly costs O(h d^2). A truncated rank-r scalar program stores dr+r
values plus bias, but the reader/writer and upstream generators are charged if
needed for the declared interface. Shared factors are counted once in a DAG.
A rank-r eigentruncation is the coefficient-Frobenius baseline for this scalar
quadratic, not automatically the best behavioral predictor on normalized data.

Report storage (values, bytes, sparse indices), scalar products/squares, linear
arithmetic, distinct shared features, live state, and required external ports.
Use matched total price, not equal rank labels. Keep output dimension, precision,
context length, nonlinear operations and intervention semantics identical.

## Evidence and red-team rules

For every candidate report coefficient/polynomial error, scalar prediction error,
intervention-change error, behavioral effect, preservation controls and reuse
separately. Keep fresh, opened and replay panels distinct. Never average these
into a score that hides a failed requirement. Dense fallback is baseline success,
not compression success. Relative errors also need absolute numerator and target
norm when denominators may be small. Optimized methods need convergence evidence,
restarts and planted recoverability before a failure becomes scientific evidence.

Red-team successes with shuffled/matched-cost controls, held-out selection,
constant predictors, residual/background accounting and extraction without hidden
model access. Red-team failures with full-capacity replay, independent contraction,
bias/norm/axis/sign checks, precision, planted recoverable examples and factor
gauges. A row rescaling L_k -> a_k L_k, R_k -> R_k/a_k preserves the joint tensor,
but raw matrix SVD need not respect it. The executable gauge control shows why a
negative factor-SVD result cannot establish absence of a simple polynomial.

See [gauge control](DECOMPOSITION_GAUGE_CONTROL_RESULT.json),
[HT objective](HIERARCHICAL_TUCKER_SHARED_DAG_DIRECTION_2026-09-20.md),
[recent-path extraction](RECENT_FOLDED_COMPONENT_2026-09-20.md), and
[negative SVD comparison](RECENT_SHARED_FEATURES_2026-09-20.md).

## Native gauge/numerical follow-up

v630/v631 add the joint MLP16 input-mode Gram and balanced-factor controls.
The native CUDA SVD full basis fails orthogonality/replay; CPU float64 repairs
full-rank recovery but leaves rank512 failure unchanged. Joint-mode rank512
also fails. See [audit](JOINT_INPUT_BASELINE_AUDIT_2026-09-20.md). This is a
measured input-mode projection baseline, not yet a native HT or sparse-core run.

## Shared native-product follow-up

v632/v633 compare full-write, contracted-parent and matched random supports,
preserving full input span. The parent isotropic functional metric (including
trace) improves strongly over coefficient Frobenius selection but no compressed
arm reaches10% error. See [metric audit](SHARED_PRODUCT_METRIC_AUDIT_2026-09-20.md).
These are fixed-dictionary baselines, not a bound on learned quadratic features.

## Radial-feature extension

v634 adds one shared radial quadratic and centered native atoms; all compressed
arms fail. v635 fits calibration-only product means with the same supports and
improves error, but the best tested compressed validation error remains26%.
[Audit](RADIAL_FEATURE_AUDIT_2026-09-20.md) distinguishes isotropic weight algebra
from native-distribution statistics. Both remain negative candidates.

## Head-conditioned joint sparse-core baseline

v639/v640 natively verify the exact128-coordinate conditional normalized map
and compare joint tensor-energy frames with sparse versus dense cores. Both
output ranks32/128 fail native cross prediction even with dense cores.
[Measured frontier](HEAD_SHARED_CORE_BASELINE_2026-09-20.md) includes pair sharing,
branch prices and uncharged background dependencies; no complete-circuit claim.

## Data-informed output-objective control

v641 compares the same dense-core output ranks with a calibration-interaction
eigenspace and random frame. The empirical frame has2%calibration error at128
but18–68%cross error on other families. [Audit](HEAD_OUTPUT_OBJECTIVE_AUDIT_2026-09-20.md).
This fitted-basis failure is not a proof against all native-distribution metrics.

## Task-specific effect baseline

The subject-number nine-edge graph costs92 native block evaluations per sequence;
the exact same five-port joint effect costs36 using two suffix corners. An
aggregate-only algebraic query plan reduces92 to85 but remains more expensive.
Individual edge attribution is a different interface and needs its own comparison.
[Cost and reuse audit](SUBJECT_NUMBER_COST_AND_REUSE_AUDIT_2026-09-20.md) verifies
the symbolic cancellation and distinguishes same-task transfer from cross-task reuse.

v643/v644 now support a conditional six-MLP response baseline: freezing six later
attention writes gives1.7–3.9% effect error on opened prompts and3.0–9.8% on new
matched-position prompts. [Conditional chain](SUBJECT_NUMBER_CONDITIONAL_CHAIN_2026-09-20.md)
charges eight vector ports,95,560,716 fixed values and27,648 products. It is an
explicit dense reference for future folding, not a simple discovered circuit.

v648 validates the513,030-value width8 joint response compiler after correcting
baseline-port precision, but state-SVD8 fails the effect gates. v649 identifies
large losses even at initial-only and final-only projections. v650 reserves two
answer readers at the same width/cost and improves strongly, but still fails.
[Observable-basis audit](SUBJECT_OBSERVABLE_BASIS_AUDIT_2026-09-20.md) preserves
these negatives and separates oracle diagnostics from executable predictors.

v651/v652 use derivative/finite-reader bases; v653 uses separate snapshot-balanced
encoders and decoders. All retain width8 and fail the all-cell effect gates. The
dual-space program costs522,246 values, including its extra initial encoder.
[Finite-reader audit](FINITE_READERS_AND_DUAL_SPACES_2026-09-20.md) distinguishes
exact contraction, planted recovery and empirical native prediction.

## Conditional positive and its limits

v654 broadens calibration coverage at unchanged width8 and passes the conditional
5% effect gate in every opened evaluation cell (1.2–2.9%). Full-native fidelity
still fails two cells (10.2–11.3%). v655 exports and independently replays the
shared-product runtime on CPU; native context generation remains charged.
[Portable program](SHARED_SUBJECT_RESPONSE_PROGRAM_2026-09-20.md).

v657 restores attention12 while keeping all suffix MLPs dense. Native restoration
recovers only17.9–24.3% of the postposed frozen-model error, missing the50% gate.
The projected restoration passes10% full-native effect error on these opened rows.
This is not yet composition with the reduced MLP chain.
[Positive and negative audit](ATTENTION_RESTORATION_AUDIT_2026-09-20.md).

v658/v660 compose the attention12 fold and reduced MLP chain. Both pass the5% conditional gate; full-native10% still fails one cell on each panel. v660 adds genuinely new longer structures under frozen bases. v659 corrects the producer export by including55,296 output-encoder values. [Composition, transfer and accounting](COMPOSED_SUBJECT_RESPONSE_TRANSFER_2026-09-20.md).

## Same-width preservation and sparse interaction controls

v665–v667 use four calibration reader contrasts at width8 and pass prospective target/control tests. v668 compares18/36 and9/36 shared-pair supports, three equal-price random18 supports and a zero-quadratic-numerator baseline. Half-support passes; quarter/zero fail; one random also passes. Dense normalization and native context costs remain. [Definition, geometry, prices and verdicts](JOINT_READER_SPARSE_RESPONSE_2026-09-20.md).

## Conditional source-amplitude baseline (later20 September)

[Native source Hessians](SEMANTIC_SOURCE_QUADRATIC_JET_2026-09-20.md) provide a measured same-output/source-interface comparison: linear3, diagonal6, signed-rank1 7, full symmetric9 values/context. Rank1 andfull pass unit/negative/mixed10% on opened contexts; doubled edits fail. Producer computation and context dependence are fully charged; this is not a global tensor-decomposition or HT comparison. CPU extraction is exact only for the quadratic observable.

Later [quadratic design audit](SHARED_SOURCE_QUADRATIC_DICTIONARY_2026-09-20.md): original four arms had symmetric designrank2/6. On six independent source settings, fullquadratic4.72% passes, rank1 15.74% fails. Always report monomial-design rank; signed/strengthened versions of the same direction do not identify additional Hessian dimensions. Sharedoutput2/input2 fail17.33/18.29% on this stronger benchmark.

## Full-path curvature controls

[Exact local Hessian sum](NATIVE_SOURCE_CURVATURE_DECOMPOSITION_2026-09-20.md) supplies a native coefficient reference for curvature-path omission. Full quadratic passes fifteen positive single/pair directions; linear, all-MLP-only, attention-only and zero-A/B-cross controls fail their stated gates. Joint omission of individually passing terms fails modal fidelity. These are conditional derivative-program baselines, not native HT or whole-model replacements.


20 September: [Exact source attention fold and spectral baseline](ATTENTION_FOLD_AND_SPECTRAL_BASELINE_2026-09-20.md). Native attention fold replays exactly but fails the one-shot speed gate (1.024x). Per-context signed rank-two Hessians pass opened finite-edit tests with68 versus80 stored values; native generators remain required. Rank-one fails. No native HT or complete-circuit claim.

20 September: [Shared five-source features](SHARED_FIVE_SOURCE_FEATURES_2026-09-20.md) fail the reuse screen: common rank2 plane17.46% heldout number error; per-output planes do not rescue it; three calibrated sparse residual pairs improve to11.86% but still fail. Full-basis recovery is exact. Opened data only.

20 September: [Dictionary optimization redteam](SHARED_DICTIONARY_OPTIMIZATION_REDTEAM_2026-09-20.md). Joint analytic coefficient fits still fail, but native-outcome minimax oracle fits number effects in all32cells (worst7.49%, gap<3.3e-8). Therefore fixed-dictionary capacity is not ruled out. Oracle uses labels and does not establish prediction or modal preservation.

20 September: [Derivative-only minimax](DERIVATIVE_ONLY_MINIMAX_2026-09-20.md) passes opened construction-heldout number9.68%/modal0.89%, but calibrationnumber12.19% fails the overall gate. Fullquadratic8.94% error combines with6.04% compression error. Native generators and group-dependent fitting remain limitations.

20 September: [Per-input minimax extraction check](SINGLETON_MINIMAX_2026-09-20.md) fails heldout number37.22%; retaining group budgets reduces it to10.104% but still fails. True singleton coefficients pass batch permutation/peer-removal checks exactly. Grouped partial success is not standalone extraction.

20 September: [Native two-MLP quartic/HT baseline](NATIVE_TWO_MLP_QUARTIC_HT_2026-09-20.md). Exact homogeneous numerator fold6.59e-15; rank8 symmetric tree9.35% error but656values loses to exact canonical280. Rank2 costs116 but fails36%coefficient error. This is a named direct polynomial branch, not the normalized full model or causal extraction.

20 September: [Matched-strength modal-null control](MATCHED_STRENGTH_NULL_2026-09-20.md). All-cell strength matching fails22/32pass; all22 matched cells show>2x collateral reduction, but original target-strength retention failure remains. Conditional reduced-strength selectivity evidence only.

20 September: [Shared selective directions](SHARED_SELECTIVE_DIRECTIONS_2026-09-20.md) have zero feasible uniform linear retention across calibration; small conflict witnesses and planted controls pass. Singular-subject fixedall5 exception has native1.55–1.68x target strength but fails modal10.422%; baseline andcandidate bothpass7/8primarycells. No general selective circuit.

20 September: [Prospective source-response transfer](SOURCE_OOD_TRANSFER_2026-09-20.md) passes full19-arm prediction on48newtexts: number6.409%/modal2.129%, all16nativecapabilitycells100%. Null strengthretention stillfails15/16cells. Posthoc fixed-rule spectral2passes7.527/3.568%; tangentfails61.47/13.41%. Fresh native derivative generators remain required.

20 September: [Budgeted modal selectivity](BUDGETED_MODAL_SELECTIVITY_2026-09-20.md) improves joint native passes from2/48exactnull to35/48 at5%gradientbudget; all13remaining failures arestrength, no nativecollateral failures. This redteams unnecessary exactnull constraints but stillfails all-cell adoption andrequires native gradients.

20 September: [Curvature-aware matched-budget test](QUADRATIC_BUDGETED_SELECTIVITY_2026-09-20.md) gives42/48joint nativepasses versus41/48linear atsame8%budget. No collateral failures, sixstrengthfailures. Native derivative/source generators remain; source capture omitsfourlate residual contributions, motivatingclosure test instead ofanotherbudget sweep.

20 September: six-source residual-complement experiment remains instrument-invalid. Float32 comparator bug is independently reproduced and float64 derivative replay passes, but absolute cast-closure still fails. See `basis_aligned/polynomial_causal/SIX_SOURCE_INSTRUMENT_AUDIT_2026-09-20.md`; exploratory45/48 versus42/48 cell counts are not promoted.

20 September: corrected six-source interface passes native instrumentation/prediction, improves selective cells45/48 versus42/48 at increased coefficient cost108 versus80. Three strength failures remain. Quadratic-selector LP upper bound rules out80%retention for one cell under its8%per-input constraints, not under native10%aggregate constraints. See `basis_aligned/polynomial_causal/SIX_SOURCE_COMPLEMENT_2026-09-20.md`; matched-width source-swap screen is next.

20 September: fixed five-source substitution improves nativejoint44/48 versus42/48original at80coefficients/context; full6passes45/48at108. Instrument/predictionpass; fourstrengthfailures. Independent5Dselection removes6Dwarmstartdependency andpredicts44/48, nativevalidation separatelyqueued. See `basis_aligned/polynomial_causal/MATCHED_WIDTH_SOURCE_SWAP_2026-09-20.md`.

Independent five-source selector native validation now passes instrumentation/prediction and retains44/48joint cells. Warm-start dependence removed; prospective SOURCE_OOD_V2_BINDING.json freezes48newtexts and support before model evaluation. Full goal remains unmet.

20 September: prospective OODv2 all3sourcevariants pass16/16native selectivity and100%capability, butswap5quadratic16.751%/full6 17.987%fail prediction; original5 7.773%passes. Openedradius/cubicdiagnostic passeshalfquadratic3.33%andfullcubic7.21%, butcubicworsens7/48cells andaddsderivativecost. Freshfailurepreserved. See `basis_aligned/polynomial_causal/SOURCE_OOD_V2_AND_RADIUS_2026-09-20.md`.

20 September: explicit final-readout fieldprogram nativeinstrumentpass, swap5error9.569%passes butfull6 10.257%failsallmethodgate;27coeff/ray exceedsoutputcubic16. CPUchainrulepartition+upstream-fieldcubicremainder passes6.319%at31coeff/ray andbothgenerators. Next quadraticstate->quarticnorm fold avoids conflatingnormgeometry withupstreamstate thirdorder. See `basis_aligned/polynomial_causal/FINAL_READOUT_FOLD_2026-09-20.md`.

20 September: quadraticnative-state/quarticnorm fold instrumentpass, full6stillfails10.103%, swap5 9.445%. ExactGramquotient reducesall288normprograms from3to2quadraticfeatures/30to29totalcoefficients with7.11e-15outputreplay; noaccuracychange, unique-featureorwholemodelclaim. See `basis_aligned/polynomial_causal/NATIVE_QUARTIC_NORM_FOLD_2026-09-20.md`. Nextprioritycross-componentcomposition, notmoreindividualrayfitting.

20 September: native two-site stateprogram joint64/64passes butattractorincrements51/64/subject62/64 failcomposition. Ignored-attractor oraclejoint63/64versusincrements0/64demonstratesmetricconfound. All13weak-axisfailuresremaininteraction-only afterexactmarginals. Exactattention11mixed-edgecompilerplantedreplaypasses, nativecausaltestpending. See `basis_aligned/polynomial_causal/TWO_SITE_COMPOSITION_2026-09-20.md`.

20 September: exact native mixed attention11 edge passes1.254e-15, but removal halves0/13failed-cell interactions. Fixed baseline reader also misses finite transport (beside_subject median57.82%). No circuit sufficiency claim; source/reader costs retained. See `basis_aligned/polynomial_causal/ATTENTION_MIXED_EDGE_NATIVE_2026-09-20.md`.

20 September: transport context test jointreader63/64, midpoint64/64; integratedeffect5.264e-14absolute replay. Allrequire native states/derivatives. Exactresidualquartet CPU split separates generated/carried interactions; nativecensuspending. See `basis_aligned/polynomial_causal/ATTENTION_TRANSPORT_CONTEXT_2026-09-20.md`.

20 September: finiteinteractioncensus instrumentpasses; no singlemodulehalvesall13, best2/13. Summedmoduleeffects7/64predictionpasses excludesrootgeneration. Surrogate rootsplitmedian26.45%; native rootclosurepending. See `basis_aligned/polynomial_causal/FINITE_INTERACTION_CENSUS_2026-09-20.md`.

20 September: nativefinalreadout closesfiniteaccountinggap:root+14moduleeffects64/64numberpasses,worst0.4755%. Nativecounterfactualgeneratorsremain; no extractedcircuitclaim. See `basis_aligned/polynomial_causal/NATIVE_ROOT_INTERACTION_CLOSURE_2026-09-20.md`. Expandedoutputcontrols frozen,nextCIRCUITscreen.

20 September: expandedcontrols rejectbroadselectivity:new6/16vsold16/16,worst44.07%; nativeprecisionandsharedsetup pass. Halfradiusnotrescue. Plainbaselineweaker:0/16boundedstrengthmatches; nextcomparelowercommonstrength. See `basis_aligned/polynomial_causal/EXPANDED_SOURCE_CONTROLS_2026-09-20.md`.

20 September: equalstrengthcomparison demotesswap5broadselectivity. Normmatch16/16butpattern10/16; subjectsbothmethods4/8passes,candidatenoadvantage0/8. Attractorsonly2matched,bothfail. CPUresponse-rotationandrowfeasibilityauditsretainunmatchedcases. See `basis_aligned/polynomial_causal/EXPANDED_CONTROLS_EQUAL_STRENGTH_2026-09-20.md`.

20 September: reusable12coefficientsource selectorfails0/16native versusoracle11/16andunitB4/16. First-ordercertifiedfitceilings47.50%subject0.1918%attractor; template24coeffCPUauditfailsheldconstraints. Boundsrestrictedsixportinterface,notgeneralcircuits. See `basis_aligned/polynomial_causal/SHARED_SELECTIVE_SOURCES_2026-09-20.md`.

20 September: exact23source refinement nativeoracle16/16vs6source11/16; shared0/16,notreuse/extraction. Fit-only12portdictionaryheldfeasibility23/24subject22/24attractor. Nativegeneratorsandoracleamplitudescharged. See `basis_aligned/polynomial_causal/REFINED_SELECTIVE_SOURCES_2026-09-20.md`.

20 September: frozenrefinedcoefficients native4/8samelexical3/8nextnountransfer,oracle8/8;minimum-normfit-onlychoice leavesderivativepasscountsunchanged. Pairwise41/48compatible preventsuniversalno-sharinginference. See `basis_aligned/polynomial_causal/REFINED_COEFFICIENT_TRANSFER_2026-09-20.md`.

20 September: edited-numbermatchedtransfer4/8andnumberconditionedrule1/8; gate48/48/replaypass. Analytic19fieldreadoutdifferentialCPUcontrol5.33e-15, nativecontextattributionnext; diagnosticnotcompression. See `basis_aligned/polynomial_causal/SOURCE_NUMBER_TRANSFER_2026-09-20.md`.

20 September: recipientreadoutroot improvesnative4/8to5/8,notall-cellreuse. Exactcontextpartitionmedianroot16.23%,numerator84.46%,radial-1.79%; noncausaldiagnostic. Nextfactorresidualreadersfromnative sourcedirections. See `basis_aligned/polynomial_causal/READOUT_FIELD_TRANSFER_2026-09-20.md`.

20 September: frozenresidualrolebank4/8causalpasses,finitepredictionfails; donorreader5/8oracle8/8. Readercontext dominatesexactgradientpartition. Fit-onlybankexportandfreshv3inputbindingready,nooutcomes. See `basis_aligned/polynomial_causal/RESIDUAL_READER_TRANSFER_2026-09-20.md`.

### 2026-09-20 source–reader review
Fresh v3 frozen-bank selectivity10/16 and prediction0/16; capability3/6 in opposite along_with singular. Balanced/POD snapshot baseline executed on opened held gradients, not native interventions. Canonical review: [review](/workspace/tensor_language/basis_aligned/polynomial_causal/THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-20_1336.md); CPU receipt BALANCED_SOURCE_READERS_V1.json; native receipt source_ood_v3_role_bank_v1_result.json. No full circuit promotion.

### Attention11 contextual reader and conditional query reuse
[Canonical dossier](/workspace/tensor_language/basis_aligned/polynomial_causal/ATTENTION_READER_FOLD_2026-09-20.md): exact five-factor adjoint; Q2-only derivative omission passes. Native separate Q2/Q1 baseline-query freezes both pass16/16 finite fidelity/selectivity cells; worst number errors1.03%/1.98%. Opened oracle-selected edits; baseline context and source/selector costs retained. Exact signed context partition executed; no independent circuit or joint-freeze claim.

### Joint query reuse and source-column folding
[Canonical dossier](/workspace/tensor_language/basis_aligned/polynomial_causal/ATTENTION_QUERY_COMPOSITION_2026-09-20.md): joint Q1/Q2 reuse passes16/16 opened finite cells, worst number2.54%. Mixed term0.105%full,19.73%smaller singleton. Explicit conditional source-column executor CPU replay4.63e-16 with live second-key negative control; cache accounting corrected in V2. Native installed extraction and fresh OOD remain pending.

### Native source-column installation
[Canonical dossier](/workspace/tensor_language/basis_aligned/polynomial_causal/FIXED_QUERY_NATIVE_INSTALL_2026-09-20.md): installed executor passes all16 opened cells; worst joint-query-reference error0.0142%. Native baseline context and oracle selector remain charged. Fresh48textv4 panel frozen without model outcomes; no fresh transfer claim yet.
