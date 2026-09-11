# Weight-only structural methods: current receipt index

Updated 11 September 09:40 UTC. Use this index before opening a structural family
or resuming a checkpoint. The25-hypothesis campaign's initial status column is
historical. Current receipts, queues and source hashes override summaries.
Different capacities, penalties and centered/full metrics are not a leaderboard.
“Converged” below means local convergence under the recorded criterion, never
global recovery or semantic circuit identification.

| Family / campaign IDs | Latest authoritative evidence | What remains unresolved |
|---|---|---|
| Free shared products (1,25) | [Penalized128-product fit](PENALIZED_WEIGHT_PRODUCT_V1_CHUNK_00.json) locally converged; [ALS](WEIGHT_PRODUCT_ALS_V1_RESULT.json) and [GN](WEIGHT_PRODUCT_GN_V1_RESULT.json) did not | Changed penalty versus original objective; global optimum and broad capacity coverage unproven |
| Signed squares (2,23) | [Polished256-square fit](WEIGHT_SQUARE_POLISH_V1_RESULT.json) locally converged | Algebraically related to two-reader products; unequal budgets are not independent evidence |
| Shared input reader and compact partner (3,19) | [Joint rank16partner fit](JOINT_SHARED_READER_RANK16_V1_RESULT.json), converged restarts; [upstream interface](SHARED_READER_UPSTREAM_PORTS_V1_AUDIT.json) | One small component with native background, not a whole-model factorization |
| Sparse multiplication graph in a shared frame (4,6) | [RCG](SPARSE_CORE_RCG_V1_RESULT.json), [independent restart](SPARSE_CORE_RCG_RESTART_V1_RESULT.json), [joint-function audit](SPARSE_CORE_TWO_FUNCTION_SPAN_V1_AUDIT.json) | Locally converged functions differ; orthogonal-frame restriction remains |
| Overlapping multi-output blocks (5,10) | [Original fit](WEIGHT_STRUCTURAL_BASELINE_V1_multioutput_block_RESULT.json) → [custom manifoldCG](MULTIOUTPUT_MANIFOLD_V1_RESULT.json) → [libraryCG](ORTHOGONAL_MULTIOUTPUT_PYMANOPT_V1_RESULT.json) → [exact-Hessian trust regions](BLOCK_TRUST_REGION_V1_RESULT.json): all unconverged | [Thin trust-region continuation](BLOCK_TRUST_REGION_THIN_V1_RESULT.json) completed 600.49 fit seconds, still unconverged; [exact writer/core gauge audit](BLOCK_WRITER_CORE_GAUGE_V1_AUDIT.json) found negligible balancing benefit; [exact conditional writer/core updates](CONDITIONAL_BLOCK_SVD_V1_RESULT.json) settled with essentially no gain |
| Independent blocks under nonorthogonal congruence (7) | [Verified full spectrum](NATIVE_CONGRUENCE_MIXED_V1_RESULT.json), [centered spectrum](CENTERED_CONGRUENCE_V1_RESULT.json) converged; structural bars missed | Does not rule out overlapping blocks; do not repeat as an untried method |
| Output-rank optimum and bounds (8) | [Full-U spectrum](FULLU_OUTPUT_FUNCTIONS_V1_AUDIT.json) | Bound is on global output rank under coefficient error, not arbitrary shared arithmetic programs |
| Concentrated output loadings (8,9) | [Varimax](OUTPUT_VARIMAX_V1_RESULT.json), [normalized rotation](OUTPUT_VARIMAX_NORMALIZED_V1_RESULT.json) converged | Rotation within a pre-truncated space cannot recover omitted structure |
| Sparse overlapping token-function dictionary (9,10) | [512-function fit](TOKEN_FUNCTION_DICTIONARY_V1_RESULT.json), locally converged; [fixed-support debias](TOKEN_DICTIONARY_DEBIAS_V1_AUDIT.json) | [Input functions remain dense](TOKEN_DICTIONARY_FUNCTIONS_V1_AUDIT.json); sparse usage does not imply simple multiplication |
| Sparse token usage with product atoms (1,9,10) | [Product dictionary](SPARSE_PRODUCT_DICTIONARY_V1_RESULT.json), jointly unconverged | Exact conditional atom updates are not joint convergence; writer-only repairs do not settle reader optimization |
| Native product dictionary selection (1,9) | [Exact greedy OLS curve](NATIVE_DICTIONARY_OLS_V1_RESULT.json), [audit](NATIVE_DICTIONARY_OLS_V1_AUDIT.json) | Restricted dictionary/greedy support;1024products capture31.58%full coefficients, not a generic decomposition bound |
| Unembedding hierarchy before folding (11) | [Token/hierarchy backward fold](UNEMBEDDING_BACKWARD_VIEWS_V1_RESULT.json) | Fixed16-leaf whole-token means failed; large token-specific remainder required. Higher-order or overlapping hierarchy models remain open |
| Shared input span with all mixed interactions (3,6) | [Native family bounds](SHARED_INPUT_SUBSPACE_NATIVE_V1_RESULT.json) | Counts every term touching selected inputs; low coefficient coverage is not semantic irrelevance |
| Joint QK source features across positions (19,20) | [Source frame fit](POSITION_SHARED_QK_SOURCE_V1_RESULT.json), [FineWeb frozen validation](FROZEN_QK_FINEWEB_V1_RESULT.json) | Numerator coverage differs from normalized routing effect; the tested physical effect was tiny |
| MLP16 producer / attention QK–OV sharing (19,20) | [Frozen fold](MLP16_PRODUCER_OVERLAP_V1_RESULT.json), [exact key envelope](MLP16_PRODUCER_KEY_ENVELOPE_V1_AUDIT.json), [joint fit](COUPLED_PRODUCER_NATIVE_V1_RESULT.json), [convergence repair](COUPLED_PRODUCER_CONVERGENCE_REPAIR_V1_RESULT.json) | Original tradeoff miss preserved; [fixed midpoint](COUPLED_PRODUCER_MIDPOINT_V1_AUDIT.json) was a post-result audit on already inspected positions |
| Simple functions inside the coupled spans (19,20) | [Exact scalar-product spectra](PRODUCER_SCALAR_PRODUCTS_V1_AUDIT.json), [optimized mixtures](SIMPLE_PRODUCT_SHARED_SPAN_V1_RESULT.json), [red-team](SIMPLE_PRODUCT_SHARED_SPAN_V1_REDTEAM.json) | All36mixture starts converge but noheadpasses combined screen; one-product ceilings<=20.57%apply only to these fixed spans |

Existing block geometry is also prior art:
[input/output spectra and overlaps](MULTIOUTPUT_BLOCK_GEOMETRY_V1.json),
[shared leading output patterns](MULTIOUTPUT_BLOCK_OUTPUT_MODES_V1.json), and
[common/centered decomposition](MULTIOUTPUT_MANIFOLD_V1_OUTPUT_SPLIT.json).
Do not describe those properties as new without comparing the latest fitted
function. High similarity between leading output vectors is not identity of
the complete block computations.

Campaign IDs12–18and22involve activation distributions, perturbation losses,
regimes, statistics or behavioral weighting. Some historical runs exist, but
new data-guided discovery is deferred by the user's weights-first instruction.
IDs21(stability),23(symmetry),24(output support) and25(cancellation) are also
cross-cutting checks; they do not each constitute a separate factorization.
FineWeb is the training-corpus validation source; Pile is separately labelled
corpus-shift testing. Never adapt on the supposedly untouched validation set.

The longer block run completed; preserve its limit and gradient failures.
The exact same-function penalty minimum is already effectively attained.
Broader adaptive/heterogeneous structure and independent solver coverage remain
open; no automatic identical chunk is queued. Convergence or capture alone does not establish OOD prediction,
extraction, selective removal or composition/reuse.

New follow-up: [conditional output-capacity spectra](BLOCK_CONDITIONAL_CAPACITY_V1_RESULT.json)
show modest rank8 gains in10/16fixed frames; isolated gains are not additive.
The [23x4 full-quadratic frame protocol](FULL_QUADRATIC_FRAME_V1_PREREGISTRATION.md)
changes input/output allocation at370944coefficients and changes the penalty.
[Native result](FULL_QUADRATIC_FRAME_V1_RESULT.json): both starts locallyconverged,
capture5.9467/5.9385%, gain targetmissed, wholefunctioncosine.9315. [Outputsplit](FULL_QUADRATIC_OUTPUT_SPLIT_V1_AUDIT.json)
gives only3.02/2.95%centered capture. [Input-union ceiling](FULL_QUADRATIC_INPUT_CEILING_V1_AUDIT.json)
is18.76%full forany92-dimensional inside-onlyinputspan. No identical fit queued.
The earlier [CG failure](FULL_QUADRATIC_FRAME_V2_RESTART_AUDIT.json) and [TR repair](FULL_QUADRATIC_FRAME_V3_RESTART_AUDIT.json)
remain recorded; memoizedV3solver used natively.

New family under development: mixed-radix full-rank linear stages around a
bilinear product bank. [Construction control](MIXED_RADIX_BILINEAR_V1_CONTROL.json)
passes execution/gradient/full-mode-rank checks, with276480proposedcoefficients
atnativewidth4608. [Small planted fit](MIXED_RADIX_PLANTED_V1_RESULT.json) recovers
2/3starts; [failed-point curvature audit](MIXED_RADIX_CURVATURE_V1_AUDIT.json)
finds no useful negative-curvature escape. [Native V2 jointfit protocol](STRUCTURED_BILINEAR_NATIVE_V2_PREREGISTRATION.md)
completed: both starts time-limited, capture1.4968/0.9388%, unconverged.
[Saved-state continuation](STRUCTURED_BILINEAR_CONTINUE_V1_PREREGISTRATION.md)
is managed-live from06:47:48, exact first-start replay passed. V1stoppedbeforejointfit onFDtruncation;
[step-sizeaudit](STRUCTURED_GRADIENT_STEP_AUDIT_V1_RESULT.json) verifiesrepair.
No converged native verdict or globalrecoverabilityclaim; fixedwiring/initialization
remain limitations. Inspect currentresult/runner before any resubmission.

[Internal stage balancing](STRUCTURED_STAGE_BALANCE_V1_AUDIT.json) is exact,
reducing squared stage norms5.55/5.66x; [actual gradient audit](STRUCTURED_BALANCED_GRADIENT_V1_AUDIT.json)
finds only1.62xrelative-stationarity and1.02xmax-gradient reductions. Both2xtargets
missed. Not a convergence repair or measured speedup; unchanged continuation
remains live. Balanced copies are separate, no optimizer-history reuse.

New full-rank shared input dictionary (IDs3/4): [MSP protocol](FULL_READER_DICTIONARY_MSP_V1_PREREGISTRATION.md),
[planted recovery](ORTHOGONAL_READER_MSP_V1_CONTROL.json), [finite-weight null](READER_MSP_FINITE_SAMPLE_V1_AUDIT.json),
[2104step dense-control repair](READER_MSP_DENSE_POLISH_V1_AUDIT.json). All1152input
directions, sparse native-reader coordinates, nativeDownretained; distinct from
64-reader subspace and outputvarimax. Two native fits queued behindstructured
continuation; no native dictionary result. Requires held-out weight-vector gain,
not training concentration alone. Orthogonality/native-product assumptions remain.

[Shared-reader executor](sparse_reader_program_v1.py) consumes savedbasis/codes
and retainedDown/bias without originalL/R. [CPU controls](SPARSE_READER_PROGRAM_V1_CONTROL.json)
verify feature removals, donorinterchange, nonzero pair interactions and exact
quadratic-dose RMS/tanh prediction. Syntheticprogramalgebra only; no native
selectivity/OOD or runtime-speed claim. Native queued sources unchanged.

[Native product energy](NATIVE_READER_METRIC_V1_AUDIT.json) isbroad (topdecile14.76%,
effective4184.65); [complete-product Gram](NATIVE_PRODUCT_GRAM_V1_AUDIT.json)
effective4594.11/4608, maxpaircos.4645. [Constructive non-bound](PRODUCT_GRAM_NONBOUND_V1_CONTROL.json)
shows orthogonalcomponents can shareinputs and admit fewernewproducts. Do not
usecurrentdictionaryGramrank as an arithmetic-complexity lower bound.
Structuredcontinuationseed0completed3.5097%,unconverged; seed937live.

Oblique full-rank reader extension: [registered comparison](OBLIQUE_READER_DICTIONARY_V1_PREREGISTRATION.md), [planted controls](TYLER_OBLIQUE_READER_V1_CONTROL.json), [native training geometry](NATIVE_READER_SHAPE_V1_AUDIT.json), [conditional sparse coding](OBLIQUE_SPARSE_READER_V1_CONTROL.json). Native four-arm runner is queued after the orthogonal dictionary job, using cached training-only shapes and exact fixed-support encoding. Tyler advantage over the simpler covariance baseline missed its toy bar; both preprocessors remain in the planned comparison. Existing orthogonal job unchanged.

Current07:54: structured continuation completed3.5097/3.3098%capture, both unconverged. Full-reader MSP is live; oblique comparison remains queued. [07:51math review](THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-11_0751.md) explains the assumptions and existing conditional writer solve. [Output-metric control](CONDITIONAL_WRITER_METRIC_V1_CONTROL.json) passes; native dictionary writer refitting is pending, not a completed improvement.

[Conditional Down followup](READER_CONDITIONAL_WRITER_V1_PREREGISTRATION.md) is queued after oblique: eight frozen dictionaries including identity/PCA controls, same output parameter count, exact PSD Gram solve with rank/conditioning diagnostics. [Spectral-vs-existing-solver CPU control](CONDITIONAL_WRITER_SPECTRAL_V1_CONTROL.json) passes<=2.84e-15. Full weights fit Down; held-out reader-vector scores remain basis-only evidence. No native conditional result yet.

First native MSPstart[receipt](FULL_READER_DICTIONARY_MSP_V1_SEED_0.json): train54.85%/heldout47.31%, belowPCAheldout47.54%; fullcoefficientcapture29.56%, combinedconvergencecriterionmissed. [Generalization audit](NATIVE_READER_MSP_GENERALIZATION_V1_AUDIT.json) shows fourth-moment criterion concentrated byindividualtrainingreaders (medianparticipation1.045); near-perfectalignmenthypothesisC/Dmissed. Secondstartlive.

Overcomplete native-input dictionary is still untested natively. [Library plantedcontrol](OVERCOMPLETE_READER_LIBRARY_V1_CONTROL.json) getsoneof2recovery successes; sparsefitslocallystationary, densecontrolunconverged evenafter[exact recoding](OVERCOMPLETE_READER_RECODE_V1_AUDIT.json). [Exhaustive2-support audit](TWO_SUPPORT_ORACLE_V1_CONTROL.json) shows greedyOMP obscuredgoodfeaturediscovery:98.09%to99.96%withsamefeatures. Supportheuristics anddictionarylearning mustbe auditedseparately. No currentnativeMSP/obliquesourcechange.

[Sample-polar null](SAMPLE_POLAR_READER_V1_AUDIT.json): one noniterative polar factor
of selected1152trainingreaders gives nativequartic.097181vsMSP.100760;
heldout.002952. IsotropicGaussian train.099462vsheldout.002598, matchingexact
3/(1152+2). Allregisteredbarsheld. Beforeinterpretinghigh-dimensionalL4fits,
comparethisbaseline; largetrainingobjective aloneiseasytoproducewithoutsharedfeatures.
Do notextendMSPmerelytoimprovethatobjective. Queuedobliquecontraststilltestsgeometry.

[Reused proximal encoder](OVERCOMPLETE_PROXIMAL_ENCODER_V1_AUDIT.json) uses unchanged
quadratic_token_dictionary_v1.conditional onfrozenovercompletetoydictionaries.
Lasso(alpha.05), top2support, thenexactLS gives99.6613%onrecoveredfeatures
(vsOMP98.0898%/oracle99.9638%). Bothproxsolvesconverged andmatchedLARSobjective
<=8.1e-16; allfourpredictionsheld. Thisis animplementedscalableencoderalternative,
notnativegainornewfeaturediscovery. Theexistingproxcorealreadyallowsrectangular
dictionaries; reuseitforfurtherL1readerworkinsteadofcreatinganotheroptimizer.

Completed[orthogonal MSP result](FULL_READER_DICTIONARY_MSP_V1_RESULT.json): bothconvergence/heldoutgainmisses. [Function stability](READER_DICTIONARY_FUNCTION_STABILITY_V1_AUDIT.json) cosine.34613misses.9, so lowatomalignment.2374isnotmerelyagaugechangeofthesamefunction. Nativeobliqueisnowlive; conditionalwritercomparisonqueued.

[Overcomplete L1 native protocol](OVERCOMPLETE_L1_READER_V1_PREREGISTRATION.md) isqueuedafterconditionalwriters:2304features,alpha.05,twostarts1800fitsecondseach,sampled-dictionarybaselines,existingproximaloptimizer,FP64jointchecksandrecoverablecheckpoints. [Rectangularexecutor](RECTANGULAR_SPARSE_READER_V1_CONTROL.json) and[conditionalencoder](LASSO_READER_ENCODING_V1_CONTROL.json) controlspass. Price9,142,272matrixcoefficients+indicesisbiggerthancompleteMSP. No nativeL1resultyet.

First oblique ordinary-covariance start [converged](OBLIQUE_READER_DICTIONARY_V1_ordinary_covariance_SEED_0.json)
at739.92s; heldoutreader capture43.7888%, folded29.9058%. Its counterpartandTyler
remain pending. [Fixed-basis native encoder audit](NATIVE_OBLIQUE_ENCODER_V1_AUDIT.json)
on256already-heldoutreaders: parent43.367%, Lasso(.05)41.885%, dimension-scaled
Lasso(.005103)45.253%. Bothcodesconverged; all256improveunderscaledpenalty,
but+1.886ppmissesthe2ppbar. Thisis a post-resultsubsetdiagnostic, notfull-Uor
freshbehavioralevidence. QueuednativeL1source/penaltyunchanged; penaltychoice
mustremainexplicit when interpreting its result.

[Initialization-background audit](INITIALIZATION_BACKGROUND_V1_AUDIT.json)
passesitsnarrowchecks: checked-inconstructorzerosDown, tinyinitializedoutputzero;
nativeL/Rnorms7.71xthemaximumallowedinitialnorm,89.55%entriesoutsideinitialrange.
No actualsavedtraininginitializationwasfoundintheinspectedlocalsnapshot;
externalinitializationoverridesnotreconstructed. Largecoordinateupdatesare
notfunctionalstructurebecauseofscalinggauges. Do notlabelrandom-lookingspectra
asremovableinitialnoise. [RMTweightstudy](https://arxiv.org/abs/2203.14661) is
motivation, notnativeproof; [optimalshrinkage](https://arxiv.org/abs/1405.7511)
assumesalow-ranksignalplusappropriateadditivenoise, notestablishedhere. Theolder
09-03shrinkagereviewaddressednoisycausalfingerprints, adifferentobject.

Previously omittedfromthisindex: [known full-U radial/traceless metric audit](FULLU_TRACE_METRIC_V1_AUDIT.json).
The exactnorm/radialterm is0.2947%ofcoefficientenergy but63.04%ofidealuniform-
spherefunctionenergy. This is priorwork, notnewdiscoveryoractualactivationstats.
[New native bias comparison](NATIVE_BIAS_RADIAL_V1_AUDIT.json) rejects cancellation:
cosine_U(radial,bias)=+.9722, biasnormonly.398%ofradialnorm. Exactsplitreplay2.90e-15;
anticorrelation/cancellationbarsfailed. NativeRMSradiusisretainedexactly, not
replacedbyaconstant; no spectraldenoisingorbodyreplacementqueuedfromthisaudit.

Current09:02: [user-requested methods review](explanations/2026-09-11/explanation_2026-09-11_0902.md) separates joint folded-tensor discovery from native-reader sparsity proxies. Ordinary-oblique seed937 also converged, heldout43.7148%, folded29.8454%; Tyler pending. [Frozen radial correction](FROZEN_READER_RADIAL_V1_AUDIT.json) preserves known radial term in three fits: ideal-sphere capture74.07–74.19%, radial+bias-only63.22%; coefficient gains only0.132–0.138percentage points. No new dictionary fit or FineWeb validation. Prior radial identity remains prior art. Current conditionalDown/L1queue unchanged; no new million-token discovery sweep.

[Full-U fixed-support reader update](FOLDED_SUPPORT_READER_V1_AUDIT.json) reuses symmetric_product_als_v1 through a sparse-support linear map, including all native/frozen products and U-weighted cross terms. One Left/Right sweep on the same128diagnosticproducts raises full capture29.90585%to29.93986%, unchangedbasis/support/Down/price. PCGsolves6/7iterations; independentCPerror replay3.60e-18. This is not independent-reader LS or newdictionarydiscovery. [Conditional continuation V1](FOLDED_SUPPORT_CONTINUE_V1_RESULT.json) reaches29.94420%, instrumentheld but jointgradient and additionalgainbarsmissed. Objectiveplateau alone was insufficient. Saved-state V2polish pending; preserve V1misses. No textvalidation orfull4608-reader optimization.

[Saved-state conditional V2 polish](FOLDED_SUPPORT_CONTINUE_V2_RESULT.json) converged after28further sweeps/45.37fitseconds: simultaneous support gradients9.34e-8/8.24e-10, five-sweep relativechange1.43e-15. Capture29.9441995%; additionalgain1.29e-12, allV2barsheld. Thus this fixed128-product local coefficient fit is settled; do not extend it identically. OriginalV1convergence/gainmisses remain. Neither result constrains alternative supports/dictionaries or all4608products; those remain discovery questions.

[Frozen radial FineWeb protocol](FROZEN_RADIAL_FINEWEB_V1_PREREGISTRATION.md) is queued after the existingweight-only L1discovery, sourceSHA18e3a6fc... . Eightfixedprograms,64historicallyopenedFineWebrows/8192predictionpositions,10bodyforwards/80sequences, no fitting. Native/physicalreplacementreplay, full-Uoutputerror, KL and CEadded. Radial-onlybaseline included; ideal-sphere74% is not FineWeb evidence. No fresh/documentholdout/OOD or circuit claim.

Completed09:15 [oblique comparison](OBLIQUE_READER_DICTIONARY_V1_RESULT.json): Aheld,B/C/Dmiss; ordinarybothconverged, Tylerbothtimeouts. Heldoutreader43.64–43.79%, beloworthogonal/PCA. [Exact Down comparison](READER_CONDITIONAL_WRITER_V1_RESULT.json),17.01s: A/B/Dheld,Cmiss; everyfamilyimproves<1pp, butalllearnedfamiliesbeatrefittedPCA29.1721%by>=1pp inbothstarts. Bestordinary30.7649/30.7081%. [Refitted whole-function stability](REFITTED_READER_FUNCTION_STABILITY_V1_AUDIT.json) Aheld,Bmiss: cosine.367805 despiteconvergedparentbases/exactDown; squaredfunctiondifference/native.38863. Similarcaptureisnotstableidentification. NativeovercompleteL1 isnowlive; FineWebradialdiagnosticqueuedafterit. NoidenticalMSP/obliquecontinuation.

09:40 encoder repair: [zero-padding permutation control](LASSO_PADDING_PERMUTATION_V1_CONTROL.json) holdsallbars; identicalfeatures/rawLassocodes but paddedtopk+LS capture73.12–86.02%. [OLScompletion](LASSO_OLS_COMPLETION_V1_CONTROL.json) matchesexistingOLS/fixescontrol. [64native-reader diagnostic](NATIVE_LASSO_OLS_COMPLETION_V1_AUDIT.json)42.28%legacy,45.52%scaledlambda,47.49%OLS atsame128terms. [Fullordinary0recoding](FULL_OBLIQUE_OLS_COMPLETION_V1_RESULT.json)allbarsheld: full-Ucapture29.9058%to33.2108%, samebasis/Down/price; historicaltestreader47.6138%vsparent43.7888%, notfreshmethodvalidation. CPU84.13s, savednewartifact. LiveL1andFineWebsourcesunchanged.

[L1initialization snapshot](L1_READER_ANCHOR_V1_AUDIT.json): unitatommatchingreader attainsLassorowlowerboundlambda-lambda²/2; iteration81owninitialreader60.42%codeenergy,medianparticipation2.33but88nonzerouses,medianinitialatomcos.8906miss. Intermediatecodeauditonly. OriginalAbooleanincorrectlyacceptedmissingobjective replay; [correction](L1_READER_ANCHOR_REPLAY_V1_AUDIT.json) marksoriginalAunverified. IndependentGramobjectiveagrees5.55e-17butdoesnotinventmissinghistoricaldiag. Review0922keepsL1liveandfeature/encoderfailuresseparate.

[Second-start OLScompletion](FULL_OBLIQUE_OLS_COMPLETION_S937_V1_RESULT.json) A/B/C/Dheld,Efailed: capture29.8454%to33.1620%,historicaltest47.5671%; nativeDownretained. Cross-startwholefunctioncos.423463misses.9 despite replicatedcapturegain. No stableunitclaim; notdirectlycomparable toearlierrefittedDownfunctioncos.3678. Bothparentbaseslocallyconverged; supportselectionremainsspecificgreedyencoder.

FirstovercompleteL1start[fit](OVERCOMPLETE_L1_READER_V1_SEED_0_FIT.json)ends1802.09s/403cycles,jointstationarity.000624381(unconverged). [Beforefitting](OVERCOMPLETE_L1_READER_V1_untrained_SEED_0.json)heldout45.9175%/full38.6816%; [learned](OVERCOMPLETE_L1_READER_V1_learned_SEED_0.json)heldout46.8562%/full43.7746%. Ownbaselinefullgain5.093pp,butheldoutgain.939ppmisses2pp. OriginalcampaignB/Ccannotpass; secondstartliveforremainingcomparisons. NativeL1encoderstilloriginalpaddedtop128, notOLSrepair. Price9.142Mmatrixcoeffvscomplete7.815M; no matchedcapacity/globalfit/circuitclaim.

[Batched OLS completion control](BATCHED_LASSO_OLS_COMPLETION_V1_CONTROL.json) passes scalar/batched support and physical-function comparisons on orthogonal and rectangular dictionaries. [Four-arm frozen re-encoding](OVERCOMPLETE_OLS_REENCODE_V1_PREREGISTRATION.md) is queued after the existing FineWeb diagnostic, SHA `7a50b7f23e57545c03882b49636c411c4c1b7ee54ccd815be8e8f22d26e2c41c`. Same initial/learned dictionaries, lambda .05 and 128 terms; no new dictionary learning or text. Full coefficient quality and two-start function stability will be measured separately from unchanged parent convergence misses. GPU scalar/batched equivalence is checked inside the managed job.

[Coupled dictionary/code optimizer control](ACTIVE_ORTHANT_DICTIONARY_V1_CONTROL.json): fixed nonzero positions/signs, nonnegative code magnitudes, unit-normalized dictionary directions, SciPy L-BFGS-B, then global proximal code refresh. Both planted warm starts converge after three rounds: full joint stationarity 6.42e-9 / 6.89e-9, versus initial .02537 / .02433; all four registered bars held. Analytic/autograd gradient error zero, directional finite-difference error 6.92e-11. This changes the optimizer, not the L1 structural assumption. No native-scale convergence, runtime comparison, global recovery or circuit claim. Native CPU/GPU parameter transfers may become a bottleneck; inspect before scaling. Sources: [SciPy L-BFGS-B](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html), [Byrd et al.](https://users.iems.northwestern.edu/~nocedal/PDFfiles/limited.pdf).

10:22 completed comparison: [L1 parents](OVERCOMPLETE_L1_READER_V1_RESULT.json) both unconverged; original A/D held, B/C missed. [Corrected overcomplete encoding](OVERCOMPLETE_OLS_REENCODE_V1_RESULT.json) A/B/C/E held, D/F missed: learned full capture53.6037/53.6073%, initial39.1697/39.3840%; historical reader47.1943/47.1879% vs initial46.2559/46.2633%. Complete-function cosine.698169<.9. Price unchanged within this comparison but larger than complete dictionaries. [Native coupled optimizer](NATIVE_COUPLED_L1_POLISH_V1_PREREGISTRATION.md) managed-live10:20:52, SHA `d8dda4d7c1443de4cf08d0e87020ad6a061cdb8562761c1cb6b5b0dcc41d34a5`; same L1 objective/encoder, two600s fits, full joint checks, no text. No convergence or circuit verdict yet.

[Frozen radial FineWeb check](FROZEN_RADIAL_FINEWEB_V1_RESULT.json) A held, all B/C/D improvement predictions failed. Orthogonal correction reduces MLP squared error but increases KL; ordinary increases both. [Cached cancellation audit](FINEWEB_RADIAL_CANCELLATION_V1_AUDIT.json) replays energies to numerical precision; normalized radial/traceless/cross terms .211+1.915-1.123=1.003, mean metric cosine-.966. Actual states violate ideal-sphere orthogonality; not a failure of the exact weight identity or a reason to switch to data-guided discovery. This is post-result diagnosis, no new forwards. [Current explanation](explanations/2026-09-11/explanation_2026-09-11_1021.md), [hourly10:22](HOURLY_STRATEGIC_REVIEW_2026-09-11_1022.md).

[Full-folded sparse dictionary chain rule](folded_sparse_dictionary_v1.py) reuses existing chunked_bilinear_coefficient_v1, with gradients through all sparse reader values and shared unit-row features. [Dense/autograd controls](FOLDED_SPARSE_DICTIONARY_V1_CONTROL.json) A/B/D held, original C missed its gradient bar despite function error1.16e-12; [direct-residual polish](FOLDED_SPARSE_CANCELLATION_POLISH_V1_AUDIT.json) resolves the same small case to error2.51e-17/gradient5.97e-13, preserving original miss. The analytic fixture x1*x2-(x1+x2)*x1=-x1² demonstrates reader-proxy conflict: independent support LS gives -x1*x2 (relative function error1.5), while full-objective code changes recover the function and increase reader squared error2 to4.898. This is not native evidence.

[Native full-tensor one-step protocol](FOLDED_SPARSE_DICTIONARY_STEP_V1_PREREGISTRATION.md), SHA `230c017e26cf1a9b53e9e8755dad0adbc77d2a99bbb8b0dc8bebc793f169d42a`, is queued behind coupled L1. Uses frozen original repaired0/937, same128supports/2304features/nativeDown, all weights in objective, no historical-weight-holdout claim. Measures full-gradient time/FD, one block-scaled Armijo step and independent full CP/execution replay. No convergence or circuit claim.

Coupled L1 [seed0 fit](NATIVE_COUPLED_L1_POLISH_V1_SEED_0_FIT.json) remains unconverged: jointstationarity.00148456 versus parent.00062438, meanobjective.2879028 to.2875330. [Fullscore](NATIVE_COUPLED_L1_POLISH_V1_SEED_0.json)53.6037 to53.6744%, only.0707pp; historicalreader+.0200pp. All-start convergence/stationarity/gain bars cannot pass unchanged. Secondseed remains live; no identical continuation.

Completed [coupled L1 polish](NATIVE_COUPLED_L1_POLISH_V1_RESULT.json): A held, B/C/D/E/F missed. Both unconverged; final53.6744/53.6425%capture and functioncos.700246. No identical continuation. [Direct full-tensor step](FOLDED_SPARSE_DICTIONARY_STEP_V1_RESULT.json) A/B/C/D all held:55.9644/55.9433%capture after one step,1.132/.982seconds per initialgradient, FDerrors<5.6e-10 and independentCP/execution errors<3e-15. Native reader error also decreases; toy objective conflict is not imported as a native result.

[Projected sparse integration](PROJECTED_SPARSE_DICTIONARY_V1_CONTROL.json) reuses the prior conditional writer solve/envelope theorem and new sparse chain; all checks held for full/singular output metrics at full candidate rank. No new factorizer/theorem. [Sustained variable-projection fit](PROJECTED_SPARSE_DICTIONARY_FIT_V1_PREREGISTRATION.md) is managed-live10:47:24, source `21359b590078914c4e615e0f45e1e5938d48b9ead8916b60b48d4505ce65eba2`, two3600s starts, exact output solve every evaluation, full native coefficient target, fixed supports/pairings, no text. First five accepted steps63.07% is intermediate only. Rank, cancellation energy and full gradient convergence are checked; no final result yet. [Current explanation](explanations/2026-09-11/explanation_2026-09-11_1021.md#update1048).

[10:51math review](THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-11_1051.md) maps dictionary/CP uniqueness assumptions and TT/Hankel/arithmetic alternatives to the actual sparse quadratic program. [Writer-compensated tangent tool/control](WRITER_COMPENSATED_TANGENT_V1_CONTROL.json) reuses exact output solve: scaling/complete quadratic-block rotations are canceled, isolated-square rotation is not; all bars held. First-order compensation is not the nonzero-residual variable-projection Hessian or a global/finite identifiability certificate. [Native coherence audit](READER_COHERENCE_CERTIFICATE_V1_AUDIT.json) mu.728/.693 makes the sufficient exact-code uniqueness bound unavailable at128terms; it does not show nonunique codes/dictionaries. [Reciprocal reader scaling control](PROJECTED_READER_SCALE_GAUGE_V1_CONTROL.json) keeps the function/writer/feature gradient unchanged while inflating the current global code stationarity metric379588x; rowwise norm-product metric invariant. No native convergence rescue: current dictionary gradient is also nonzero, and live source/criteria stay frozen. Reuse these controls rather than opening another generic gauge audit.
