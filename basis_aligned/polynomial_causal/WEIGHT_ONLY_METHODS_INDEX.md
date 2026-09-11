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

[Full-tensor support exchange](folded_support_exchange_v1.py) extends existing OLS/Schur selection to the exact conditional polynomial residual. [Dense exhaustive control](FOLDED_SUPPORT_EXCHANGE_V1_CONTROL.json) allbars held, maxerror5.69e-14. [Frozen native32 probe](NATIVE_SUPPORT_EXCHANGE_V1_AUDIT.json) allbars held: median swapgain2.52e-7 versus same-support6.25e-12,2.77CPU seconds. [Sequential32](SEQUENTIAL_SUPPORT_EXCHANGE_V2_AUDIT.json) allbars held, fullgain1.02494e-5 (0.001025pp), CP replay5.09e-19; preserves [V1execution failure](SEQUENTIAL_SUPPORT_EXCHANGE_V1_FAILURE.json). Frozen parent iteration729 was unconverged; small conditional gains do not imply major/fullgraph or circuit recovery. [Full matched-capacity sweep](FULL_SUPPORT_EXCHANGE_V1_PREREGISTRATION.md) queued after live projected fit, SHA e549f80e9e161c4a0fd9e5692ec94a3335b26fb4390089079a4417d137f7d42a: refit-only/exchange, both final parents, fixed features/actualDown, zero text.

[Frozen feature-removal computation audit](FROZEN_FEATURE_REMOVAL_V1_AUDIT.json): exact16removal/120pair executor checks held<=1.01e-13; compact/simple B/Cmiss. Rank1captures14.8–31.1%, rank16captures40.4–55.7%,90%needs136–167directions. This applies the existing shared-reader spectral metric to current internal features, no new fit. [Executed coordinate counterexample](FEATURE_REMOVAL_COORDINATE_V1_CONTROL.json) holdsallbars: exactlyseparable128squares, samefunction underHadamard features but rank16removal capture12.55%/rank90=115 versusnaturalrank1. Dense coordinatescostmoremeaningfulconnections; no native sameprice repair or absentstructure inference. [11:26explanation](explanations/2026-09-11/explanation_2026-09-11_1126.md), [hourly11:22](HOURLY_STRATEGIC_REVIEW_2026-09-11_1122.md).

[Projected product Gram](PROJECTED_PRODUCT_CANCELLATION_V1_AUDIT.json), frozen1227: A/Cheld,Bmiss. Componentenergy2.102native vs totalfunction.646849; top16share70.57%, top128negativepairs91.07%, minimumsignedcos-.8195 (notnearopposite-.95). [Exact group audit](CANCELLATION_GROUP_V1_AUDIT.json): A/Bheld,C/Dmiss; top16netenergy.009738 vsindividuals1.48345, groupcos.93836<.95 across729/1227; deletinggroup costs.9738pp. Groupoutputrank1capture95.962% does not imply removability/stability. [Constructed compact group](COMPACT_CANCELLATION_GROUP_V1_AUDIT.json) Aheld,B/Cmiss: fourrealproducts preserve92.8136%groupfunction, fullerror+.06963pp,10368floats vs22528oldplus4096oldindices. Oneproduct85.63%at3456floats; otherbackground/U/sharedbasisretained. [Output-direction red-team](CANCELLATION_OUTPUT_DIRECTION_V1_AUDIT.json) A/Bheld,Cmiss: allthree4product output-direction starts locallyconverge to92.8138%, negligible improvement, no global/multi-output verdict. Reuse signed-eigenvalue pairing and exactsmallcore; do not continue identical outputfits. Group-boundary/stability red-team remains pending; top16energyselection may omit cancellation partners. General CP degeneracy literature motivates checks, not a proof of ill-posedness for this constrained model.

[Cancellation boundary](CANCELLATION_BOUNDARY_V1_AUDIT.json) A/Bheld,Cmiss: graph53cos.994278/drift.000113681 vsbase16.001213508; energy53drift.000189476 (graph40%lower, misses50%bar). [Frozen futureboundary](CANCELLATION_BOUNDARY_FUTURE_V1_AUDIT.json) A/B/Dheld,Cmiss at1580: group53cos.999149, drift1.684e-5, dominantoutputcos.9999993. Rank1outputcapture falls frompartial16group95.96%tofull53group69.90%; stronger boundary stability does not preserve apparent output simplicity. [Token readouts](CANCELLATION_TOKEN_READOUTS_V1_AUDIT.json) Aheld,B/Cmiss: leadingmode88.89%common-token loadingenergy; all8top128token shares<4%; one modeextra-rowshare1.051%>1%bar. [Centering/padding redteam](CANCELLATION_TOKEN_CENTERING_V1_AUDIT.json) A/Cheld,Bmiss: centeredtop128shares2.27–3.70%, aggregate8modepadding.3574%; no lexical-sparse-unit or removable-common-shift claim throughRMS/tanh. Leading8modes cover95.68%ofgroupcoefficientenergy. Native tokenlabels are exploratory, not behavioral semantics.

First [projectedfit seed0](PROJECTED_SPARSE_DICTIONARY_FIT_V1_SEED_0.json) completed3603.57fitseconds: numericalinstrumentheld, capture64.685245%, gain10.73137pp over initialoptimaloutput, unconverged(time_limit); dictionary/code stationarity.00460049/.01079388, componentenergy3.06914native, productGramcondition11714.2. Originalall-start convergenceB and70%captureD cannotpass unchanged. Secondseed937nowlive; fullsupport sweep remainsqueued. No identicalcontinuation or stablecomponent claim.

12:22: [Exact output penalty grid](PROJECTED_OUTPUT_PENALTY_V1_AUDIT.json) A held, B/C missed. [Continuous component-budget red-team](OUTPUT_COMPONENT_BUDGET_V1_AUDIT.json) A/B held: fixed final seed0 readers, lambda .001882335489 reduces summed component energy 3.069139→.767285 (4x), capture64.685245→64.597390%, loss .087855pp. Exact convex output-only optimum at that budget, not joint convergence. [Reusable penalized normalized-code kernel](penalized_projected_sparse_v1.py) and [dense controls](PENALIZED_PROJECTED_SPARSE_V1_CONTROL.json) pass gradient/scale checks. [Native one step](PENALIZED_SPARSE_STEP_V1_AUDIT.json) A failed its coarse FD bar, B/C held:16.72CPU seconds per initial gradient, objective gain.0003660, capture64.62447%, component.716693. [Smaller-step FD](PENALIZED_SPARSE_FD_V1_AUDIT.json) all bars held, error2.28e-7,100.20x reduction; original miss retained. No sustained penalized native fit queued. [Current methods explanation](explanations/2026-09-11/explanation_2026-09-11_1222.md), [hourly12:22](HOURLY_STRATEGIC_REVIEW_2026-09-11_1222.md). Existing second projected start live, full support exchange queued; no new text discovery.

[Intermediate cross-start function comparison](INTERMEDIATE_FUNCTION_STABILITY_V1_AUDIT.json) A held, B/C missed: final unconverged0 versus frozen937iteration1097/2323fitseconds, captures64.685245/64.685964%, functioncos.744808, residualcos.532563, squaredfunctiondifference.330145native. CP loss replay2.55e-15,12.35CPU seconds. Not final convergence or unit stability. [Output-only repair bound](OUTPUT_ONLY_STABILITY_BOUND_V1_AUDIT.json) excludes cosine .9 for any output-only changes losing <=.001capture each: conservative cosine upper bound.844799. Derivation uses conditional-projection Pythagoras and normalized-vector triangle inequality; numerical projection residuals are tiny but not interval-certified. Reader/span changes remain unconstrained. This motivates the already queued support exchange and, if warranted after final results, joint penalized reader optimization; output-only cancellation repair cannot establish stable factors at this budget.

[Penalized joint controller](penalized_projected_sparse_fit_v1.py) reuses the existing projected L-BFGS loop. [Planted end-to-end control](PENALIZED_PROJECTED_FIT_V1_CONTROL.json) both starts converge (58/28steps), packedFD<=3.94e-10 and dense objective replay<=4.41e-16. [Native two-start protocol](PENALIZED_PROJECTED_FIT_V1_PREREGISTRATION.md) is queued after full support exchange, SHA305dd727edc31f4552dd26769b16dee5179073c4925b1d766bef58c8ed336d84. Original final unpenalized parent graphs, lambda.001882335489, normalized code/feature rows, two1200softsecond fits, exact conditional regularized outputs. No text, native convergence or stable-function result yet. Native sources/binding frozen after model-free gate. This changes penalty/coordinates, not the structural family. [Cache relocation receipt](RESEARCH_CACHE_RELOCATION_2026-09-11_1233.json) preserves older generated data-cache bytes and its path while freeing295.9MB disk for live outputs.

[Same-output agreement spectrum](OUTPUT_FUNCTION_AGREEMENT_V1_AUDIT.json) Aheld/B/Cmiss: all1152outputdirections covered; minimum relative squared disagreement.0667827>.05, so no direction passes the registered threshold. CP/covariance replay1.55e-15, native output condition945.22. [Separate-readout canonical comparison](CANONICAL_FUNCTION_AGREEMENT_V1_AUDIT.json) Aheld/B/Cmiss: one of1152correlations>=.95, not16. Leadingcos.964261, nativeerrors6.675/6.630%, corresponding nativefunctioncos.999933, ordinaryreadoutcos.999850. This uses classical principal-angle/SVD geometry ([Björck–Golub](https://doi.org/10.1090/S0025-5718-1973-0348991-3)); our cached-Gram implementation has measured covariance replays, not the paper's QR algorithm or a claim of identical numerical guarantees. All sourcefits remain incomplete; one positive function does not repair coverage misses.

[Native scalar arithmetic](SHARED_NATIVE_FUNCTION_PRODUCTS_V1_AUDIT.json) allbars held for that fixed firstreadout: bestone realproduct81.016%,four90.671%,16products92.230%coefficientcapture; functioncontains.108142%ofnative coefficientenergy. Exact eig/CP/executor<=3.66e-15, radialfraction1.20e-5. Nativeproduct loadingparticipation8.60, leading3547/4182/3093; searched consolidateddossiers, no matching entry found. This identifies a compact weight-selected scalar computation candidate, not its semantic function or selective effect.

[Frozen cached FineWeb validation](SHARED_FUNCTION_FINEWEB_V1_AUDIT.json) Aheld/B/Cmiss: oneproductrelativeMSE48.50%,four48.80%,16products9.941%;16centeredcorrelation.97321. The16result is descriptive, not the originalfourproductpass. 64previouslyopenedrows/8192positions,no newforwards/fitting/OODclaim. [Tail diagnosis](SHARED_FUNCTION_TAIL_ENERGY_V1_AUDIT.json) allbars held: products5–16have1.559%coefficientenergy but24.077%natural scalarenergy (15.44xrelativeamplification); centerederrorfalls86.786%, so mean mismatch alone doesnotexplainfourproductfailure. Preserve coefficientstructure positive and native-distribution limitations separately; no data-weighted refit authorized fromthisresult alone.

[Final full-folded sparse fit](PROJECTED_SPARSE_DICTIONARY_FIT_V1_RESULT.json) completed12:47:48: A/C held, B/D/E missed. Both one-hour starts unconverged, finalcapture64.685245/64.686557%, functioncos.7448168. Seed937 final dictionary/code gradients.0023053/.0039727, componentenergy.96996native versusfirst3.06914. [Full support exchange](FULL_SUPPORT_EXCHANGE_V1_RESULT.json) completed12:49:15, allfourregisteredbarsheld: captures64.997825/64.998444%, gains.312579/.311886pp; refit-only gains.000027/.000015pp. Exchange changes9215/9216readers and takes25.50/25.44compute seconds; whole83.57s. Cross-startcos.746837remainslow (descriptive, no registered stableunitpass). Graph selection matters, but one sweep doesnot establish graphconvergence or stableidentification. Jointpenalizedfit305dd727is nowlivefrom12:49:15; firstnativepreflightFD1.83e-7,gradient1.48GPU seconds, allreplaysheld.

[Shared function physical interface/readouts](SHARED_FUNCTION_INTERFACE_V1_AUDIT.json): A/Cheld,Bmiss; top128tokenenergy20.155%, common-token1.02e-5; positivehe/his/him, negativeshe/her. Prior pronoun/gender work is explicitly in the MLP17dossier/oldjoint32, not new. [Alias comparison](SHARED_FUNCTION_PRIOR_ALIAS_V1_AUDIT.json) A/Bheld,Cmiss: oldtwoinputfunctionspan explains82.164/81.168%currentnativeform, oldtwowriter span captures62.203/61.952%currentoutputdirection. Partial recovery ofknownstructure, notequivalence or anewcircuitcount. Earlier failed reflection sufficiency/selective removal remain. [Punctuation check](SHARED_FUNCTION_PUNCTUATION_V1_AUDIT.json) allbarsheld onhistoricalcache: all32eligible rows loweratopeningparenthesis,50positions,medianwithinrowdifference-2.775globalRMS. Post-selected observational association, notmatchedtoken intervention or clean gender semantics. Physicalread/writeinterface is saved for reuse; coordinate-selective edits do not imply behavioralselectivity throughRMS/tanh.

[Conservative matched-price native control](MATCHED_PRICE_NATIVE_V1_AUDIT.json) A/Cheld,Bmiss:2645energyselectednativeproducts68.2069%withretainedDown,68.6260%refitted versuslearned65.00%; native9,141,120matrixfloats,no graphindices, lowerthanlearnedbudget. Randomnative59.3894%; no globalpruningbound. [Frozen128-position FineWeb tail validation](MATCHED_PRICE_FINEWEB_V1_AUDIT.json) reversesthat ranking: native selectionCEadded+.354613/KL.294016/relativeMLPerror.28060; dictionary0+.011893/.021248/.03724; dictionary937+.035846/.022412/.03520. Aheld, nativebehavioradvantageBmiss, allprogram-preservationCmiss because nativefails; bothdictionaries individuallymeet.05meanCE/.05KLbars. MeanabsoluteperpositionCEchange~.133forbothdicts, not selective preservation. No newbodyforwards,factorfits,fresh/OODvalidation. This executed red-team limits any coefficient-only rejection of the dictionary. [13:10explanation](explanations/2026-09-11/explanation_2026-09-11_1310.md).

First [penalizedjointstart0](PENALIZED_PROJECTED_FIT_V1_SEED_0.json) complete1201.75s: numericalheld, convergencefalse(time_limit), capture64.685135% versusparent64.685245%, componentenergy.613092 versus3.069139, objectivegain.00116769. Finaldictionary/codeunitstationarity.001367/.001522, still>1e-5. Originalall-startconvergenceBcannotpass. Secondstartlive; no identicalcontinuation. Penalizationreducescancellation without materiallychangingcoefficientcapture; finalcross-startstabilitypending.

### 11 September13:30 — Penalized fit completion and shared-reader allocation

[PENALIZED_PROJECTED_FIT_V1_RESULT.json](PENALIZED_PROJECTED_FIT_V1_RESULT.json): both20min starts numerical/objective/energy bars pass, convergence/stability fail; capture64.6851/64.6855%, componentenergy.61309/.61356, cosine.744842. No identical continuation. [SHARED_READER_GROUP_ALLOCATION_V1_AUDIT.json](SHARED_READER_GROUP_ALLOCATION_V1_AUDIT.json): cached shared-input bound at9.142Mmatrixfloats excludes rank32/122groups from65%coefficientcapture (ceiling46.20%); rank16/240groups ceiling76.11%. No native multi-group fit or new conditional-reader implementation yet. [Full requested explanation](explanations/for_logan/research_update_2026-09-11_1327.md).

### 11 September13:42 — Shared-reader multi-group objective and optimization red-team

[Exact conditional math and all receipts](SHARED_READER_GROUPS_V1_MATH.md). New shared_reader_conditional_v1.py implements exact sphere reader update plus prior rankedpartner SVD. Near-planted1/2groups recover; random and spectralstarts can fail. Existing reducedRCG from one randomstart recovers bothgroups; spectral fails. Jointtrustregionpolish fails and develops1.24million-timesgroupenergy cancellation. New shared_reader_group_objective_v1.py adds basis-invariant whole-groupenergy penalty and exactchunkedgradient; controls<=3.28e-16. Penalizedtoyfit locallystationary but5.51%residual andknownfeasibleobjective muchbetter. No nativegroupfit queued, no negative structure inference, no identicalcontinuation.

Restart follow-up [SHARED_READER_RESTARTS_V1_AUDIT.json](SHARED_READER_RESTARTS_V1_AUDIT.json):3/8randomstarts recover low-objective plantedgroups (registered>=4miss). Best selectedbyweightobjective has.0001604residual and.999752matchedgroupcos. Numeric/selectedrecoveryheld, ratefailed; ABNORMALsolvertermination retained despite smallgradient. This is planted initialization evidence, notnativefit.

### 11 September13:54 — Output-sharing symmetric LL1 clarification

Shared-inputgroups are notoutput-sharingLL1. Userrequests latterinterpretation and fullerexplanation. [13:51mathreview](THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-11_1351.md) maps ranks/gauges andLL1literature; [conditionaltool](symmetric_ll1_conditional_v1.py), [CPUcontrols](SYMMETRIC_LL1_CONDITIONAL_V1_CONTROL.json) verify signedinputeigen truncation/fixedquadraticoutputsolve, nearplanted2blockrecovery4.63e-13. No nativeLL1fit orverifiedgenericuniqueness. Native shared-inputpilotdraft unbound/notqueued.

### 11 September14:04 — Matched native shared-input/LL1 pilot queued

[MATCHED_SHARED_GROUPS_V1_PREREGISTRATION.md](MATCHED_SHARED_GROUPS_V1_PREREGISTRATION.md),21dependencybinding, managedsource60c9dd7feed04cd47dd0e8bfb41ef80b7ea4102854607711b7faf41c5b069f9a. Fourarms:64shared-inputrank8groups vs64output-sharing symmetricrank16LL1groups,1,253,376/1,254,400floats, spectral/nativeweightedstarts,120softsecondsperfit. Whole-group penalty.01, exactconditionalinitialization andjointamplitudes. NormalizedcontrolsFD<=3.83e-10. Cached64groupceilingssharedinput28.8318%,LL1output23.8398%; no9.14Mcomparison or convergence/circuitclaim. Originalshared-onlydraft superseded/unrun.

An independent [LL1projectedoutputkernel](symmetric_ll1_projected_v1.py) eliminates alloutputvectors by a groupGram solve. V1control serializationfailed onnumpybool; [failure](SYMMETRIC_LL1_PROJECTED_V1_FAILURE.json) preserved, [control-onlyV2repair](symmetric_ll1_projected_v2_control.py) passesdensegradient<=6.54e-16,packedFD3.38e-11, normal1.36e-16, signedgroup-scaleinvariance. This changes only a controlserializer, not the kernel or anyqueuedsource. No nativeprojectedfitqueued; decide afterpilotresults.

### 11 September — LL1 optimization and algebraic initialization

[LL1_PROJECTED_RECOVERY_V1_AUDIT.json](LL1_PROJECTED_RECOVERY_V1_AUDIT.json): joint/projectedeach1/8goodplantedrecovery; projectedcoveragebarfails, bestgroupcos.999920passes. [Pencilmathandlimits](SYMMETRIC_LL1_PENCIL_V1_MATH.md), [control](SYMMETRIC_LL1_PENCIL_V1_CONTROL.json): exactindependent-input-block case recovers4.22e-14Frobenius withoutoptimization; equaloutputcolumns collapsegap andinvalidate rawdecomposition. Fixed1e-5noisetestinitialerror9.77e-4; [eightmixturepairs](SYMMETRIC_LL1_PENCIL_NOISE_V1_AUDIT.json) selectedbyobservederror yields3.09e-5cleanerror,31.67ximprovement, confirmedexecutor3.32e-16. Notgeneralshared-intersection/overcomplete/nativeguarantee.

### 11 September14:19 — Matched group pilot complete / DAG clarification

[MATCHED_SHARED_GROUPS_V1_RESULT.json](MATCHED_SHARED_GROUPS_V1_RESULT.json): LL1 capture11.6604/11.6396%, shared-input8.5285/8.5355%, approximately1.25Mparameters perarm. All120-second endpoints unconverged. LL1registeredbars hold; shared10%capture misses. Completefunctioncosines .90065/.86428 are not groupidentification. [User DAG explanation](explanations/for_logan/research_update_2026-09-11_1327.md#hierarchy-and-dag-discovery) distinguishes dense/sparse Tucker from jointly learning reusable arithmetic. [Exact toy](DENSE_CORE_DAG_V1_CONTROL.json) recovers common linear parent from two dense15monomial quadratics; no native/noisy DAGsearch implemented.

### 11 September — Explicit shared-parent graph proposal and joint refit

[LL1_SHARED_PARENT_GRAPH_V1_MATH.md](LL1_SHARED_PARENT_GRAPH_V1_MATH.md): cross-group principal-space overlap proposes one shared reader, private-core diagonalization gives an executable arrowhead graph. Nativepair move saves1122floats withoneextra variableproduct; projectionbarholdsoneoftwostarts. Exactcorefitgainmisses; jointreader/coreV2reduces mergecaptureloss16.05/26.93%, first25%barfails andhits500iterations. V1QRsign initialfunctionfailure preserved; V2exactobjective/gradient/executorchecksheld. Selectedparents crossstartcos.01088; no stablevariable/nativebehaviorclaim. This implements one actual graphmove/refit, not a general DAGsearch.

### 11 September — Group instability, stable lower-level atom, prior alias

[LL1_GROUP_MATCHING_V1_AUDIT.json](LL1_GROUP_MATCHING_V1_AUDIT.json) failsgroupstability, but [distributedprojection](LL1_DISTRIBUTED_FUNCTION_V1_AUDIT.json) recoversselectedparentcos.996389. [OLS](LL1_SHARED_FUNCTION_SUPPORT_V1_AUDIT.json) needsONEcomponent;99.6705%parentenergyis square, multiple-groupbarfails. [Alias](LL1_SHARED_SQUARE_ALIAS_V1_AUDIT.json) matchesoldconvergedsquare150(.999108input/.994917function). [Nativeinterface](STABLE_SQUARE150_NATIVE_INTERFACE_V1_AUDIT.json) freezesoldreader andextractsexactnativewrite with.999944oldwriteragreement. This isknownatomrecovery acrossfamilies, notnewhierarchy/circuitidentification. See short[mathrecord](LL1_SHARED_PARENT_GRAPH_V1_MATH.md).

### 11 September — Shared-square graph kernel and proposal limitation

[Shared square bank](SHARED_SQUARE_LL1_V1_MATH.md): exact output projection and shared-reader gradients pass dense controls. Canonical eigen-square clustering removes zero/one readers across native starts; both registered saving bars fail. The executed uv/uw counterexample shows shared linear parents can be invisible to eigen-square merging. No native fit of the uninformative topology was queued. Next graph proposals must go beyond identical squares. Frozen square150 validation is recorded in its MLP17 dossier; quote gating and whole-reader sufficiency predictions fail without invalidating the underlying weight atom.

### 11 September — Projected LL1 convergence run and mixed-parent incidence

Managed wrapper V3 runs the [registered projected comparison](PROJECTED_LL1_CONVERGENCE_V1_PREREGISTRATION.md), two1200-second budgets with measured local stopping. Exact output elimination now uses diagonal equilibration; [control](EQUILIBRATED_LL1_PROJECTED_V1_CONTROL.json) passes. V1/V2 enqueue-format failures are preserved, with no native execution. V3 source fe66af8d88555577609e6965fccc3c6bcc77d50ecfc4387823013adf82f0fc5c started15:21:15; results pending.

[Shared subspace parent census](LL1_SUBSPACE_PARENTS_V1_MATH.md) finds15/17 mixed-parent proposals in frozen pilot starts; only1/0 have three consumers, so broad-sharing prediction fails. Some groups use several proposed parents. Their overlapping components require joint accounting before they can form an executable graph. Proposal readers are now durable; no new circuit claim.

### 11 September — Executable multi-parent graph and marginal-space repair

[Joint graph results](LL1_JOINT_PARENT_GRAPH_V1_MATH.md): shared readers and shared/shared products execute once; full cores prevent overlap double-counting. Storage saves1.43/1.52%, but approximation bars miss. A mixed-only control exposes private-space selection failure; using the full quadratic marginal removes56–61% of loss at unchanged size. Joint parent-span incompatibility remains measured. Spectral projected LL1 completed20minutes at11.8891%capture without convergence; second arm live.

### 11 September — Joint-compatible parents and converged all-core solves

[Matched result and derivation](LL1_COMPATIBLE_PARENTS_AND_CORE_SOLVE_V1_MATH.md): joint membership selection removes only4–7%graph loss. All64symmetric cores can be solved together with a matrix-free SPD operator, including cross-group cancellation; four matched solves converge in38–57iterations. Graphs improve but still miss.001capture-gap bar. Fixed interaction coefficients are no longer the optimization uncertainty; shared-reader/private-space movement is next.

### 11 September — Joint reader/core fitting and coordinate stopping confound

[New objective and red-team](SHARED_READER_VARIABLE_PROJECTION_V2_MATH.md): move shared/private readers and output directions while solving allcores. Gradient checks hold; plantednear4/4,independent0/4. Same-function coordinate reset rescuesone while raw optimizerrestart stopsimmediately; raw norms reachedmillions. Four-cycle recovery1/4stillmisses. Nativekernelpreflight is managed/queued, nooptimizationhidden. Earlier projectedLL1finished11.8891/11.8510%capture withoutconvergence; .90849functioncos/7matchedgroups missstability.

**16:22 continuation:** native joint-reader preflight passed at~.28s/evaluation. Bounded/re-encoded controller preserves function but plantedindependent1/4stillmisses; no credible negativecurvature in two tested misses. Four matched original/shared-graph20-minute native arms are live/queued under [this protocol](SHARED_READER_JOINT_FIT_V1_PREREGISTRATION.md). Results pending.

###16:51 — exact fixed-bank relation diagnostics
[Math review and receipts](THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-11_1651.md): exhaustive<=4group coarsening repairs0/10tojointbar; generalized-mixture and independently whitened span alignment eachfind2/64directionswithin10%error, missing16bar. Exactlinearrestrictions solved; no absent-native-structure claim. StablemodesrecapitulateLL1groups8/18, notnewcircuits; whole-group/prior-squarealiasbar misses. Firstboundednativeoriginalcompletedunconverged; remainingmatchedgraph/nativearmsmanaged.

### 11 September 18:44 — Residual-aware graph incidence and bounded private-space search

[RESIDUAL_PARENT_EDGE_V1_MATH.md](RESIDUAL_PARENT_EDGE_V1_MATH.md): all62possible new consumers of frozen spectral parent1 are scored against the full native residual, preserving oldparents/rank16. Best meaningful addition saves1137floats/one reader but costs2.90e-5 objective versus1e-6bar. A one-dimensional eigenvalue-envelope solver closes conditional private-space numerical bounds for all62and barelychanges the result; all-core fits converge. These are real incidence changes with conditional refits, not general topology convergence. No text fitting or circuit adoption. Earlier [two-branch rotation results](SHARED_NODE_CANONICAL_BRANCHES_V1_MATH.md) separately establish that varimax and disjoint token support are different objectives; fixed-subspace overlap can fall only3.93% under orthogonal rotation.

## Joint composed interaction paths, 11September22:10

[User proposal](explanations/for_logan/interaction_path_decomposition_proposal_2026-09-11.md) → [coupled sparse path kernel and controls](COUPLED_SPARSE_PATH_V1_MATH.md) → [matched native pilot](COUPLED_SPARSE_PATH_PILOT_V1_PREREGISTRATION.md). Full-U residual/attention self and mixed blocks, exact eliminated edge writers, jointly optimized source readers. Shared versus independent features at147456fitted floats each. Edge/group sparsity, not learned output-core entry sparsity. Native status in managed runner/results; controls do not establish circuits.
