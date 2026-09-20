# Two stages: discover candidate computations, then simplify their shared graph

User clarification incorporated20September2026,19:09UTC. This is a real extension to the current search, not a rename of sparse Tucker. The final target is an acyclic computation graph with arbitrary reuse; decompositions supply candidate building blocks and initializations.

## Current implementation versus intended method

| Capability | Current evidence | Missing work |
|---|---|---|
| Candidate discovery from contracted weights | CP, sharedquadratic/block forms, hierarchical native-root baselines, isotropic/covariance objectives, plantedcontrols | Broader Tucker/HT proposals as useful, withoutassumingonearchitectureisfinal |
| Reuse | Sharedquadraticbanks feedmultiple roots; exportedprograms reuseleafcomputations | Generalcross-branch/cross-depthsubexpression discovery andskip-levelconnections |
| Change feature basis | Four-featuremixing turnsfixed-coordinateeightrooterror73–75%into2.4–2.5%vsfullstudent; nativegainretention99.93–99.94%verified | Jointchangesacrossseveralintermediate depths |
| Discrete edits | Exhaustiveeight-of-tenrootselection andfixed-supportrefinement | Introduce/removeintermediates, distributivefactoring, exact/approximateintermediate merges, splittingoutput-shared sums |
| Complexity | Explicitparameter/outputframe/supportcounts, selectedoperationcounts | Onegraph-wide count ofdistinctproducts, additions, storedcoefficients andnormalizationoperations |
| Semantics | Functions/feature-space overlap andempiricaldiagnostics | Constituent-condition descriptions andvalidatedprediction/intervention/reuse; monosemanticitynotguaranteed |

The existing sharedbilinearbank is already a restricted DAG. StandardHT instead organizes tensor slots in a tree; tied computations can addreuse, buttreecompressiondoesnotautomaticallysearchgeneral arithmeticcircuits.

## Representation and scoring

Represent scalarlinearfeatures, products, linearsums andoutputs as explicitnodes. Permit outputs toreadanynode andnodes tofeedmultiplelaternodes. Preserve a list of constituentconditions insideeachoutput-shared feature; do notforceonehumanlabel. A sumofreal-valuedconditions is not automaticallyBooleanOR, andtheirproduct is not automaticallylogicalAND.

The algebraicobjective is

$$
\mathcal J=\mathcal E_{\rm weights}
+\lambda_\times N_{\rm distinct\ products}
+\lambda_+N_{\rm additions}
+\lambda_cN_{\rm stored\ coefficients}.
$$

Countonlyreachablecomputations, eachonce, acrossalloutputs. Separate nonlinearproducts fromcoefficientmultiplications; statewhether±1/0constantsarefree andincludevectorwidths. Chargefeaturemixing, references/support andsharedoutputframesconsistently. Consider error–complexityParetofrontiers ratherthanonearbitrarypenaltysetting.

Exactrewrites preservefunction: common-subexpression caching, factoringcommonlinearcombinations, mergingexactduplicates andexposingconstituentproducts. Approximatepruning/mergingandnewfeaturedirections requireexplicitreconstructionbudget andrefitting. Splittingafeaturemustnotdestroyoriginalsharingelsewhere; comparewholegraphcostbefore/after, notlocalcore nonzeros.

For$y=u(ab+ac)+v(db+dc)$, anexactgraphrewrite gives$t=b+c$, $h_1=at$, $h_2=dt$, reducingfourdistinctproducts totwo. Ifotheroutputs stillneed$ab$ or$ac$ separately, theglobalcost maydiffer. A dense low-rankcoreslice mayalready becheap asaproductoflinearsums; countingexpandedcoreentriesoverchargesit.

## How the paper helped, and the important limitation

[The paper](https://arxiv.org/pdf/2605.15183) provides a function-aware weight comparison through tensor contractions. We use squaredmatchinglosses builtfromself/cross innerproducts ratherthanjustrawmatrixmatching. Its$M$isageneralmetric onliftedtensor space; aninputcovariance inducesparticularmetrics butisnotidenticaltothefullobject. Repeated-inputGaussianfunctionmatchinginvolvesfourthmomentsforquadraticfunctions andeighthmomentsforquartics. Ourmainnativerunssofar use symmetriccoefficientFrobenius orfour-slotcovariance-weightedlosses; theyarenotallimplementationsofthepaper'sfullGaussianmetric.

Thisenabledexactstudent-self/teacher-crossoptimization, independentgradientchecks, andfixed-featurewriterrefits withoutmaterializingthequartictensor. Fullteachernormshaveoftenbeenestimated; do notcallallreportednormsexact. Thepaper'sefficientrecursionhasstructural/metricassumptions: arbitraryDAGtopologydoesnotautomaticallyinheritcheapexactscoring. Trackcontractioncost/widthandusecontrolledapproximationswhennecessary.

## Evidence as of19:09

- Isotropicnativequartic learnedbankcaptured0.2015%coefficientenergyvsCP8 0.0863%; globallypoor. Native8mixedrootsretain99.94%ofbankgain at46,096floats+16supportintegers, demonstratingoneusefulrestrictedgraphedit.
- Same48,384parameterarchitecture withsecond-momentmetric reaches25.28%empiricalpanel2error bytrainingselection, versus58.48%forisotropictrainingselection. Constantcalibrationoutputbaseline65.51%; centeredvariationerror27.81%, soimprovementisnotmerelyconstantmatching. NoOOD/circuitidentityclaim.
- Exactmean-centered$f_0+f_1+f_2$ baseline reaches18.73%butquadraticfactorscost31.85M. Matched-pricecompressionofitsjointthird-ordertensoristhenextcandidate-discoveryexperiment, withhelpervalidationalreadycomplete.
- Weightednormprecisionaudit nowestimatessecondmomentcoverage~99.52%, teacherenergySE~0.27%relative; mostweightedteacherenergy liesinfourleadinginputdirections. Thisis not99.52%isotropic/globalrecoveryor99.52%empiricalaccuracy.

## Next bounded graph-stage work

Inparallelwiththealreadyregisteredcandidate-compressionwork, establishanexplicitDAGexportforoneselectedcompactstudent. Validateexactfunctionalreplayandliteralglobalcounts, then exercisecommon-subexpressioncache, sharedlinearcombinationfactoring, andconstituentproductexposureonknownsmallcircuits. Requiretheuser'sfour-to-two-productexample, reuseacrossdepths, a casewhereotherconsumerspreventlocalsavings, andsignedcancellationcontrols. Onlythen applycertifiedrewrites tonativeexports; approximateedits followwithweightloss/refitting andfreshfrozenvalidation.

Thisworkmustnotclaimnativecircuitsmerelybecauseatoyrewritesuccessfully. Theendgoalremainspredictive,extractable,selectivelymanipulable,reusableandunderstandablecomputations. Theirmeaningsmaybeclearlydescribedunions/sumsofconditions; monosemanticityisavalidationhypothesis, notacoreconstraintguarantee.

Concrete implementation specification: [Arithmetic-program search from folded weights](ARITHMETIC_PROGRAM_SEARCH_SPEC.md). This extends the conceptual proposal with target boundaries, constant and degree rules, probe fitting, graph edits, and cost-matched baselines.
