# Candidate: letter-continuation group 2

Status: **behavioral candidate; not adopted or fully identified**. Updated2026-09-21.

## Exact executable boundary

Use `half0`, component2 of `MIDPOINT_STABLE_GROUP_REMOVAL_PROGRAMS_V1.pt` (SHA256 recorded in the frozen native plans). Let $h$ be the original last-MLP input, $m_0$ the preceding MLP polynomial output with last-block residual scaling, and $s(h)$ the original recipient RMS denominator. Define

$$
n=(h-m_0/2)/s(h),\qquad m=m_0/s(h).
$$

The extracted contribution is

$$
u=a^\top n-\alpha,\qquad v=b^\top m-\beta,\qquad
\widehat y=wuv.
$$

Here $a,b,w\in\mathbb R^{1152}$ and $\alpha,\beta$ are frozen calibration centering constants. Output $\widehat y$ uses the exact QR-reduced vocabulary frame. If $U=QR_U$, the residual write is $R_U^{-1}\widehat y$. Removal subtracts this write from the original final residual, followed by native final RMSNorm, unembedding and softcap.

Local specification: one variable product, two dense readers and one dense writer (3,456 weights plus two centering constants), with the readout-frame mapping shared. This price excludes upstream computation, normalization and the native background; it is not a whole-model extraction or acceleration claim.

The term arose from a six-product local approximation to eight terms of an earlier512-product folded-path approximation. Therefore, it is not automatically a unique exact summand of the original native operator. Direct original-weight testing now shows that it is nearly the optimal rank-one component of the fixed writer-derived scalar observer: coefficient cosine0.999953, frozen/native-optimal errors31.469%/31.455%. The remaining modes are needed to reconstruct the complete observer.

## Evidence ledger

| Requirement | Evidence | Remaining gap |
|---|---|---|
| Computational specification | Frozen scalar readers/product/output writer above; legal residual intervention | Upstream production still supplied by the model |
| Weight-derived discovery | Folded source-dependent path; output-sharing factors, joint local refactor | Original scalar observer checked; full observer and low-energy cohort fidelity remain incomplete |
| Stable identification | Two converged covariance-half fits have 1.47% removal-effect disagreement on16 unused documents | Teacher and initial selection used all calibration rows; broader split/gauge/OOD robustness absent |
| Held-out prediction | Frozen continuation hypothesis passes32 new FineWeb rows | Same corpus family; no broad OOD result |
| Selective manipulation | Removal harms alphabetic-continuation examples while spaced-word average CE stays near zero; norm-matched writer control weaker | Conditions use observed next tokens; broader collateral tasks and controlled minimal pairs absent |
| Extraction/sufficiency | Standalone executable contribution given native inputs | No independent recovery of upstream inputs; no unique native-unit sufficiency claim |
| Composition/reuse | A scalar feature and write can be evaluated independently | Joint deployment with other extracted circuits untested |

## Operational behavioral hypothesis

Among prefixes ending in an alphabetic character with no pending UTF-8 bytes, the group supports a following token that begins with an ASCII letter without leading whitespace. Contrast: a following token that begins with whitespace followed by an ASCII letter. These labels are evaluated using the realized next token; they are not available as an input to the candidate.

On32 new FineWeb rows, removal CE added is0.12555 nats/token on514 continuation sites and−0.00021 on4787 spaced-word sites. A control uses the same scalar activation with another component's writer, matched in vocabulary-centered norm; it causes0.05249 continuation CE damage. The real direction decreases bare-versus-spaced-letter logodds by2.31797, versus1.16971 for the control.

A same-current-token stratified comparison covers34 token types,51 continuation sites and257 spaced sites. The min-count-weighted effect contrast is1.59844 logodds, with paired-document bootstrap interval[0.85836,2.02112]. This is evidence of context dependence beyond current token identity, not control of all context or difficulty confounds.

Do not label this monosemantic, a general word-boundary detector, or a UTF-8 circuit. Those broader claims are not established. Static vocabulary writer energy is diffuse (9.81% in its top256 coordinates), so a handful of top tokens cannot define its meaning.

## Next decisive checks

1. Preserve the original-observer grounding result and failed full-fidelity gate. Rank-three native effect error is5.72% on continuation sites but33.44% on spaced-word sites; near-zero spaced-word CE is not sufficient.
2. Test a frozen prediction on code or controlled tokenization examples, keeping current-token identity and perturbation magnitude explicit.
3. Fold the scalar readers backward through the preceding MLP and test whether their production can be extracted more simply without losing the behavioral effect.
4. Check composition, interventions on unrelated behaviors and robustness across discovery splits before adoption.

Evidence: `MIDPOINT_STABLE_GROUP_REMOVAL_NATIVE_V1.json`, `MIDPOINT_STABLE_GROUP_PROFILE_V1.json`, `MIDPOINT_STABLE_GROUP_BOUNDARY_AUDIT_V1.json`, `MIDPOINT_CONTINUATION_GROUP_NATIVE_V1.json`, and `MIDPOINT_CONTINUATION_GROUP_AUDIT_V1.json` in the parent directory. [Timed explanation](../../explanations/for_logan/research_update_2026-09-21_0346_continuation_candidate.md).

## Original-weight grounding update, 03:56 UTC

[Grounding report](../../explanations/for_logan/research_update_2026-09-21_0356_original_weight_grounding.md): exact projection of the original centered source-dependent operator has continuation/spaced CE damage0.13290/0.00017 on the reused32-row panel. Its leading mode gives0.12615/−0.00021 and is nearly identical to the discovered product. This supports the dominant native-component interpretation, not a unique global circuit identity.

Two upstream scalar reads now have exact quadratic forms through MLP16. Isotropic90%coefficient-energy ranks480/485 prevent calling this a small standalone upstream extraction. Native normalization and last-MLP input remain external.

Metric limit: the frozen product has99.35% isotropic coefficient error, versus31.47% covariance-weighted error. Isotropic-optimal rank1 still has95.51% isotropic error and98.99% covariance error. The dominant-component statement is explicitly conditional on the calibration input metric, not a globally simple polynomial tensor.


## 21 September,04:18 — Source extraction and stdlib transfer

Frozen stdlib point-estimate behavioral checks pass; same-current-token code coverage is insufficient. Upstream source forms now have frozen constant/linear and rank16/64 quadratic approximations, tested against native leading-feature removal on newFW144:176 and reused16stdlib snippets. Covariance16 effect-error<=15% allcohorts passes; CE agreement<=.02 fails on FW continuations (.10038 native versus .07486). Linear-only source control also passes the effect bar and has better FW continuation fidelity. Thus global quadratic fit is not a sufficient task-fidelity criterion.

Compact executable z,h interface exports store4612scalars(linear) or41508(rank16), retaining explicit native RMS. Exact export replay <5e-14. Native producers of z,h are still required; composition/reuse and complete upstream extraction remain open. See [report](../../explanations/for_logan/research_update_2026-09-21_0418_upstream_extraction.md), [results](../MIDPOINT_SOURCE_NATIVE_V1.json), [bootstrap](../MIDPOINT_SOURCE_NATIVE_AUDIT_V1.json), and [executable prices](../MIDPOINT_SOURCE_INTERFACE_V1.json).


## 21 September,04:35 — Shared upstream dictionary and donor-family failure

A shared16 input dictionary for both source quadratics reduces41508 to23588 stored scalars with32source squares unchanged. Initial frozen comparison on unusedFW176:192/reusedstdlib passes. Source/context interchange distinguishes quadratic readers from linear-only controls on FW overall. However, balancing donors exposes code spaced-word relative-fidelity failures (hybrid1.115x, change1.148x baseline); absolute15%/20%limits remain satisfied. Not adopted; no independent-task reuse or semantic identity claim. Shared24 holds opened comparisons but needs fresh selection-independent confirmation. See [report](../../explanations/for_logan/research_update_2026-09-21_0431_shared_source_circuit.md) and [package](continuation_source_shared16/README.md).

Math checkpoint: common-input dictionary spectral bounds are restricted coefficient bounds; generalized pencil for shared16 has8complex eigenvalues, obstructing exact common-real-square diagonalization. Mixed-product blocks are the next structural candidate. All current skip11000FineWeb rows are used; do not label further reuse fresh.


## 21 September,04:51 — Exact original-source mixed-product baseline

The full original two source quadratic forms now have an exact numerical block-product implementation:1152source products,1331714floats+3456indices, versus native4608products/10628354floats and independent fullspectral2304products/2658818floats. Signed whitening repaired a rejected raw-pencil attempt without loosening1e-10matrix tolerance. Native balanced-donor replay maxeffecterror1.08e-5/CE9.97e-8 passes. Inputs nativez/h and RMS remain; no wholemodel speedup or newsemanticclaim. Shared16/24 approximations also compileexactly to16/24sourceproducts andretainpreviousfailure. See [report](../../explanations/for_logan/research_update_2026-09-21_0451_exact_shared_products.md).


## 21 September,05:04 — Reuse beyond the original pair is insufficient

Fixed exact1152product dictionary fails additionalnative-mode2/3reader gates despiteoptimal linearcoefficients/no Gram directionsdiscarded. Covariance sourcevariation16–24%; mode2/3errors14.23%/43.96% despitecombined4.98%. Private residualrank16 lowersmode3to28.01% butmisses15%bar. No nativepromotion. Exactquartic numerator coefficient objective verifiedagainstexplicit24-permutationtensors/gradients;40toyfits recover5/5structures, Adam.05 9/10individualfits versusMuon.05 5/10 at400steps. Trained jointfitnotyetrun. See [report](../../explanations/for_logan/research_update_2026-09-21_0504_reuse_limits_and_joint_fit.md).

## 21 September, 05:34 — Additional native-mode joint fitting does not transfer

Mode3 is an additional component of the same output observer, not a new semantic identification. Joint quartic coefficient fitting improved its covariance coefficient score but worsened normalized function error (37.71% to48.97%; [report](../../explanations/for_logan/research_update_2026-09-21_0524_joint_quartic_metric_failure.md)). Rebuilt24training/8adaptation-evaluation comparison confirms direct functional fitting can interpolate training but does not transfer: primary normalized+.01coefficient gives0.09%train/78.18%evaluation versus32.80%/39.98%initial. Frozen32product-weight control avoids severe overfit but stays~40%evaluationerror. No native promotion; leading continuation candidate evidence unchanged.

A full empirical moment identity and exact Gaussian numerator moment kernel are independently verified. The Gaussian metric evaluates the existing initial/coefficient/regularized-functional programs at9.03%/17.19%/20.18%error; optimization of this new metric is pending. It retains native-input distribution assumptions, fixes the constant coordinate, and excludes Gaussian RMS division. See [report and receipts](../../explanations/for_logan/research_update_2026-09-21_0534_functional_moments_and_overfitting.md). General reuse and standalone input closure remain open.

Split correction at05:36: prefix audit found evaluation row28 duplicates training19 over64inputtokens. Original eight-row scores remain but do not qualify as fully separated. Additional seven-distinct-row rescoring, with no refit/reselection: initial39.87%, primaryfunctional79.81%, fixed-productprimary40.07%. All fidelity failures remain. The audit is linked in report0534; future split preflight must exclude exact input-prefix duplicates before fitting.

## 21 September, 05:54 — Restricted rank bound and covariance sensitivity

Exact Gaussian moment fitting reaches8.98% versus9.03%initial, nativeopenedseven40.02%; bothtargetsFAIL. Fixed-affine rank16 Gaussianerror lowerbound8.02% rulesout registered7.22%target for this family, not arbitraryDAGs/nativefunctions. Rank128spectral Gaussian3.19% butnative28.38%; rank64primary35.17%fails15%. Evaluation/training squaredMahalanobis ratio9.41 suggests poor small-calibration geometry. Fixedrank64rho.1 covariance shrinkage35.04%fails; isotropic27.59%betteropened butnotpromoted. Gaussianplanted3/5at400steps; other2recoverat2000verifiedbyexplicitHermitetensor differences. Preparedadditional232deduplicatedtrainingprefixes for256total covariancecalibration, excluding allold32prefixes; capturepending. See [report](../../explanations/for_logan/research_update_2026-09-21_0554_rank_limits_and_input_geometry.md). No new semantic/circuit adoption.

## 21 September, 06:12 — Chunk/document provenance correction

The original FineWeb rowcache can harvest several513tokenchunks from one underlyingdocument and does not retain sourceIDs. Earlier descriptions of32documents and different-document donors based onrowindices are therefore onlyverified atchunk level. Exact-prefix deduplication does notrepair missingdocumentprovenance. Earlier point estimates remain, but document-level bootstrap/generalization claims needreplication withidentifieddocuments. New mode3 confirmation uses32explicitdocumentIDs, oneprefixeach, rejectinganydocumentcontainingaknowncached32tokenprefix. This does notretroactivelyconfirmtheleadingfeature'ssemantics. Overallreview and recentreports carrythecorrection.

## 21 September, 06:45 — Local shared-core fidelity and cross-component reuse limits

Expanded256-chunk metric and shared-input Tucker plus exact pair-core arithmetic compiler improve mode3 opened error16.34% to11.20% at256products. Two document-identified native panels pass absolute effect limits, but primary code-hybrid relative ratios1.183 and1.11014 fail1.10. This is partial extraction with a code tradeoff, not new semantic identification. Three separate256-product pair programs resist common/partial projection sharing: every registered topology misses per-component limits despite lowerstorage. Numerical input-subspace unionrank768 rules out exact narrowing of these frozen linear banks; arbitrary DAGs remain unconstrained. See [follow-up report](../../explanations/for_logan/research_update_2026-09-21_0638_shared_cores_and_reuse.md). Nextjoint512 shared-product fit changes product features themselves; five planted structures verified, trained fit pending. Native input closure and circuit adoption remain unresolved.

## 21 September, 06:54 — Joint products and fixed-readout falsifier

Four4000step shared-square fits: selected512product model has604428floats, modeerrors3.55/2.49/17.54% vsseparate3.06/2.76/11.94%; fidelityFAIL. Independentdense/original-coordinate audit passes. Exactreadoutleast-squares discardsnoGram directions butweakmode17.55% stillfails. Lowerbound6.39%coefficienterror doesnotprove fitted10.30%optimal. Mixed384products changesprimitive atmatchedstorage; fiveplantedcontrols recover. See [report](../../explanations/for_logan/research_update_2026-09-21_0654_joint_products_and_readout_control.md). No nativepromotion ornewsemanticclaim.

## 21 September, 07:07 — Mixed-product and graph-sharing tests

Jointmixed384 atmatchedstorage improvesmode1/2 to2.48/1.87% butmode3 19.74%fails. Denseauditpasses. Removingcross-componentreadoutsharing increasescoefficienterror7.76->32.20% mixed,10.30->33.07% squares; greedyprivate squareassignmentstill31.41%. These arefixed-dictionary computationalsharingtests, notsemanticidentification/orglobaloptimality. Partialgraph256mixedshared+256private exactlymatches897804floatbaseline with512vs768products, butunrefittedmode1/2 6.64/5.10%failsrelativebars. Variableprojectionrefit registered; nativeinputclosure remainsopen. See [report](../../explanations/for_logan/research_update_2026-09-21_0707_mixed_products_and_graph_reuse.md).

## 21 September, 07:48 — Graph refit, fresh comparisons and identity failure

Partial graph refit reduces768to512source products at897804floats, opened scalar errors2.65/2.48/11.94%. Fresh panelV1 all72relative comparisons pass; third-component natural continuation16.73%fails15%. Private320 capacity in both arms gives576vs832products; freshV2 relative passes, same absolute failure16.01% persists (recipient95%11.15–24.92%). No complete adoption. Individual joint atom matching and full-span principal-angle tests fail stability; Gramcondition24–26 and FP32discrepancy<3.3e-6 oppose numerical breakdown. Three consensus directions capture99.994%fitted-function energy but leading one already99.987%, and all256shared products remain: this is stable total function, not three discovered semantic nodes. Consensus native opened replay preserves accuracy without reducing cost. See [follow-up report](../../explanations/for_logan/research_update_2026-09-21_0748_stable_functions_unstable_products.md) and [math review](../../THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-21_0744.md). Original native input dependencies and selective semantic validation remain open.

Additional conditional Jacobian screen: winner original-relative errors20.09/17.93%, tangent20.52/18.61%; finite-difference controls<8e-10. All fits fail15%native derivative bar but crossfit differences<9%. Native h held fixed; not full upstream response and separate-baseline derivative comparison remains untested. Receipt `PROFILED_SENSITIVITY_V1.json`.

## 21 September, 07:54 — Matched sensitivity baseline and source-response fitting

Matched original/separate/shared Jacobians show graph improves Euclidean response23.89->20.09%,19.14->17.93%; private3unchanged19.40%. All9relative checks pass, including covariance-shaped and tangent metrics. Covariance-shaped shared errors4.64/4.63/6.49% contrast with Euclidean failures; no singlegeometry provesOOD. Exact readout solves with derivative/coefficient blend fail10%improvement while preserving scalarfidelity. Five structural toys4/5recover; signedcase retains2.55%failure, includingrestart9473. Managed same-cost product-direction refit registered, primarylambda1 andlambda0budgetcontrol; no successassumed. See [response report](../../explanations/for_logan/research_update_2026-09-21_0754_response_metrics.md). Native h fixed throughout; fullupstream response remains open.

Signed Sobolev toy follow-up: seed5473/rate0.05/4000steps recovers6.95e-10. Seed and rate changed together; original two misses remain. No universal optimizer claim.

Completed source-Sobolev refit: lambda1/.005 primary Jacobians18.565/17.427%, improvements7.59/2.82%fail10%both; scalar2.319/2.590/11.942%passes. Eightfits200.04seconds, fixedcost512products/897804floats. Lambda10haslittleextra responsegain, coefficienterrorworsens. Source-gradientrank512lowerbound19.08%applies to source metric, not nativecomponentJacobian. Exact derivative-error decomposition finds gradienttermcosine>.986but value+crossnorm11.7–16.4%failsstrict10%remainderbar. This supports downstream-weighted gradient work, notglobalimpossibilityoradoption. Receipts SOURCE_SOBOLEV_REFIT_V1.json and JACOBIAN_ERROR_TERMS_V1.json.

## 21 September,08:11 — Random discovery versus internal identity

Eight random shared-direction fits at fixed256mixed+private256capacity recover useful source approximations: coefficient-only8.779–8.787%, selectedscalar2.562/2.549/11.942%; selectedlambda1scalar2.094/2.811/11.942%, derivative18.496/17.572%fails10%improvementbar. Parenthasmorelifetimeoptimization; no equal-budgetglobalconvergence claim. Independent-start atomidentity fails(mediancos.054, noatoms>.90), totalfitdifference4.49%; approximate fits aredifferentfunctions, so no exactnonuniquenesstheorem. Coupled native-amplitude readoutcorrection negligible. Finite45degree nearest-output rotations changefunction13.51%fail2% despite .9993mediancos; poorforcedtailpairsdominatedamage. Preservenulls, nativeportsandfreshmode3failure. [Report](../../explanations/for_logan/research_update_2026-09-21_0811_random_discovery_and_identity.md).

## 21 September,08:21 — Constructive approximate freedom and weak source contrast

Selective79pair45degree rotation changesfittedtensor1.863%atunchanged512products897804floats, affects50.33%atomenergy andmediancos.520; scalar2.660/2.594/11.942%. Deletingthose158products changes70.81%andfailsfidelity(15.22/8.82/11.94%), rulingoutmereemptyterms. Canonicaloutputgrouping acrosswarm/randomlambda0fits: first3corescos>=.996, fourth.948–.955fails.99; originalweakfeaturecos.858. Weakcontrastonly.186%teacherenergy butwhole-read removalcauses7.93/6.97%scalarerrors; centered-remainderremoval1.40/.64%. Preserveweakcontrastandexactaffineports; nosemantic/OODpromotion. Sourcecontrastbalancing rank512bound6.40/7.43/9.65%foroutputwhitening0/.5/1 underdifferentmetrics. [Report](../../explanations/for_logan/research_update_2026-09-21_0821_product_freedom_and_weak_contrasts.md).

## 21 September, 08:43 — Output contrast weighting

Full output balancing makes all four quadratic groups repeatable across tested seeds/rates (minimum cosine .9957), but original-feature cosine .9874 and component errors5.14/3.59% fail fidelity gates. Partial balance preserves scalar gates, fails group fidelity. Independent export/mean/affine replay passes. All-six audit exposes weak combinations involving private3; prepared next graph shares383products across all6reads at896262coefficients. Native ports, selective manipulation, semantic identity, OOD and fresh third-component failure remain unresolved. See [report](../../explanations/for_logan/research_update_2026-09-21_0843_output_weighting_and_group_stability.md).

## 21 September, 08:53 — Global sharing and targeted private residual

Global383fits terminal257.62s; primarypartial-balance values2.658/2.602/18.690% fail third. Fullbalance25.60%third and.98959minimumrestartgroupcosfail. Independentexports pass. Exact valueerror terms localize main third miss to secondsource read; exactreadoracle leaves3.31%, notdeployablefreecapacity. Targetedfixedprojectionbudget exchange16mixed->32privatesquares yields399products896198floats, errors3.812/3.437/17.125%; exactconstrainedreadoutrefit4.390/3.526/17.837%fails. No semantic/OODpromotion or nativeportclosure. [Report](../../explanations/for_logan/research_update_2026-09-21_0853_global_sharing_and_targeted_graph_edits.md).

## 21 September, 09:03 — Joint directions and outer component identity

Four fits154.54s; selectededited399program errors2.576/2.518/16.090%, versuscontinuedglobal3832.477/2.443/18.786%. Registeredabsolute/relativebaseline and20%thirdgainfail. Privatecenteredremoval increases16.09->24.88%, othersunchanged; privatequadraticcos.7066betweenstarts failsidentity. Independentexportpasses. Exactcommonwriterouterfactorrotations preservecombinedwrite butchangeindividualcomponents; this is an algebraicgauge, notpermissiontorotateawayregisteredfailure. Combinedopenederrors2.55/2.57/2.71%edited/global/partial cannotreplacefreshinterventions. [Report](../../explanations/for_logan/research_update_2026-09-21_0903_direction_refit_and_component_identity.md).

## 21 September, 09:10 — Fresh compact-group native interventions

Frozen399graph tested32newFineWebdocs+16newcodefiles, all3individual+combined removals andsame-token cross-document swaps. Nativeinstrumentpasses; original72cells4absolute/25relativefail. Combinedabsolute18/18pass(max6.60%), relativeagainstBOTHseparate768andpartial512fails11/18. All-site combinedFineWebnatural2.720%vs2.471%partial;code2.311%vs1.905%. Codecontinuationhybridratio1.885withpairedrecipient95%interval1.471–2.422, donorsfixed. Exactreportedpointsrecomputed; nativeports and allconstituentfailures retained. No adoption. [Report](../../explanations/for_logan/research_update_2026-09-21_0910_fresh_group_and_constituent_interventions.md).

## 21 September, 09:17 — Native value metric geometry

All24matched-panelGaussianquadraticMSEratios.325–.860fail10%agreement; downstreamsensitivityindependence20/24fails,ratios.305–.980. Compactread6combinedsurrogate.238/.383ofactualweightedtrain/openederror. FiveGaussianformulacontrolsandnativeexecutorreplayspass. ActualnextCPUfiveequal-covdistributionmetrics giveq=x1²-x2²MSE4/0/16/6/3.2; empiricalmetricscanhaveOODblindnullspaces. RevisitedpaperMscope; proposeddatafourth-momenttermmustretainpositivecoefficientconstraint andmatrix-freeevaluation. No newcandidatefit/adoption. [Report](../../explanations/for_logan/research_update_2026-09-21_0917_covariance_and_lifted_polynomial_metrics.md).
