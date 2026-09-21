**Synthetic function fitting interpolates its probes but does not recover the folded tensor.**

Matched32x4features/16writers,100Adamsteps, sameinheritedinitialization. Synthetic-only2048nativeGaussiantrainingprobes versus lambda1hybrid with6144textstates. FreshGaussian1024probes. NumericalintegrityPASS: teacherfold8.26e-15,FP32export6.49e-7. GlobalimprovementFAIL; textpreservationFAIL. Runtime23.73s. Both384products322048coeff768indices.

|Program|TrainingGaussian aggregateerror|FreshGaussian error|Second-panel textroot1weightederror|Second-panel meanrootweightederror|
|Previous text-sensitive fit|notfitonthispanel|117.40%|16.19%|49.23%|
|Synthetic-only|3.53%|94.50%|223.42%|602.22%|
|Hybridlambda1|4.72%|95.24%|17.90%|48.47%|

Hybridglobalerrorimprovesrelativeoldtextfit butmisseshalvingcriterion. Textroot1exceeds1.1old(17.81%) whilemeanrootpreserves; conjunctionfails. Ordinaryhybridtextvalueerror10.63%. Ridge/data-lossratios.00214synthetic/.00208hybrid: a dominatingridgepenaltydoesnotexplainthistrain/testgap. Thisdoesnotestablishoptimizerconvergence orabsenceofsmallcircuits.

ActualCPU successor addedstrongerGaussiancontrols withidenticaltrain/evaluationseeds. Fullradial rho=(||x||²/1152)² withfitted16outputcoefficients gives84.74%fresherror, but1153variableproducts exceeds384budget. Randomorthogonal32projection radial rho=(||P^Tx||²/32)² gives87.40%with33products/36880coeff beforecommonwriter. Bothbeatnew384productfitsonfreshGaussianprobes; neitherisclaimedfaithfulorsemantic. Zero100%alonewasatooweakcomparison. Prices explicitlyexcludecommon1152x16writerforbothradialcontrols; current384programpriceincludesit.

Crucial distinction: sampledGaussianfunctionloss usesnativeweights forlabels, but itdoesnot directly equatecoefficienttensors. A flexibleprogramcaninterpolate2048probes withoutlearningtheglobalquartic. Inputfeatureandwriterinitializationsremaindata-informed, so syntheticonlyobjectiveisnotpureweights-onlydiscovery.

ActualnextCPU work implemented exactsymmetricquartic coefficientobjective forsharedquadraticroots, reusingexistingbank_gram andnative_bank_cross. If G_ij=<phi_i,phi_j>andX_gi=<H_g,phi_i>, solve C(G+ridgeI)=X and optimize tr(CGC^T)-2<C,X>+ridge||C||². Teacherconstant||H||² isomittedbutdoesnotaffecttheoptimizer. Unlikeprobeobjectives, theself/crosscontractionsareexact. Fivefamilies comparedagainstdenseorder5coefficientarrays andgradients; Gram/cross/losserrors<1.7e-15, densegradienterrors<9e-16, profileversusfull-solvegradients<3e-11includingrankdeficientcases. No nativefeatureoptimizationresultyetfromthisobjective. Earlierexactreadoutonlyexperimentsarenotforgotten: thenewquestionisfeature-directionlearningunderexactcoefficientsatthissharedgraphbudget.

CoefficientFrobeniusandGaussianfunctionmetrics remain different. Exactcoefficientoptimization maystillgivepoorbehavior orrequirehighercapacity; it prevents finiteprobeinterpolationfrombeingmisreportedastensormatching. Nativeprojectiontargetstillcontains16fixedreaders, notallvocabularyoutputsorfulltwo-blockfunction. Fullgoalremainsopen.

[Registeredfit](HYBRID_ROOT_FEATURE_PLAN_V1.md) · [Results](HYBRID_ROOT_FEATURE_V1.json) · [Radialbaselines](HYBRID_RADIAL_BASELINES_V1.json) · [Exactobjective](exact_root_tensor_objective.py) · [Densecoefficientcontrols](EXACT_ROOT_TENSOR_OBJECTIVE_CONTROLS_V1.json).
