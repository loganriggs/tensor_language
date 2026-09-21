**Exact native coefficient optimization works numerically, but the current graph remains far from the target.**

25Adamsteps,32quadraticfeatures4products each,16fixedoutputwriters. No text orGaussian probes inoptimizationloss. SelfGram/nativecrosscontractions areexact symmetricquarticcoefficient contractions; profiledridge1e-6, teacherconstant omitted. Native115.54s, peakallocated13.784GB. InstrumentPASS, registeredcoefficient-improvementPASS, functionalpreservationFAIL. Bothinitial/finalcompiledgraphs384products322048coeff768indices.

|Program|Uniform4096coefficient-query error|1024Gaussian functionalerror|Openedtext aggregate scalarerror|
|Initial exact-readout fit|99.90%|99.89%|50.48%|
|After exact feature learning|98.47%|96.35%|57.70%|
|Inherited data fit|notcomputedhere|147.09%|8.13%|

The negative regularizedobjective changes-.00030419→-.00331560, about10.90timesitsinitialmagnitude. Thisisnot a10.9-fold reductioninreconstructionerror: omittedteacherconstantisunknown. Queriesaboveareindependentdiagnostics, not anexactrelativeFrobeniuscertificate. Exactcoefficient andGaussianfunctionalmetrics differ. Bestselectedstep25; shortpilotdoesnotestablishconvergence. NeitherFrobeniusgainnorfaithfulexportestablishesbehavior/semanticidentity.

Two competing limitations remain. Architecture has atmost256linearinputreaderdirections(32features x4products x2factors), evenbeforeitsquarticstructure restrictions. Earlier full-outputisotropicJacobianboundsalreadyruledout256readersfor10%localderivativeerror; thoseboundsarenotautomaticallyvalidfor this16readercoefficienttarget. Separately, optimization remainsdifficult evenonknown-representabletoyproblems.

ActualparallelCPU optimizer audit: fivefamilies, twostarts each, exactcoefficientobjective; allplantedfactor solutionsreplaybelow6.2e-8. At100constant-rate.01steps, Adam0/10andMuon0/10reach1%error. With1000stepspluscosine1%floor, Adam1/10andMuon2/10reach1%; medians4.75%and8.64%. Onlycancellationfamilyreaches1%intheselongruns. Durationandschedulebothchanged, so donotcallthisapuretraining-lengthcomparison. Adamhasbettermedian,Muonmorethresholdsuccesses; no universaloptimizerwinner. Thesearepost-pilotdiagnostics, not retrospectivejustificationforitsoptimizer.

This prevents interpretingnativefailureasproofthatnoeconomicalcircuitexists. The next decision must separate capacity fromoptimization, and comparearchitectures thatavoidthecurrentnarrowsharedinputspan, withliteralcosts. Repeatingmoredata-onlyfitsorcallingthelargeexactobjectivegainasuccesswouldmisstateevidence. Original fullthird-ordertensor/deeperDAGgoal remainsopen; thisexperimentcoversone16-outputpurequarticprojection.

[Plan](EXACT_ROOT_FEATURE_NATIVE_PLAN_V1.md) · [Native receipt](EXACT_ROOT_FEATURE_NATIVE_V1.json) · [100step controls](EXACT_ROOT_RECOVERY_TOYS_V1.json) · [1000step controls](EXACT_ROOT_RECOVERY_TOYS_V2.json) · [Exact dense objective controls](EXACT_ROOT_TENSOR_OBJECTIVE_CONTROLS_V1.json) · [Earlier full-output reader bounds](QUARTIC_READER_RANK_INTERPRETATION_V1.md).
