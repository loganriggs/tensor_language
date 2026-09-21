**Moving the quadratic features greatly improves calibration, but still fails held-state component prediction.**

Two100-stepAdamfits fromidenticalinherited U/V,32quadraticfeatures4products each,16fixedwriters, perrootexactprofiledridge. Managed26.46s. NumericalexportPASS4.02e-7, transferimprovementFAIL, value/costpreservationPASS. Bothcompiledprograms384products322048coeff768indices. Oneinheritedinitializationpermetric; no convergence orrestartstabilityclaim.

|Program|Calibration meanroot weightederror|Second-panel meanroot weightederror|Second-panel root1 weightederror|Second-panel aggregate scalarvalueerror|
|Fixed sensitive readout|22.24%|47.32%|14.79%|8.30%|
|Learned uniform|11.31%|50.34%|16.84%|9.35%|
|Learned sensitive|6.14%|49.23%|16.19%|8.77%|

The primaryfeature1calibrationerrorincreases2.61→4.28%in thesensitivefit despiteimprovingthemeanof16rooterrors. Optimizingmanynormalizedrootobjectives cantradeaccuracyamongcomponents. The testroot1erroralso worsens; do notattributeallfailuresonlytooverfittingorclaimimprovementfor the previouslyhighlightednewline feature. Ordinaryaggregatevalues weightlarge-outputfeatures differentlythan equal-rootrelative errors.

FivefamilyCPUenvelope-gradientcontrols compareagainstautodiffthroughentireweightedridge solve, includingcancellation(max6.4e-11). End-to-end train/compile smoke passes5.4e-14. Known numericalinstrumentswork; this finite pilot doesnotestablish expressivity oroptimizationimpossibility.

ActualCPU successor:256freshsyntheticGaussian probes N(0,I1152), seed926, evaluatedthroughnativeMLP16→MLP17purequarticweights andsame16frozenreaders. No textactivations ornormalizations inthisdiagnostic. Aggregate nativefunctionalerrors: inherited147.35%,fixedsensitive147.78%,learneduniform135.89%,learnedsensitive117.02%. Allareworsethanzeroprediction's100%inthismetric. Learned sensitivityprogramdiffersfrominheritedby72.53%ofinheritedoutputnorm. Thus thefunctionsareunderconstrainedawayfromtextcalibration, but theobservedmovement doesnotmerelyworsenGaussianaccuracy—it modestlyimprovesan alreadyverypoorbaseline. GaussianfunctionerrorisnotcoefficientFrobeniuserror ornormalizedlanguage-model OODbehavior.

Implication: these activation-fitted decompositions are conditional approximations, not globalfolded-tensorequalities. Pureweights/function objectives mustremainexplicitcontrols. Futurehybridfitwouldneednativeweight-generatedconstraints aswellastext/sensitivityconstraints andreportthetradeoff, notregularizetowardtheinheritedprogramassumingitistruth. None ofthecurrentrootcase-selectivity,full-logitfidelity,stableunit,extraction/OOD/reuse gaps isclosed.

[Plan](SENSITIVE_ROOT_FEATURE_FIT_PLAN_V1.md) · [Fit results](SENSITIVE_ROOT_FEATURE_FIT_V1.json) · [Gradient controls](SENSITIVE_ROOT_FEATURE_CONTROLS_V1.json) · [Synthetic native probe audit](SENSITIVE_FEATURE_GAUSSIAN_AUDIT_V1.json) · [Probe code](audit_sensitive_feature_gaussian.py).
