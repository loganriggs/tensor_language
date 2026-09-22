# Matched continuation tests whether output balancing improves components

22 September2026,01:15UTC. Target unchanged: selected purequartic MLP16selfterm throughMLP17, fixed16readers; all otherpaths and nonlinear normalization remainoutside. This is a data-informed weight-matching experiment, not data-free discovery.

Question: small outputs have70–151%relative error in the learned shared dictionary. Does equalizing output emphasis improve their computations, or simply damage high-energy directions without improving native responses?

Two starts1101/1102, each frozen SHARED_MIXED_FEATURES_SEED artifact. For EACH, run uniform and balanced continuations with identical resetMuon,11steps,cosinerate.1sqrt(4/1152)to1%floor, samefixed512rootpairs, normalized144quadratics×4products andridge1e-6. Same1088products1353728floats1024indices. Estimated~40minutes total from measured53seconds/step; managedqueue only. Keepgradienthelpersunchanged.

Balanced outputweight w_g is inverse calibration mean-square target, floored at largestenergy/1000 and normalized tomean1. Exactnumbers in OUTPUT_BALANCED_CALIBRATION_METRIC_V1.json use6144calstatesonly. This caps relative emphasis1000; it doesnot fullyequalize allsmallfeatures. Teacher outputs and candidate readout are scaled bysqrt(w) for the existingexactgradient. Readout solve itself is unchanged because output-weighted ridge scalesconsistently. Shape/decomposition of teacher projection transforms with the same outputfactor. Controlagainst directweightedautodiff on5structures passes<1e-10; actual144×4/512pairs/16outputsCPUdryrun covers newoutputbroadcasting and4-armreports.

Matcheduniformcontinuation is essential: comparing with the old11-step checkpoint alone would confoundweighting withanother11steps. Both armsselect lowest ownfittingobjective, neverevaluationlabels. Preserve all4arms regardlessoutcome.

Registeredcriteria:
- Integrity: initialphysicalprediction replay<1e-4, FP32export<1e-4, normalresidual<1e-8, finitegradients. No readout-onlychange atinitialfeatures beyondroundoff.
- Balance: BOTHseeds balancedRMSof per-feature relativeerrors(features4–15)<=.85matcheduniform. RMS= sqrt(mean_g(errorenergy_g/targetenergy_g)); not pooledsmallfeatureerror.
- Retention: BOTHseeds balancedpoolederror<=1.25uniform androot1same-tokenresponse/sensitivityerrors<=1.10uniform. Report absolute10%componentbar separately; relativecriterion cannotpromotealreadybadcomponents.

Score on previouslyopened2048statepanel forregistered screen. No new heldout/OOD claim, no tuning theweights afterresults. Newlyopened256documentpanel remainsavailable onlyasexplicitfollowup, notfreshvalidation. Ifpositive, seek newdomain/newdocuments and finite-removal atpricedcost beforepromotion. Ifsmallfeaturesimproveandresponseworsens, recordobjectiveconflict; do notcallrecovery. Ifbotharmsimproveequally,moreoptimizationexplainsgain. Ifneitherimproves, thisshortbudgetdoesnotexcludealternativemetricsorDAGs.

Next circuit-level consequence: improved per-feature and response fidelity would justify extracting candidate computationalnodes for nativeintervention. Reconstruction alone remains insufficient; this run diagnoses capacity allocation, not semantic identification.
