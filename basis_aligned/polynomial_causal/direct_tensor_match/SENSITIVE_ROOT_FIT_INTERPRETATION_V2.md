**Sensitivity-weighted root fitting helps calibration but fails transfer at fixed graph cost.**

Frozen32quadraticfeatures and16writers, same528pairdictionary perroot. Fitrootcoefficientsdirectlytonativeprojectedpurequartic using ordinary versus final-logit-sensitivityweighted scalar leastsquares, ridge1e-6. Fit6144historicalcalstates; evaluate2048openedsecond-panel states. Neither inputfeatures noroutputdirections changed. Sensitivityweights measure eachcomponent'sremoval tangent separately; offdiagonal interactions among16writers are omitted.

|Program|Calibration root1 weighted error|Second-panel root1 weighted error|Second-panel mean of16root weighted errors|Second-panel ordinary scalar aggregate error|
|---|---:|---:|---:|---:|
|Inherited|3.03%|14.54%|45.94%|8.13%|
|Uniform root refit|3.00%|14.61%|46.49%|8.16%|
|Sensitivity root refit|2.61%|14.79%|47.32%|8.30%|

NumericalintegrityPASS, registeredtransferimprovementFAIL, value/costpreservationPASS. Bothcompiledprograms384products322048coeff768indices. FP32replay4.51e-7. Calibrationmeanrooterror27.20→22.24%, butsecond-panel47.32%worsensversus46.49%. Aggregatevalues canbe muchbetterthan averageindividualrootrelativeerrors because eachrootnormalizesitsownenergy. No finite-removal orselectivityimprovement followsfromcalibrationloss.

Precision correction retained: V1usedfloat64epsilon aftercastingfinalstates. An attemptedpre-enqueuepatchfailedbecauseofwrongworkingdirectory;V1ranandisarchived. V2explicitlyusesnativefloat32epsilon, measuringmaxrelativeweightchange5.93e-13. V1/V2verdictsidentical;epsilonwasnegligiblehere, notthetransferexplanation. CPUexplicit-epsiloncontrolusesnearzeronormstateswhereepsilonmatters, versusforwardAD. Fiveweightedleast-squarescontrolsagreewithaugmentedSVDsolves<5e-16. Correctedmanagedrun~5s;see receipt forprecisetime.

CPU successor: fixedroot1 ridgefeature-leverage diagnostic oncal/evalstates. Median evalleverage isbelowcalibration forbotharms. Top10%evaluationleveragepositions contain only5.09%weightederror(uniform) or2.02%(sensitive), with8.27%/6.80%referenceenergy. Only20/2048uniform and6/2048sensitiveevalstates exceedtheirtraining99thpercentile. Thus failuredoesnotconcentratein the most extrapolative states underthisparticularfeature-designmetric. Thisdoesnotprove distributionmatch orruleout missingfeatures; featureleveragecannotsee informationabsentfromitsdictionary.

Scientific consequence: downstreamsensitivityexplainedwhyordinaryscalaraccuracywasmisleading, but changingreadoutweightsalone didnotresolvegeneralization. Do notpromote sensitivity-weightingasafix. The nativecaseprojection itselfstillfailsselectivity independentlyofhowwellitsrootisfit. Nextfeaturelearningcomparison musttest whether changingquadraticdirections improves held-statecomponentresponses atmatchedcost, keepingordinary/coefficientcontrols; broadDAGproposalstillincomplete. Merelyrewritingthealreadyfitprogram morecheaply cannotrepairtheseerrors.

[Plan and correction](SENSITIVE_ROOT_FIT_PLAN_V1.md) · [Corrected results](SENSITIVE_ROOT_FIT_V2.json) · [Original results](SENSITIVE_ROOT_FIT_V1.json) · [Controls](SENSITIVE_ROOT_READOUT_CONTROLS_V1.json) · [Feature-leverage diagnostic](SENSITIVE_ROOT_LEVERAGE_AUDIT_V1.json).
