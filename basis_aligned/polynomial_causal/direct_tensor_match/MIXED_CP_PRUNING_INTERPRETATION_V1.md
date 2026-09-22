Pruning a strong fitted CP program does not preserve component fidelity cheaply

22September2026,00:31UTC. CPUexactGram backwarddeletion. ParentisfrozenmixedCP512learnedmodel, notnativeweights. Bothstarts, budgets512/384/256/128terms, mixedprimaryandGaussiansecondary, allreported. Eachdeletion choosesminimumexactregularizedscoreloss andrefitsall16outputs. Columnnormalization makesridge1e-10scale-consistent; fivecontrols comparefirstdeletion againstexhaustive directsolves. Laterinverseupdates checkedbycapturedscoreidentityandperiodicrefits.

IntegrityPASS. Primary256term parentGaussian<=1%FAIL; nativefidelityretention<=1.1parentFAIL; absolutecomponent<=10%FAIL. No adoption orfinite-removalexperiment promoted. These areopened-paneldiagnostics; no freshOODresult.

|Start|Terms|Variable products|Gaussian error relative to fitted parent|Native16-output text error|Native root1 sensitivity error|Native root1 same-token response error|
|---|---|---|---|---|---|---|
|1001|512|1536|0.000%|6.322%|10.005%|9.723%|
|1001|384|1152|0.785%|6.614%|12.792%|15.317%|
|1001|256|768|2.390%|8.150%|22.160%|27.328%|
|1001|128|384|4.735%|10.574%|25.795%|38.146%|
|1002|512|1536|0.000%|6.588%|10.783%|11.781%|
|1002|384|1152|0.626%|6.814%|12.738%|13.774%|
|1002|256|768|2.064%|7.653%|17.984%|18.965%|
|1002|128|384|4.609%|10.623%|23.530%|37.097%|

The384term candidate removes25%ofCPterms, preservesaggregateGaussianparentfunction within.63–.79%, andbarelychangesaggregate texterror, butcomponentresponses degradefrom9.72/11.78%to15.32/13.77%. At256terms(768products), text7.65–8.15%stilllookspromising beside the1088-productsharedbank~20%, yetroot1response18.97–27.33%andsensitivity17.98–22.16%erase muchofmixedCP'sadvantage. A 384-product CPprogram(128terms) has~10.6%textbut37–38%responseerror, andstillstores610304floats versus322048forold384graph. Do notclaim a cost/fidelitywinfromproductcountalone.

TheGaussian-onlysecondary doesnotrescuethispattern. Normalresiduals<1.3e-13 anddeletionformulaerrors<5e-16support numericalcorrectness. Thedeletionsearch isgreedy,not a global sparseoptimalitycertificate. Moregeneral graphfactorization couldretaincomputations thattermpruningmisses.

Posthoc explanatory audit, not fitting change: root1accountsfor10.92%ofparentGaussianoutputenergy; it isnotan almostignoredoutput. At384terms itsownGaussianparenterror is1.11/.80%, andcenteredGaussianerror1.29/.93%. Removingthemean changesaggregateerror to1.27/1.01%; varianceis38.05%oftotalGaussianenergy. Thusmeanenergy explains someofaverage-versus-variationdifference, butdoesnoteliminatethelargerconditional-responsefailure. At256termsroot1centeredGaussianerror3.57/2.74%stillcoexistswithmuchlargerselected-nativeeffecterrors. Thisdoesnotproveasinglecause: distribution/conditionalmetric andnormalization sensitivitiesremainmaterial.

Nextaction remainsregisteredsharedproducerlearning, nowqueuedafterpassingnativegradientprofile. PreserveunprunedCPasstrongercomponentbaseline. A futurecompressionobjective mayneed explicitresponsegeometry; adding empirical labels alone previouslyfailedtransfer, so thatisnotanassumedrepair. No semanticselectivityclaim iscreatedbyreproducinganoperationalprojection.

Artifacts: [results including all secondary arms](MIXED_CP_PRUNING_V1.json), [pruning algorithm](cp_backward_pruning.py), [experiment](audit_mixed_cp_pruning.py), [frozen candidate bundle](MIXED_CP_PRUNING_CANDIDATES_V1.pt).
