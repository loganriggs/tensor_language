# Refine aggregated sources while retaining a tied-parent baseline

Split early0–3andmiddle4–7writes into their individualattention/MLPwrites; split the residualcomplement intoattention8,attention9,MLP9,attention10. Retainembedding,MLP8,MLP10. Total23ports. Observeactualwrites throughforwardhooks inside the existingprefix, then apply eachlaterlambda0 in originalfloat32order. Noextra prefixpass.

Eachparent's lastchild absorbs the floating-gauge correction so summingchildren reproduces the oldsixsource deltas. Require collapse<=1e-10 and correctionnorm<=1%ofexplicitlastchildnorm (perbatch/parent). Reportallcorrections. With amplitudestiedwithinparent, the sixportcandidate is exactlyembedded. Plantedstate/gradientreplaypasses~1e-15. Nativefolded9-outputgradients must replay saved sixportgradients<=1e-8; nativeunitBendpoint must replay prioroutput. FixedFDrelative<=1e-3.

Use same first-orderLP,8controlbudget,box[-1,1],sharedoppositefit/congruentheldsplit as SHARED_SELECTIVE_SOURCES_V1. Sharedprice46coefficients versus12; perinputoracle23versus6perrow. No claim that morecapacity alone identifiessemanticfeatures. Nativegates unchanged:80%alignedunitBretention and10%maxcontrolrelativeeffect. Reportsharedandoracle,old6baseline,allcells. Nativeboundaries are proposed sourcecomponents,notassumedcircuits.

Counts12prefix24double72gradient28native. All23sourcegenerators remainnative, so extraction remainsincomplete. Opposingpredictions: improvedsharedselectivity implicates aggregation as alimitation; onlyoracleimprovement leavesreusabilityunresolved; noimprovementrequires richerwithin-write features orchangedcontrolsemantics, not repeatedscalarcoefficienttuning. No freshOODclaim.
