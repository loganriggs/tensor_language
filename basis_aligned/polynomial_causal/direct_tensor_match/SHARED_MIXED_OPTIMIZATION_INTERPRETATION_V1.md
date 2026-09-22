Shared-feature optimizer controls under mixed exact metrics

22September2026,00:20UTC. Fiveplantedfamilies,tworandomstarts,matchedinitialization peroptimizer/rate. This establishes that the proposed sharedfeatureparameterization canbe optimized on finiteplantedtargets; it isnotnative convergence orsemanticidentification.

Teacher:3quadratics, eachsum2bilinearproducts;2outputs combineall6uppertriangularrootpairs. Families independent/sharedinput/sharedoutput/squares/cancellation. Students4or6quadratics x2products, allrootpairs. Eachquadratic normalized tofixedcoefficientnorm; outputreadout exactlyprofiled withridge1e-8. Optimize average of relative squared fullcoefficienterror and relative squaredGaussianfunctionalerror. Full d^4coefficiententries and5^dGauss-Hermitenodes integrate degree8exactly. These are notfinitely sampledtrainingprobes.

Recovery requires BOTHcoefficient andGaussianerrors<1%. Allarmsreported; bestcheckpoint selected byfittingobjective, notanexternaltextpanel.100or400steps,cosinelearningrate to1%floor. Adam andMuon(match_rms_adamw), rates.03/.1. Nativewarmstart dimensions/metricweight/ridge differ; do nottransfer guarantees.

|Inputdimension|Studentquadratics|Steps|Optimizer|Rate|Recovered|Median coefficient error|Median Gaussian error|
|---|---|---|---|---|---|---|---|
|3|4|100|Adam|0.03|2/10|4.82988%|1.21193%|
|3|4|100|Adam|0.1|5/10|1.57436%|0.73343%|
|3|4|100|Muon|0.03|0/10|8.02825%|5.45910%|
|3|4|100|Muon|0.1|2/10|3.17680%|1.90847%|
|4|4|400|Adam|0.1|4/10|1.29279%|0.97935%|
|4|4|400|Muon|0.1|10/10|0.00807%|0.00479%|
|3|4|400|Adam|0.1|7/10|0.12884%|0.10037%|
|3|4|400|Muon|0.1|8/10|0.00918%|0.00714%|
|4|6|400|Adam|0.1|9/10|0.36292%|0.30412%|
|4|6|400|Muon|0.1|10/10|0.00609%|0.00357%|
|3|6|400|Adam|0.1|10/10|0.00001%|0.00001%|
|3|6|400|Muon|0.1|10/10|0.00005%|0.00001%|

Redteam of the positive result: d3homogeneousquartics have15independentcoefficients. The6-quadraticstudent supplies21products, so it can generically span theentirequarticspace beforefeaturelearning. Its10/10result istherefore weak optimizer evidence andmust notbe usedalone tochooseanoptimizer. We repeated in d4, whereambientdimension35 exceeds both10and21productcolumns. Actualinitialdesignranks10/21 andlargeinitialerrors are recorded; these fitsrequirefeaturelearning.

In d4width4, Muonrecovers10/10 versusAdam4/10 after400steps, frommedianinitialcoefficienterror92.73%. This supportsMuon as the firstnative sharedproduceroptimizer once resourceprofile passes. The100step d3comparison insteadfavoredAdam5/10 versusMuon2/10, demonstratingwhy a shortschedule alone canreverse thedecision. Keep bothcontrols in the record. Widerd4results in thetable also testsparecapacity withoutspanning theentirepolynomialspace.

Preserved nuance: d3usesdiagonalGaussianstandarddeviations[.7,1,1.4]; d4useslinspace(.7,1.4,4). Both includea seedednonzeromean. The code briefly generalized thed3scale tolinspace duringd4runs; it wasrestoredbeforecommit so archivedd3resultsremainreproducible. No nativehelper orqueuedscriptwasmodified.

Decision: retainqueuednativegradientprofile unchanged. Ifpractical, firstsharedproducerfit usesMuon, normalizedquadratics andfixedroot-support atmatched1088productcost; schedulelength chosenfromnativecost, withfailedshortschedule treatedas inconclusive aboutglobalarchitecture. Actualnativeeffect/OOD testsremainnecessary andthelargerCPfinite-removalbaseline remainsbinding.

Artifacts: [implementation](check_shared_mixed_optimization.py), [initial100-step controls](SHARED_MIXED_OPTIMIZATION_CONTROLS_V1.json), [d3width4long](SHARED_MIXED_OPTIMIZATION_W4_S400_V1.json), [d3width6long](SHARED_MIXED_OPTIMIZATION_W6_S400_V1.json), [d4width4long](SHARED_MIXED_OPTIMIZATION_W4_S400_D4_V1.json), [d4width6long](SHARED_MIXED_OPTIMIZATION_W6_S400_D4_V1.json).
