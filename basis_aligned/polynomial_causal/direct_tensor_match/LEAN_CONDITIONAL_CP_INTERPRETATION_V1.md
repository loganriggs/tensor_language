# Reusing quartic pair products removes the conditional program's extra product cost

22 September2026,01:43UTC. Concrete second-stage graph edit of the conditional CP programs. The fullconditional atom computes six pair products and then one quartic product. Keep only the two correction terms whose pair products form the quartic. Four additional pair-product nodes per atom disappear; constants remain.

Three global pairings are possible: (01)(23), (02)(13), (03)(12). Select one per whole program by4096syntheticGaussian fit probes, not text labels. Independent4096Gaussian probes evaluate parent error. No continuous refit. This is an approximation to the conditional mean, not its exact formula.

| Seed/rank | Held Gaussian parent error: full→lean | Opened256doc native value error | Old root1 response | Root1 sensitivity | Floats incl. writer |
|---|---:|---:|---:|---:|---:|
|1001/64|2.698→3.423%|9.640%|12.946%|14.359%|234512|
|1002/64|2.639→3.204%|9.602%|13.064%|14.828%|234512|
|1001/128|1.804→1.823%|7.873%|11.285%|12.563%|439312|
|1002/128|1.721→1.738%|7.979%|12.137%|13.084%|439312|
|1001/256|0.876→0.901%|7.244%|10.266%|11.189%|848912|
|1002/256|0.787→0.820%|7.457%|10.551%|11.684%|848912|

All six lean programs use1536variableproducts, matching originalCP512parents, versus3584fullconditional. Rank256densecoefficientmultiplications828416/additions828160 excludingfixedwriter, versus2367488/2365424parent. Floats848912versus2385920parent. Oneglobalpairingselector, not per-atomsearch.

Rank64 loses moreGaussian accuracy; at128/256omittingtheextraquadratic nodes changesparenterrorlittle. Rank256native response essentiallymatchesfullconditional10.260/10.543%, now10.266/10.551%. This doesnotestablishsemanticfidelity: sensitivitystill>10%, smalloutputfeaturesremainpoor, nativefiniteeffects pending.

Controls comparelean evaluator with thefullconditional evaluator aftermaskingomittedcorrections for allthreepairings; equality<1e-12. ExportFP32checks cover16384openedstates. Initialauditcalledthewrong evaluator forfullbaseline andfailedKeyError before resultsexport; correctedcaller andreranallarms. The frozen fullconditional helper/queuedrunner wasnevermodified.

This is an explicit improvement underproduct anddenseoperation accounting relative tofullconditional, withanobservedsmallaccuracytradeoff atlargerinputranks. Againstnativeprogramgoal itremains a compressionbaseline: no newOOD, selective manipulation, jointcomposition orsemanticidentification. Registeredfinite-removal test comparesbothrank256seeds againsttheiroriginalCPparents, withabsolute andrelativecriteria separate.
