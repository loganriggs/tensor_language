# Signed-square stopping repair — 10 September 22:31 UTC

Same256squares, full-U coefficient objective,lambda.01 andimmutableV2CHUNK00best.
Normalize rawreaderrows; the model already uses normalizedrows, so this changes
coordinates without changingthe function. FreshL-BFGS lr1,max_iter20,history8,
strongWolfe,tolerance_grad1e-14,tolerance_change1e-18. No new data,rank orpenalty.

A: gaugefunction/gradienttransport controls<=1e-10; initialobjectivebridge<=1e-10;
initialoriginal-gaugestationaritybridge relative<=1e-6; finiteobjective and
regularizedGramcondition<=1e12; checkpointfinalreplay<=1e-10.
B: fivechecks plateau<=1e-5 ofcapturedenergy AND relative stationarity<=1e-4
in BOTH unit-reader coordinates and originalperrowgauge; bothmaxgradients<=1e-7.
This prevents rescalingalone from bypassing the original convergencebar.
C: finalpenalizedobjective improves by>=1e-10 and retainsoldrawcapture within1e-8.
This is a stopping-repair prediction; originalone-percentage-pointgainfailure
remainsfailed. Noneofthese gates establishesglobaloptimality orcircuitidentity.

120s initialpolish,timerbetween fullL-BFGScalls,25unchangedcallsstopasunfinished.
Savecompactfinalparameters/writers/diagnostics~5MB; no optimizerhistory retained
for thispolish. Ifunfinished, continuation isexplicitly a freshrestart fromthe
savedfunction. Original49.6MBresumablecheckpointpreserved. Require15MBfree,
900s alarm, managedlane1; no competingGPU or corpusaccess.
