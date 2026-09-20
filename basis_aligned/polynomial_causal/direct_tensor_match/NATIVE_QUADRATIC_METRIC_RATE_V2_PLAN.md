# Full quadratic objective/rate/radial control

V1: all12randomCPfits improve; bestwidth512Muon Gaussianerror56.67%, Frobenius95.97%. NativeQRreplay7.60e-7. These are optimizerresults, not an impossibility theorem. MuonoutperformsAdam atthat rate/budget; toyrestarts show optimizerwinners change withrate andwidth.

V2 freezes a discriminating grid: Muon,width128/512/1024,lr.005/.05,seeds0/1,GaussianversussymmetrizedFrobeniusobjective,600steps,24fits/14400studentupdates. No nativeforwards orcheckpointupdates. Originalexactfullteacher andQRframeunchanged. Reportbothmetrics for everyfit and include a closedformradialquadratic projection Q_o=(trQ_o/d)I inthereducedinputframe. This exposes whetherGaussianimprovement mainlycapturesisotropictrace. Its denseinputframe andoutputwriter arechargeable, not free.

pred_a_replay:originalvsQR<=2e-4. pred_b_improves:allfinalobjectives belowinitial. pred_c_small_error:anyGaussianerror<.1 (mayfail). Tracebaseline Gaussianenergy=(1+2/d)||trQ||², Frobeniusenergy=||trQ||²/d. Compare againsttargetenergies; retainFP32/TF32off caveat andfloat64metricvalidation ifsmallerrors warrantit. No newtext orcausalclaim. No adaptivewidth restriction basedonoutcomes.
