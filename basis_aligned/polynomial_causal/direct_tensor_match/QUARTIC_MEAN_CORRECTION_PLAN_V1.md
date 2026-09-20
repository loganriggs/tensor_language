# Frozen quartic plus Gaussian mean correction — 2026-09-20 19:40 UTC

Follow-up to mean/variation audit: centered8-product totalerror23.12% is lower than quartic25.28%, but variationerror34.99% is worse than27.81%. Test whether constant flexibility explains the aggregate comparison.

Freeze training-selected ('second_floor01','muon',1) quartic. Fixed output ranks10(original),8(primary),4(descriptive); truncated writers from exact metric-weighted SVD, no evaluation selection. Add b=E_N(mu,cov)[teacher−student] from native weights and calibration input mean/covariance. These statistics are used offline only; inference stores just b. Original10root costs49,536 withbias (over priorbudget); rank8costs47,312,rank4costs42,664. All26products.

Evaluate both cached panels: uncorrected; Gaussianbias(primary); empirical calibration residualmean(diagnostic,data-fitted); panel2 residualmean(oraclebound, NEVER exported ascandidate). Report total/centerederrors and Gaussian vs actualmean mismatch. Means from evaluation panel never choose configuration or candidate coefficients.

pred_a: small independent moment quadrature<1e-10; allbiasvariants have centeredpredictiondifference<1e-10 inFP64.
pred_b: primary rank8Gaussianbias improves panel2total error below.23122887 (centeredmodelbenchmark) at47,312coefficients.
pred_c: Gaussianbias rank8panel2error within.03absolute of empiricalcalibrationbias; failure implicatesGaussianhigher-moment mismatch rather than silently treating covariance asfull data law.

No teacher forward passes or coefficient refitting. Native fullquarticGaussianmean exact underdeclaredlaw, notexactempiricalmean. Scope remains selectedpurequartic; no fullmodel/semantic/OODclaim.
