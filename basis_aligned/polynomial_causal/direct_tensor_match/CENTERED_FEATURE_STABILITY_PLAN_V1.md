# Gauge-aware feature stability — 2026-09-20 19:58 UTC

Fixed8×16×16 weightedquadratic target, width4, Adam/Muon at.03, seeds0..3,1000steps. Repeat priorseeds0/1 onlybecause theirper-run factors were notarchived; retainall8fits. No best-fittingselection or nativepanelreuse needed.

Compare all28pairs using: fullreconstructedtensorcosine; four-dimensionalquadraticfeaturespaceprincipalcosines; optimalpermutation mean/minabsquadraticfeaturecosine; optimalpermutation signedoutput-componentcosine. ScalarfeatureQ=sym(a⊗b), componentC[:,k]⊗Q. Feature sign/scale and a↔b ambiguity are removed; output-awarecomponents preserve signedfunction effects. Exactpermutation,rescaling,sign-flipcontrols must replay invariants.

pred_a: gaugecontrol feature/componentmatching andfulltensorcosines differfrom1 by<1e-10; repeatedexistingconfigenergy matchespriorwithin1e-8.
pred_b: allpair fulltensorcosines>=.99.
pred_c: median pair's meanmatchedabsfeaturecosine>=.9 and globalminimum individualmatchedcosine>=.8.

Passing means stability underthisoptimizer/seedgrid, not monosemanticity, datasplitstability or causalidentity. Failure withstrongfunctionagreement means feature interpretation remainsambiguous. Pairwisecomparisonscorrelated,noCIclaim.
