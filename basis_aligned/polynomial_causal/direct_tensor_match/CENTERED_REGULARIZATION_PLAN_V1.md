# Bounded-component refactor — 2026-09-20 20:04 UTC

The8four-productfits have63–1367×sumcomponentenergy/functionenergy despite96.6–96.8%quadraticretention. Test an explicit penalty, after fixing eachquadraticfeature coefficientFrobeniusnorm to1:
L=||T-CQ||² + lambda||C||², lambda in[1e-4,.001,.01]. Analyticwriter C=X(G+lambdaI)^-1. Warmstart eacharchived8fit, preserveitsoptimizer,600steps,lr.005cosinedecay;24refinements. Selectbestpenalizedtrainingobjective; reportunregularizedfit,cancellation,featureagreement.

pred_a: normal-equation relative residual<1e-9; allfeaturecoefficientnorms1within1e-10; gaugecontrol remainsvalid.
pred_b: at leastonepenaltygroup has medianretention>=95% and mediancancellationratio<=10.
pred_c: amongsuchgroups, smallestlambda has medianoutput-componentmatchingcosine>=.85 and minimumindividualfeaturematching>=.8 acrossall8starts. Ifnogroupqualifies,pred_cfails.

No empiricalpanels selectpenalty. This tests compactness versus boundedcoefficients and stability, not semantics. Existingfinitecancellation is consistentwith,butdoesnotprove,CPborder-rankdegeneracy. Scalarprice andproductcountunchanged.
