# Article-confound correction: token routing V2

Corrected The British/The American panel; same cases and fidelity bars as V1.
Original V1 semantic interpretation was confounded. Other frozen weights,
zero-anchor recipe (where applicable), and diagnostics are unchanged.

# Weights-only producer routing baseline

Same frozen first-token branch and fresh32behavior-panelrows. Replace only
upstream routing atA8/A9/A13 with QK projections of normalizedtoken-first inputs,
usingactualfrozen QK1/QK2 weights, nativeQKnorm and roundedRoPE. No datafit.
Currentvaluebranch cancels infirst-source intervention. Downstream17ports staynative.
Fivearms:baseline,nativefirstswap,puretokenroutingfirstswap,nativeQ/tokenKhybrid,
tokenQ/nativeKhybrid. Both QK factors move together on each side.

A:nativeproducer/attention andsavedfirstbranchwrite replay<=1e-5relative.
B:A+puretoken branch writeerror<=.1relative toactualfirstwrite across32rows.
C:A/B+regional signedprefixeffecterror<=.1; eachcontrol RMS<=.05nativeRMS,
meanabs<=.25nativefirst-regional-effectRMS,no baselinecapableanswerflips.
Hybrids diagnostic only; originalpuretoken verdict staysfrozen.
Null: naive token embeddings miss contextual routing information. Known gain
baseline and repeated-residual computations can invalidate this approximation;
a miss does notreject weight-onlystructure or optimizedfactorization.
Price6bodyforwards32rows,5suffixarms,3extra producerQKcomputations perbatch,
180sec managedcap,~1.7MBartifact. Allfirstsourcechangesatcueposition1 asserted.
