# Last attention response term test

13 September2026. Builds on exact actualweight FP64control and native lastblock split. Existing160prefixes,800nativefullforwards,160extraMLPcalls and1280finalreadouts,180seconds. Frozen before scoring.

Use native zBar rounded exactly as previous native MLP replay, v=zA-zBar, and exact double-precision bilinear expansion for g(zBar+v)-g(zBar). Terms: v, cross/newrho, square/newrho, rescaled base product. Finalstate backgrounds are unchanged hBar. Eight finalreadouts: hBar, native hBar+Gatt, full formula, direct, direct+cross, direct+cross+normalizer, normalizer alone, square alone. The native prefix runs onlyfive times perrow; finalstate term interventions need no remaining transformer forwards.

A: originalfive full outcomes, base and native attention reference replay<=1e-4relative eachregional/FineWebpanel; fullformula localstate response error<=1%aggregate eachpanel. B: direct+cross effect versus native attentionbranch error<=20%everyregionalgroup. C: direct+cross+normalizer effect error<=10%everyregionalgroup. Report fullformula effect discrepancy and absolute FineWebprecision irrespective ofA, since localstate accuracy doesnotbound tinybehavioraleffects.

Null: normalization/square terms or outputnonlinearity prevent the proposed smallresponse. No fitted coefficients, heldout/OOD orautonomous extraction claim. Candidate code retains fullL/R/D andnativebaseline states. Exactnativeweights andpreviousfrozenresponse algebra reused ratherthan newgenericrankfit.
