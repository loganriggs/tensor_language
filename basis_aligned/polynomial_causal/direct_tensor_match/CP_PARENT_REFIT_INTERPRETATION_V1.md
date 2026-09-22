# Parent compression passes Gaussian fidelity but loses native responses

22 September2026,01:12UTC. CP_PARENT_REFIT_NATIVE_V1.json completed13.94seconds,100Muonsteps per seed,768products1202176coefficients. Fixed learnedCP512parent target, not nativeweights. Exact profile/readout and FP32export integrityPASS; parentGaussian<=1%bothPASS; native retention<=1.1parent acrossvalue/response/sensitivityFAIL.

Seed1001: parentGaussian2.390->0.896%; nativevalue8.150->6.886%; sensitivity22.160->14.363%; response27.328->14.950%. Unprunednative6.322/10.005/9.723% respectively.
Seed1002: parentGaussian2.064->0.769%; nativevalue7.653->6.969%; sensitivity17.984->13.650%; response18.965->14.976%. Unprunednative6.588/10.783/11.781%.

Parentcoefficienterrors remain68.018/66.825%, despiteGaussian<1%. This is a metric-specificcompressionresult; no globalpolynomialrecovery. Bothabsolute10%componentcriteria also fail. Directions moving substantially repairs the frozenpruningloss but doesnotpreserve nativeeffectaccuracy.100stepsselected inbothruns, so longeroptimization notexcluded. This doesnotfalsify arbitraryDAGsimplicity.

Do not adopt or reportGaussianparentfit as nativecircuitfidelity. Keepthe improvedhalf-productbaseline. Full finite-removal/OOD/compositionstilluntested. Neithernewparentrefit is includedinthefivecandidatefreshpanelpreregistration. Nextlearningobjective must addressnativefeature/response fidelity explicitly, whiletrackinglargeroutputs andliteralcost. Baselineandexportreplay<1e-7; resultnotattributedtoknown numericalbug.
