# Second-stage graph edit: share CP input readers

22 September2026,02:35UTC. Frozen CP512 parents, seeds1001/1002. Identify repeated signed linear readers exactly; approximate proposals use each reader's nearest different reader under the calibration second-moment metric S S^T+mu mu^T. A local near-match is only a proposal, not proof of downstream interchangeability.

For source reader a in a quartic atom with other three factors b,c,d, propose replacement q and solve the scale beta minimizing E[((a-beta q)bcd)^2] exactly under the calibrationGaussian. This uses eighth-degree moments, not unweighted linear regression. Rank proposals by their exact edit energy times the output-vector norm. Select32/64/128 disjoint-atom substitutions; primary128. Never remove a retained target reader; only one source per atom. No native evaluation labels select proposals or beta. Fold beta into that atom's output coefficients. Fixed trainedfunctions, no native refit.

Compile the resulting unique linear-reader bank with indexed quartic consumers, preserving arbitrary cross-slot reader reuse. Count stored coefficients and2048reader indices; unchanged1536variable products unless exact intermediate duplicates independently emerge. Primary128merges save147456projectioncoefficients (~6% total), not a claimed dramaticcompression.

Controls: exact shared readers/squares/fourthpowers/sign flips/crossslotpermutations replay; independentGaussianquadrature verifies beta and downstream error for a proposed merge. Native screen: compiledreplay<1e-10; primary parent-relative value AND all16matchedresponseerror<=1% on originalopened2048panel, bothseeds. Also report nativeerrors and smallerfeatures, plusopened256documentdiagnostic. A cheap inaccurate merge is rejected; no circuitsemantic/OODadoption.
