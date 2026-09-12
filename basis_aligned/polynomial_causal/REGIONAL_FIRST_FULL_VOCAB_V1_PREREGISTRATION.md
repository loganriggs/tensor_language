# First-token branch: full-vocabulary effect audit

Recompute exact native lastMLP17/finalRMS/capped unembedding from savedprestates,
for baseline andfirst-onlydonorswap. Original32andheldout48rows, no new bodyforward
or fitting. This is broader endpoint measurement on already tested panels.

A:target andoldcontrolcontrast baseline/first-swapreplay<=1e-5relative.
B:A andoff-pair probability-weighted conditional logprob-changeRMS <=.25 times
regional-target logodds-changeRMS in EACHfamily. Exclude only thatrow'sUK/US pair,
renormalize each distribution over remainingvocab, weightbybaselineconditional
probabilities; average squared changes acrossrows then root. C:A/B plus mean
offpairTV<=.005 EACHfamily. Null:binarycontrastselectivity hid broad endpointchanges.
Offpair includes other regional words; this is a stringent pair-local test,
not a semanticlabel for all other tokens. No absent-circuit claim froma failure.

Report full/conditionalKL,TV,baselinepairmass, pairlogmasschange, perprefix ratios,
and descriptivelargest affected tokens separately. Preserve original conditional
contrast successes. CPUonly2threads, mmapcheckpoint, atmost~60MBlogitstate,
no vocabulary-size arrays saved. Perfamily metrics saved (~KB); no dataset download.
