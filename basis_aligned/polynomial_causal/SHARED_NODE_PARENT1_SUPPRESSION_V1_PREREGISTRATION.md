# Separate-prefix suppression and branch-specificity screen

The original support prediction failed: deleting either frozen branch improved
its target-family CE. The sign audit found negative direct target writes in
87.5% of both own-family samples; normalization was a smaller correction.
The own-minus-other intervals included zero. This follow-up tests a **new
suppression hypothesis** and stronger branch specificity; it does not change
the original failed verdict or refit the weight-derived factors.

Keep parent1 branches0/1, the two original top-token families with overlap
excluded, and the native background, bias, RMS normalization, unembedding and
tanh. Remove branch0, branch1, and both. Positive CE added is damage.

Rows are frozen in `SHARED_NODE_PARENT1_SUPPRESSION_V1_ROWS.pt` with full metadata.
Sources are cached FineWeb prefixes: n480/skip80, n96/skip1200, n192/skip7000.
Exclude every original n192/skip11000 full prefix and exact duplicates by token
hash. The three sources yield128 eligible prefixes, all retained and ordered
with seed4801. For each prefix take the first family0 target after128 inputs,
the nearest family1 target, and nearest target outside both families, breaking
ties earlier. These give384 endpoints. Bootstrap the128 prefix units with shared
indices across families. Raw document identity and near-duplicates have not been
independently verified; these are separate candidate-validation prefixes, not a
virgin corpus or OOD panel. Other research previously used these caches.

- **A:** frozen source/row checks, finite metrics, exact body counts, and native
  and branch0 physical endpoint logit replay within1e-5 relative error; MLP
  input agreement within1e-6.
- **B, conditional on A:** native actual-target top20 fraction at least0.5 in
  each target family; own removal mean CE added at most−0.02 in each; each
  individual branch mean absolute control CE change at most0.05.
- **C, conditional on A/B:** both own-minus-other mean CE changes at most−0.01
  and both paired prefix-bootstrap95% upper bounds below zero. Use2000 bootstrap
  replicates with seed4802, fixed before outcomes.

The null is nonreplication of suppression or failure to separate the branches
by token family. Report all effects and intervals even if a bar fails. Shared
output loadings partly predetermine token-family comparisons; a passed screen
still needs input-gating controls, stronger context variations, extraction and
OOD prediction before circuit identification.

Reuse the first runner in a separate frozen variant. Price:48 batches of8 native
prefixes plus two physical controls, **50 body forwards,400 sequences,128 input
tokens**. No fitting,900-second alarm. Save only endpoint ports and metrics.
The 5760-coefficient branch bank retains the native model background. Queue
behind the live retained-history fit through the standard managed gates.
