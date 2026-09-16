# CrossFirst per-layer causal-Hessian allocation V1

## Question

The frozen CrossFirst child/remainder hierarchy has exact interface
containment but its separately measured behavioral effects miss additivity by
5--47% of the child effect.  The earlier boundary-reset curve is cumulative,
uses a different background at every boundary, and is not an additive layer
attribution.  Test the interaction-decomposition briefing's Section 5 proposal
on this concrete failure: propagate the two head9 removal tangents through the
native suffix, apply each native stage's mixed Hessian once, and propagate each
local term linearly to the two output readers.

This is a diagnostic on the already-open 96-row regional panel.  It makes no
new OOD claim and does not select a circuit from behavioral outcomes.

## Frozen object and stages

The input edits at the post-attention9/pre-MLP9 residual are the exact frozen
child and remainder removals

`a = -child * w`, `b = -remainder * w`,

where `w` and both scalar fields come from the previously validated native
CrossFirst executor.  The ordered differentiable stages are:

1. the MLP9 residual step;
2. attention-residual and MLP-residual steps separately for blocks 10--17;
3. final RMS, unembedding, and score softcap as one readout stage.

For stage `l`, propagate first tangents `a_l,b_l`, compute
`q_l = D2 F_l[a_l,b_l]` by nested exact JVP, and contract `q_l` with the exact
reverse-mode derivative of the remaining suffix.  The sum of all 18 scalar
terms must equal the direct second mixed derivative of the complete suffix.
Prompt contributions remain separate in the artifact; family aggregation is
reporting only and never defines a factor.

The target reader is the row's UK-minus-US logit margin.  The control reader is
the frozen unrelated-token margin.  Four deterministic random equal-norm
writer directions reuse the exact child/remainder scalar fields as specificity
nulls.  They receive no finite-effect or behavioral selection.

## Frozen predictions

- **A -- instrument:** native replay relative error is at most `2e-6`; for the
  real writer and all four random writers, summed stage allocation versus the
  recursively propagated complete Hessian has relative error at most `2e-5`
  for both readers.
- **B -- finite relevance:** for every 24-row regional family, the actual
  writer's second-order target prediction leaves at most `0.50` of the observed
  finite child/remainder interaction norm.
- **C -- sparse allocation:** within every family, the three largest stage
  Frobenius energies contain at least `0.80` of total stage energy and the
  participation ratio is at most `4.0`.
- **D -- random specificity:** the actual writer's pooled top-three energy
  fraction exceeds the median random writer by at least `0.10`, and its pooled
  participation ratio is at most `0.80` times the random median.
- **E -- context resolution:** the artifact records every row/stage term and
  the result reports per-family stage RMS, signed mean, and sign fractions; no
  prompt averaging occurs before factorization or allocation.

If A fails, the run is invalid.  If A passes but B fails, the finite
composition error is materially higher-order at the actual edit scale and a
second-order DCT allocation is not an adequate behavioral explanation.  If B
passes but C/D fail, the causal Hessian is relevant but not a sparse reusable
layer graph.  Only A--E together license a sparse second-order allocation
candidate; they do not license autonomous extraction or selective removal.

## Price and scope

One checkpoint load; 96 already-open prefixes; no parameter updates, fitted
coefficients, rank/support search, or fresh outcomes.  Five fixed writer
directions (one real, four random) are evaluated with nested JVPs through 17
native suffix stages; baseline suffix graphs and reader gradients are shared
within a row.  Native upstream states, exact child/remainder scalar generators,
the full suffix weights, and output readers remain charged ports.
