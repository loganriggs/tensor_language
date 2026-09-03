# Rung 544: four-closer capability and complete-state site gate

**Frozen:** 2026-09-03 15:26 UTC, before any R544 model forward

## Decision

Does the balanced R543v2 dataset express a real four-valued pending-delimiter behavior, and is either the residual
stream entering block 8 or the complete pre-projection state of attention head L13H8 a live causal location for both
answer-changing constructions and all three answer-preserving controls?

This gate precedes any learned subspace. It also blocks the binary-output shortcut seen in R540 by scoring all 12
ordered changes among `)`, `]`, `}`, and `"`.

## Data and outcome boundary

Use only FIT (96 semantic groups) and SELECT (48 groups), with all five constructions in each group. FINAL_TEST and
OOD remain unopened. R543v2 row, receipt, builder, test, and correction hashes are inputs. The local checkpoint bytes
must match the registered SHA-256 value.

## Capability

For each answer-changing family, split, and ordered closer pair, both native prompts must rank their registered closer
above the other three closer tokens on at least 75% of rows. The mean symmetric margin—correct closer logit minus the
mean of the other three—must be positive, with a group-bootstrap 95% lower bound above zero when pooled across the 12
pairs.

For each answer-preserving family, split, and endpoint, the registered closer must rank above the other three on at
least 75% of rows. Failure stops interpretation of site results and forbids any subspace fit.

## Complete-state interchanges

At each site, save the complete donor state and replace the base state, then perform the reverse replacement. Variable
sequence lengths use their own final-token indices; padding positions are never treated as the source state.

Candidate sites, in frozen order:

1. residual stream entering block 8 at the final position, shape 1,152;
2. L13H8's complete 128-dimensional pre-output-projection vector at the final position.

For target rows, the effect is the donor-correct versus base-correct logit margin after the swap minus before it. A
site's target ceiling passes only if, for both families and both directions:

- pooled mean movement is positive;
- 95% group-bootstrap lower bound on the mean is positive;
- at least 70% of rows move donorward; and
- every ordered closer-pair cell has positive mean movement and at least 50% positive rows.

For answer-preserving rows, define the endpoint margin as the registered closer logit minus the mean of the other three
closer logits. A negative-control family is causally live at a site only if both directions have bootstrap-lower mean
absolute endpoint change above 0.03 logits and mean full-vocabulary logit RMS above 0.01. This ensures a future quiet
projector is selective rather than merely acting at a dead site.

A candidate site is authorized for a learned subspace only if capability passes, both target families pass, and all
three answer-preserving families have live complete-state effects there. If both sites pass, select the earlier
residual site without looking at effect size. Save row-level effects for an independent CPU audit.

## Price and interpretation

There are 720 FIT/SELECT pairs. With batches of eight, one native capture and four patched forwards per batch cost
exactly 450 model forwards, zero backwards, and zero model-weight updates. This can establish only a live causal site;
it cannot identify a low-dimensional circuit. A pass licenses a contrastive ordinary-versus-readout-deflated DAS
preregistration. A failure redirects the circuit/site definition without a rank sweep.
