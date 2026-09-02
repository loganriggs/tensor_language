# Rung 474 preregistration — subtractive query-position factorial

Registered after rung 473 and before running any subtractive multi-MLP outcome.

## Why this control is necessary

Rung 473 removed a later MLP by replacing its current product with an equality-absent product captured while all
earlier MLPs were intact. If an earlier MLP is removed in the same arm, the later MLP's current input has changed.
The fixed replacement therefore changes both the equality contribution and the later MLP's recomputed background
state. That is a valid intervention, but it can make pair/triple interactions depend on the replacement frame.

This rung uses an equally explicit alternative. On the intact source state, capture each query MLP's product change

`delta_j = product_j(source) - product_j(equality_absent)`.

During a multi-MLP arm, subtract this frozen `delta_j` from the MLP's *current recomputed product* at the query
position. Earlier removals may change that current product, but do not change the frozen component being removed.
For a one-MLP arm, current equals intact source, so the subtractive arm must reproduce rung 472's replacement arm.

This tests stability under two plausible causal coordinates. It does not declare one coordinate semantically correct
in advance, reduce rank, fit a decomposition, or claim storage savings.

## Exact computation

Use exactly rung 472/473's targets, windows, sources, absent states, query masks, and context cells. For every target
run all seven nonempty subsets of MLP8/9/12 under the subtractive rule above. Compute the same main, three pair, and
triple Möbius terms. Compare them with the fixed-replacement factorial without changing either receipt.

## Frozen predictions

### A — valid alternate intervention

- all source/preregistration/parent hashes match;
- native replay relative squared error is at most `1e-12` and product reconstruction error at most `1e-10`;
- an empty mask changes no logit;
- every requested subtraction fires exactly once;
- each subtractive singleton reproduces rung 472's corresponding singleton target effect within `1e-6` nat;
- Möbius closure is at most `1e-12` in float64 analysis;
- forwards and subtraction calls equal the formulas printed before model load;
- SEALED attention-0 confirmation remains unopened.

### B — the identified query circuit is stable to causal coordinates

The subtractive and fixed-replacement all-three query effects have per-token Pearson at least `.80`, four-context
cosine at least `.80`, and projection in `[.50, 1.50]` in every window and source.

### C — fixed-baseline state mixing materially caused the natural interaction instability

Under subtraction, both natural windows either reduce total interaction norm by at least 25% under both sources, or
improve N/H total-interaction cosine by at least `.30`; code N/H interaction cosine remains at least `.80`.

### D — register-conditioned composition persists under the alternate coordinate

Code remains MLP8+MLP9-led under both sources, while at least three of four natural window/source conditions remain
MLP8+MLP12-led, or at least one natural N/H total-interaction cosine remains at most `.20` with both interaction norms
at least `.003` nat.

### E — natural-hybrid document-half fragility was coordinate-specific

Both natural-H largest-pair interactions now have the same signed mean in their two fixed document halves and each
half norm is at least 20% of the pooled norm.

## Strong null and routing

The strong null fires if A fails, if any subtractive/fixed all-three per-token Pearson is below `.50`, or if every
subtractive pair and triple four-context norm is below `.003` nat. A+B+C would attribute much of rung 473's higher-
order instability to baseline-state mixing. A+B+D would establish register-conditioned composition that survives
the intervention coordinate. Failure of B means the query circuit is identified only relative to a specified removal
coordinate and must not yet be treated as one portable executable variable.

## Price

Diagnostic only: zero deployed parameters saved or added. Report model forwards, subtraction calls, runtime, and peak
GPU memory. Execute only through the managed runner.
