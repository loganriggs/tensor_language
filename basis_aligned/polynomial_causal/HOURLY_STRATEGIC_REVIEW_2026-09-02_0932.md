# Hourly strategic review — 2026-09-02 09:32 UTC

## Current state

No new scientific receipt landed during this hour because rung480 is deliberately collecting individual-target
gradients. The repaired managed process has run continuously since08:26 UTC, remains at roughly93--94% GPU
utilization with stable memory, and has emitted no new error. A path-specific reproducibility diagnostic is queued
behind it, so the runner will not fall idle at process exit.

Rung480 remains the correct active direction. It asks whether downstream circuit effects identify stable,
gauge-covariant score-branch or payload directions inside the already replicated attention0 continuous interface.
The experiment does not claim that a low rank is interpretable and does not select a new rank.

## Binding definition of success

The program wants circuit descriptions that provide:

1. a concise computational explanation;
2. grouping across heads/modules when downstream computation treats pieces alike;
3. splitting within a head/MLP when pieces have different downstream roles;
4. held-out and out-of-distribution prediction;
5. extraction as an executable subcomputation;
6. selective removal/editing without damaging unrelated circuits; and
7. reusable composition in larger circuits.

Rank reduction, quantization, local reconstruction, or fewer stored values alone does not satisfy these goals.
Rung480 can only become a circuit result after a stable response-defined projector passes exact held-out removal.

## Efficiency reflection caused by the reminder

The current implementation performs one backward for every `(circuit mask, target position)` occurrence. This is
scientifically exact, but positions shared by multiple circuit masks repeat the same derivative. A future collector
can preserve the exact computation while reducing work:

1. within each document batch, build the map from a unique `(document, query position)` to all circuit/mask labels
   containing it;
2. compute that target's CE gradient with respect to the fitted attention0 query write once;
3. compute its affine-triplet audit and the three `sym(z outer grad_z CE)` operators once; and
4. add those same values to every attached label's sufficient statistics.

This is an exact common-subexpression elimination, not grouped-loss differentiation: it never sums losses from two
positions, so later-token-to-earlier-query cross-terms remain excluded. It should be the default for future
response collectors. It is not being patched into the active process because that process has already begun the
frozen collection and no implementation or execution-count rule should change mid-run.

## Result-conditioned route remains unchanged

- **Full480 pass:** exact projector-defined removal on the reserved30 odd-root circuits and documents500:1000,
  measuring named-circuit effect, unrelated-circuit collateral, CE, and later recomputation.
- **Scientific480 strong null:** retain424/425 as a continuous predictive interface and start the exact MLP0
  `T/C/I/S` downstream-effect decomposition against the62 circuit tags. Preserve exact token-private computation;
  accept shared token or context components only when effect profiles transfer and exact removal is selective.
- **A fails only on old424/425 cross-session number checks:** preserve480 as failed as written, then register an
  in-session bridge repair using identical response statistics and recorded deterministic backend state. Do not move
  a scientific threshold.
- **Scoring audit:** any proposed winning slab must itself belong to the modes that passed the response eigengap,
  aligned-refit, response-half, and activation-control gate. Otherwise use an analysis-only child on the same frozen
  sufficient statistics; do not promote the parent scorer's candidate.

## Scheduled reconsideration

The next hourly review is due after10:32 UTC if the active chain still runs. The next three-hour mathematical review
is due after10:13 UTC and should specifically ask whether tensor-network identifiability, operator algebras, or
multilinear invariant theory provides a more exact route than the current response projectors. The overall goal
remains active until a scored receipt and a genuinely started successor exist.
