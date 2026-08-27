# MLP0 Stage-0 causal quotient: result and revised interpretation

## Verdict

The reader-defined K=64 token partition is not a downstream-equivalence quotient,
even on the registered finite grid. More basically, the static token-mean table is
not an adequate interface for live MLP0. The evidence therefore does not license the
story “MLP0 maps each token to a discrete class that downstream computation reads.”

This does **not** say lexical classes are absent. It says class similarity is a
partial, lossy coordinate of MLP0, not a state-complete replacement interface.

The authoritative result is
`bilinear_quotient/mlp0_quotient_stage0_v2_results.json` (SHA256
`35215d561ac12acd2bc1e11138c48c5566701bb5d0b8ea7b8019241ec9b86759`).
All four reports were independently reproduced exactly from the serialized
document-by-cell sufficient statistics.

## Registered gates

| Gate | Result |
|---|---:|
| covered evaluation positions | 92.6819% — pass |
| token table T equivalent to live MLP0 O | fail |
| reader Q64 equivalent to token table T | fail |
| reader Q64 pointwise dominates activation A64 | fail |
| global mean control sensitive for all consumers | pass |
| Stage 0 overall | fail |

The failures are not near a threshold. For T versus live MLP0, the simultaneous 95%
upper bound on the maximum standardized effect is 25.760; passing requires less than
1. Its worst cell is first-half, low-frequency, non-punctuation predecessor, high raw
pre-MLP0 residual norm: KL is 0.2130 against a 0.01 margin and signed CE harm is
0.1708 against a 0.0075 margin. Worst block-1 attention and MLP output nRMSE are
0.4021 and 0.6089 against 0.05 margins.

For Q64 versus the token table, the simultaneous upper bound is 59.688. The worst CE
cell is second-half, high-frequency, non-punctuation predecessor, high residual norm:
0.4182 nat/token. A64 is also far outside equivalence, but has a smaller maximum
standardized effect (48.657 versus Q64's 55.756), so the registered reader metric does
not win its matched allocated-price comparison.

Only 25 Q64 and 31 A64 clusters have positive fit mass. Both were conservatively
charged at nominal K=64 for the preregistered comparison. A future minimal-price
frontier must charge occupied states plus the producer needed to assign them; this
cannot rescue either candidate's large distortion.

## Resolving “clusters” versus “everything is separated”

These statements concern different equivalence notions:

1. Literal block-1 reader matrices have full joint rank. Distinct MLP0 writes are
   exactly distinguishable at those declared interfaces; the exact-kernel quotient
   has singleton states.
2. Geometric and semantic experiments find nearby token/class directions and partial
   hard-cluster recovery. Coarse lexical organization exists approximately.
3. The new causal screen asks whether merging those approximate states preserves all
   registered downstream responses in every background. It does not.

Full rank does not make compression useless: exact distinguishability asks whether a
difference is mathematically zero, while simplicity asks how cheaply behavior can be
represented within a declared distortion budget. The correct object is a causal
rate-distortion frontier, not either raw rank or an unvalidated class count.

## Revised MLP0 model class

The evidence favors a hierarchical continuous interface:

```text
MLP0(t, s) = shared mean
           + coarse lexical coordinate(t)
           + within-class/token-specific coordinate(t)
           + low-rank contextual correction(t, s)
           + residual.
```

The terms are not assumed independent or uniquely identifiable. A gauge must be
fixed by fit-row centering/whitening, and producer plus consumer wiring must be priced
jointly. “Class” earns explanatory status only if its shared atom reduces total
description length at the same causal fidelity.

The next discriminator is a matched-price hierarchical screen:

- `T + P_r(m0_live-T)` for context ranks r=16/32/64;
- `Q64 + U_k(token) + P_r(context)` with a priced within-cluster continuous code;
- a continuous response-metric PCA control at matched total bits/rank;
- a native polynomial/program generator control rather than a 50,257-row lookup.

Every arm must be scored against attention-1, MLP1, final KL/CE, the same worst-cell
gate, and then the exact MLP0/1/2 composition cube. Since T versus O failed, the
registered within-class donor-swap Stage 1 is not licensed: a class swap would
confound token code with missing contextual state.

## What this changes globally

It changes the qualitative MLP0 interpretation and prunes hard clustering as the
next route. It does not add whole-model explained-behavior credit. Inventory remains
36/36, named behavior 32.1%, named causal paths 10.92%, legacy composition 12.4%, and
executable recovery zero. The clean-to-ship gap remains +0.8976 nat/token. MLP0's
static and coarse interfaces are measured failures that constrain the next model
class, not recovered pieces of the whole-model numerator.
