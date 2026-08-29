# Hourly strategic review — 2026-08-29 00:35 UTC

## Bottom line

The whole-model explanation ledger has not moved this hour. The useful progress is
that the most important ambiguity in the new context result now has a complete,
tested, source-closed experimental implementation. It can be answered with 16 new
role-cells as soon as the independent audit passes and the GPU is free.

The question is precise. Existing measurements include:

- `E`: no late suffix replacement;
- `A`: replace attention outputs in layers 3 through 8;
- `AM`: replace both attention and MLP outputs in layers 3 through 8.

They omitted `M`: replace only MLP outputs in layers 3 through 8. Without `M`, the
apparently simple broad-suffix behavior can conflate two different mechanisms. The
new experiment measures exactly the eight early-prefix masks crossed with `M`, on
both independent document roles. It introduces no fitted correction.

## How much of the model is actually explained?

These are different currencies and must not be added.

| Currency | Current result | What it does **not** establish |
|---|---:|---|
| Top-level modules with an executable structural surrogate | 36/36 | meaning, minimality, causal equivalence, or faithful composition |
| Whole-program storage certified removable for its registered consequence | 5.3481% | that 5.35% of behavior or semantics is understood |
| Exact context-free class ceiling on covered token types | attained at full rank | the roughly 2.74 CE-nat context gap to the live model |
| Strict named causal CE headroom recovered | 10.923% | the remaining 4.72714 nats / 89.077% |
| Final extraction/removal/OOD actions evaluated | 0/68 | any final-role semantic or manipulation credit |

Thus the honest answer remains: structural inventory coverage is complete, but a
small fraction is consequence-certified and most causal behavior remains unnamed.

## What changed since the previous review

### 1. The retrospective sparse result survived, but its interpretation narrowed

The closed 8-by-8 early-prefix/context grid remains highly reproducible across the
two document populations: the 49 non-anchor CE interaction cells correlate 0.9963.
A 16-term sparse Möbius description predicts omitted cells better than both fewer
and more terms, and transfers between document roles at normalized error about 0.19.
That is real predictive simplicity on already measured intervention cells.

However, because `M` was absent, one coefficient family aliases an early-prefix by
MLP interaction with a three-way early-prefix by attention by MLP interaction. It is
therefore a descriptive grammar, not yet a causal interface or compressed program.

### 2. A full prospective de-alias implementation now exists

Implemented and tested:

- the exact eight-mask `M` registry;
- per-document sufficient-statistic aggregation;
- a physical backend that counts all 36 native attention/MLP module calls and the
  exact requested substitutions;
- receipt-last two-role execution with pre-outcome authority;
- exact replay of the protected old `E/A/AM` measurement;
- old/new joins on ordered document identity, row identity, row-to-document map,
  common support, token denominators, model, and shared program;
- both 2,000-document bootstraps and both directed conditional cross-role tests;
- explicit claim boundaries forbidding OOD, semantic, compression, or global-ledger
  credit from this CE-only assay.

The focused CPU suite passes 32/32 tests. The protected parent artifacts and both row
populations replay exactly. Independent review caught and we corrected two pre-outcome
contract defects: the descriptive raw-synergy norm now excludes the baseline cell as
preregistered, and the scorer rechecks immutable inputs immediately after bootstrapping
and before publication. The final independent pre-execution verdict is pending.

### 3. No GPU launch occurred

Claude's independent second-class dominance experiment currently occupies the GPU.
The de-alias run is also correctly gated on a committed, pushed, clean source closure
and independent audit. CPU implementation and verification were therefore the
highest-priority safe action this interval.

## The exact mathematical test

For early-prefix mask (P_i), let (C(P_iS)) be token-weighted cross-entropy when
suffix (S\in\{E,A,M,AM\}) is replaced. Define the early-prefix interaction with a
suffix by

$$
D_i^S = C(P_iS)-C(P_i)-C(S)+C(E).
$$

The prediction frozen before observing `M` is

$$
\widehat{D_i^M}=D_i^{AM}-D_i^A.
$$

The prediction error is exactly

$$
Q_i = D_i^{AM}-D_i^A-D_i^M.
$$

This (Q_i) is the three-way early-prefix by attention by MLP contrast. If the
prediction works on both roles and transfers with a fixed source-role prediction,
then attention approximately does not change the way the early prefix interacts
with the broad MLP suffix. If it fails, the attractive broad-suffix simplicity was
another observational alias and should be pruned.

The experiment separately reports the raw attention/MLP synergy

$$
R_i=C(P_iAM)-C(P_iA)-C(P_iM)+C(P_i)
$$

and the standalone `M` marginal (C(M)-C(E)). Neither is confused with (Q_i).

## Largest remaining gaps and confusing observations

1. **Context remains the main class boundary.** The best context-free compiler is
   still roughly 2.74 CE nats worse than the live model. Local MLP0 compression alone
   cannot close this.
2. **Composition is strongly nonadditive.** Frozen rank-3/rank-4 dense cross models
   failed, even though the broad attention-plus-MLP suffix was extraordinarily easy
   to predict. The missing `M` corner is the immediate ambiguity.
3. **Scalar CE may hide the mechanism.** Even a successful 16-cell result would only
   identify a stable scalar interaction law. It would not identify which residual or
   logit directions carry it.
4. **The final causal interface is unopened.** The 68 extraction/removal/OOD actions
   remain 0/68 because the semantic reducer, comparator arms, uncertainty contract,
   and all receipt gates are not yet frozen together.
5. **Simplicity depends on consequence.** Sparse-term count helped omitted-cell
   prediction; storage cost helped construct Pareto curves; rank helped context-free
   CE. They are not interchangeable. Claude's latest frontier audit also withdrew an
   overgeneralized top-1 claim: the final-rung differences were only 4–24 correct
   tokens out of 36,800, too small for that instrument to resolve. CE conclusions
   remain, but metric-specific claims must stay metric-specific.

## Pruned and ranked next actions

Ranking criteria are expected information gain, causal relevance, whole-model
composability, falsifiability, GPU cost, and nonduplication.

1. **Execute and score the 16-cell missing-`M` assay.** It directly resolves the
   largest ambiguity in the strongest recent context result. It is prospective,
   exact, cheap relative to the 68-action suite, and can decisively prune or preserve
   the proposed broad interface. The implementation is complete; remaining gates are
   independent audit, clean pushed source closure, and a free GPU.
2. **Freeze the 68-action semantic reducer and comparator contract.** This is the
   shortest route from “low reconstruction error” to useful extraction, selective
   removal, OOD transport, and editing. It has higher causal relevance than another
   local fit, but its current 0/68 contract is not executable because objective and
   transport gates, gauge replays, uncertainty semantics, and comparator arms are not
   jointly frozen.
3. **If the scalar de-alias test passes, collect a vector-valued response basis on the
   same masks.** Residual/logit response vectors can locate the carried subspace and
   support composition or edits. This is conditional: it is redundant expense if the
   scalar interface already fails.
4. **Prospectively test the sparse hierarchy at an adjacent cut.** Freeze terms from
   this cut and predict a neighboring suffix boundary without refitting. Success
   would turn a same-grid grammar into evidence for a reusable program interface;
   failure cheaply prunes it.
5. **Fit a joint downstream-weighted MLP0/MLP1/MLP2 dictionary and evaluate at matched
   consequence.** Shared atoms, sparse coefficients, and residual directions should
   be optimized jointly across producer and consumers, then compared at matched CE,
   extraction, and removal quality. A weight SAE or HOSVD alone stays lower priority
   because it can be simple in coordinates while being useless downstream.

Not promoted: increasing the failed dense-cross rank, fitting corrections on seen
cells, another context-free table variant, or claiming coefficient-square “energy”
in the nonorthogonal Möbius basis.

## Action executed this interval

The complete source-closed CPU implementation of priority 1 was built and tested.
This is not a status-only review. It reduced the next GPU action from an informal
idea to a falsifiable 16-cell transaction with exact provenance, physical call
census, protected old/new joins, deterministic uncertainty, and receipt-last scoring.
No model outcome has been opened yet.
