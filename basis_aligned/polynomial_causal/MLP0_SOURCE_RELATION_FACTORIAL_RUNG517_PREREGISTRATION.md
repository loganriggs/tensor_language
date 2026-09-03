# Rung517 preregistration: cross-head source-relation factorial for MLP0

Registered 2026-09-03 01:43 UTC before any rung517 model outcome.

## Question and non-duplicate check

MLP0's exact output has already been split into token-only `T`, context-only `C`, token-by-context `I`, normalization
effect `S`, and a retained numerical remainder. Generic sparse codes, token and bigram lookup tables, reader-weighted
low-rank bases, head-level duplicate-service tests, and low-rank quadratic/Tucker refactorizations have all failed or
closed. Rung402 found one dominant attention head plus a distributed tail; rung417 found no duplicate head service.
Rungs480--496 followed MLP0 into attention1/MLP1 and rejected a global62-circuit label, token/bigram effect tables,
Jacobian substitution, and several finite downstream equivalences.

This rung changes the basis and question. It groups attention0 contributions by the relation between query and source
positions, summing across every head before MLP0. It asks which exact source relations supply MLP0's continuous
context computation and whether the answer changes between prose and structured text. It is not a rank or
compression experiment.

## Exact object

For query position `q`, source position `s <= q`, and head `h`, attention0 contributes

`a[h,q,s] = O_h( score1[h,q,s] * score2[h,q,s] * value[h,s] )`.

The score and value tensors are those used by the deployed model, including rotary position encoding, causal masking,
and deployed data types. Sum these contributions across heads and assign each source position to exactly one group:

1. `SELF`: `s=q`;
2. `PREVIOUS`: `s=q-1`;
3. `NEAR`: `2 <= q-s <= 7`;
4. `DISTANT_SAME`: `q-s >= 8` and `token[s]=token[q]`;
5. `DISTANT_OTHER`: every remaining causal source.

The five group writes plus one explicit arithmetic remainder must sum to the deployed attention0 write. The remainder
is always retained and is never assigned semantic meaning.

For subset `U` of the five groups, form `a_U` from exactly those group writes plus the arithmetic remainder. Recompute
only MLP0's deployed input normalization and bilinear output from the native token term plus `a_U`. Attention0's
ordinary residual-stream write remains native, so the intervention changes only what MLP0 computes from context.
Run the rest of the model normally. There are exactly32 subsets.

The full subset must reproduce native MLP0. The empty semantic subset is the token-only MLP0 boundary with only the
arithmetic remainder retained. Boolean-lattice inclusion--exclusion gives all31 nonempty finite interaction effects;
five-variable Shapley values allocate each subset effect across source groups without choosing an ordering.

Separately, using the frozen mean gain and reference means from the exact MLP0 `T/C/I/S` decomposition, report the
vector identities

`I = sum_g I_g + I_epsilon`

and

`C = sum_g C_gg + sum_{g<k} C_gk + C_epsilon`,

where `I_g` is the symmetric token-by-group bilinear term, `C_gg` is the within-group quadratic term, and `C_gk` is
the symmetric cross-group context term. These are algebraic accounting identities, not causal importance scores.

## Data and separation

- Reuse the frozen prose FIT and SELECT roles from rungs401--402:96 documents each, scoring positions64:256.
- Use64 FineWeb structured/diverse documents split before outcome into FIT32 and SELECT32, with the same scoring
  positions. Hash the exact token rows in the result.
- FINAL rows from the prose role remain unopened.
- Reference means and any scale used for summaries are fit separately on each corpus FIT role and applied unchanged
  to its SELECT role.

## Measurements

For every corpus, split, subset, and document, retain the per-position change in cross-entropy loss relative to the
full subset. Also retain MLP0 output changes and immediate attention1/MLP1 output changes so that a stable source
role is not inferred from average loss alone. Split comparisons never align different documents by row number.
Instead, within each split average over documents first and retain vectors indexed by the same192 absolute token
positions (`64:256`). For CE this is the signed mean loss change at each position. For a consumer write this is the
root-mean-square change over documents and its1,152 coordinates at each position. These192 coordinates have the same
meaning in FIT and SELECT and can be compared legitimately.

For each group report:

- Shapley cross-entropy contribution in nats;
- singleton sufficiency: improvement from empty to that group alone;
- leave-one-out necessity: damage from full to full-minus-that-group;
- its share of the sum of positive endpoint-average benefits;
- cosine and fitted-scale residual between FIT and SELECT192-position effect profiles;
- its shares of exact `I`, within-group `C`, and cross-group `C` vector energy.

## Frozen predictions

### A — exact, live instrument

All source positions belong to exactly one group; the five group writes plus remainder reconstruct deployed attention0
with relative squared error at most`1e-8`; full-subset MLP0, logits, and CE match native within the existing rung401
BF16-derived limits; empty-subset replay is deterministic; all32 arms are live; Möbius closure is at most`1e-10` in
float64; call counts and row hashes match; and eight planted five-factor tables recover every planted interaction and
Shapley value to`1e-10`.

### B — prose localization

On prose SELECT, `PREVIOUS` is the largest positive endpoint-average group and `SELF+PREVIOUS` supply at least70% of
the sum of positive endpoint-average benefits. Both groups' necessity and singleton effects must be positive. This is
the source-resolved version of the old prose bigram result.

### C — structured-text widening

On structured SELECT, the positive endpoint-average share of `SELF+PREVIOUS` is at least10 percentage points lower
than on prose, while `SELF+PREVIOUS+NEAR` supplies at least70%. This tests the old finding that structured text needs
about8--16 recent tokens rather than assuming that law transfers to the MLP0-only source decomposition.

### D — split-stable source roles

For both corpora, FIT-to-SELECT Spearman correlation of the five endpoint-average benefits is at least`.70`, the top
group is unchanged, and every promoted group's fitted-scale192-position signed-CE profile has cosine at least`.70`
and relative residual at most`.65`. `DISTANT_SAME` is promoted only if it is material on both splits; no rare-group
claim is made from a large per-occurrence value alone.

### E — downstream specificity screen

At least one promoted group has an immediate attention1 or MLP1 192-position RMS-effect profile whose FIT-to-SELECT
cosine exceeds every one of eight source-position permutation controls by`.15`, and one consumer receives at least
1.5 times its total effect RMS at the other consumer. This is only a route to a later physical transfer test, not a
circuit claim.

## Controls and interpretation rules

- Eight fixed source-position permutations preserve each query's group sizes while destroying relation identity.
- A same-count random distant group controls `DISTANT_SAME` when enough repeated-token positions exist.
- Report group support and effect per occurrence; do not let rare support create an unqualified semantic headline.
- Shapley allocation is an averaging convention over all subset orders. A large Shapley value is not itself a
  circuit, source sufficiency, or interchangeability result.
- The algebraic `I/C` energy report cannot substitute for finite causal effects.
- No rank, singular-value cutoff, sparse-code width, SAE, quantization, or reconstruction metric may license a claim.

## Decision rule

- If A and B--D hold, retain the exact relation grammar and expand only the dominant relation by token semantics or
  downstream physical interchange.
- If A and C--D hold but B fails, retain a register-dependent local relation grammar and do not claim prose
  universality.
- If A holds but D fails, conclude that these source relations are not stably identified and leave this basis.
- E can license a separately preregistered consumer-specific physical test only; it cannot license a circuit here.
- If A fails, repair only the instrument without interpreting model outcomes.

Maximum planned work is32 subset arms times the frozen batches for both corpora, plus native and immediate-consumer
replays. No model training and no backward pass. Zero deployed parameters are added or saved.

## Pre-outcome stability-coordinate correction — 2026-09-03 01:54 UTC

The first registration said “per-document effect vectors” in D/E. FIT and SELECT contain different documents, so
their row coordinates are unrelated and a direct cosine would be meaningless—the exact error class already exposed
by the rung514 task-space companion. Before implementing or observing any model outcome, the shared comparison object
is corrected to the192 absolute-position profiles defined above. No source group, row, arm, threshold, prediction
direction, or outcome-dependent choice changed.
