# Rung514 preregistration: constrained multi-term programs inside attention11 and MLP11

**Frozen:** 2026-09-03 00:29 UTC, after rung513 scoring and before any rung514 model outcome.

## Decision and circuit target

Rung513 exactly split the first consumers of18 source-equivalent MLP10 branch changes into31 attention11
`Q/K/Q2/K2/value` interactions and3 MLP11 Left/Right/joint interactions. Every one of612 relation-by-term responses
was material, but no singleton term preserved even one relation under the fixed two-half cosine and residual rule.
The median sum of absolute attention mismatch shares was2.47 for a signed total of1, so cancellation among terms is
part of the observed computation.

The registered B-false route calls for multi-term combinations with a planted identifiability test. Rung514 asks
whether the downstream variable is either an architecture-fixed allocation of higher-order interactions or a small
signed program over the exact terms. The circuit targets are within-module splitting, cross-action grouping,
held-out prediction, selective physical manipulation, and reuse across MLP10 branch subsets. This is not a rank,
variance, reconstruction, or parameter-compression experiment.

## Exact response statistics

The branch interventions, six subsets, four score actions, factor definitions, attention corners, MLP corners, and
18 relations remain exactly those of rung513. The collector additionally stores, separately for each branch subset
and document window, the complete joint Gram matrix between every action-by-term response. A Gram entry is the dot
product between two flattened response tensors on the fixed positive-copy positions. It is sufficient to compute
the norm, cosine, fitted scalar, and residual of any registered linear combination without storing token activations.

Discovery documents are divided before execution into four windows:

- A-fit:500:560; A-test:560:624;
- B-fit:624:684; B-test:684:748.

The unequal60/64 sizes keep all boundaries aligned to the four-document model batch. Confirmation remains
documents752:1000, split752:876 and876:1000. The30 confirmation circuit families remain unopened until a discovery
candidate exists.

## Object class 1: fixed factor allocations

For every attention interaction `T_S` indexed by a nonempty subset `S` of the five factors, define the exact Shapley
allocation to factor `i` as

`phi_i = sum over S containing i of T_S / |S|`.

Every higher-order term is divided equally among the factors that participate in it, so the five `phi_i` sum exactly
to the complete attention response. This gives five fixed objects: Q, K, Q2, K2, and value. For MLP11 the analogous
two objects are `phi_L = L + joint/2` and `phi_R = R + joint/2`; they sum to the complete MLP response. Across six
branch subsets this is42 fixed groups. The allocation is a declared accounting rule, not a claim that Shapley value
is the unique semantics.

## Object class 2: sparse signed exact-term programs

Within attention11, enumerate every sum of exactly two or three distinct Möbius terms with coefficients in
`{-1,+1}`. Overall sign is redundant with the fitted cross-action scalar, so the first included coefficient is fixed
to`+1`. This gives

`2*C(31,2) + 4*C(31,3) = 18,910`

attention programs. Apply the same rule to MLP11's three terms, giving10 MLP programs. There are18,920 sparse
programs per branch subset and113,520 across six subsets. Support size is literal program complexity. It is not
called a compressed circuit unless the causal gates below pass.

For every program and each of the three source relations, fit one signed scalar on A-fit and test it without refitting
on A-test. Repeat the entire search independently with a scalar fitted on B-fit and tested on B-test. A program is
split-stable only if the exact canonical term support and signs pass in both searches. Each test requires response RMS
at least10% of the complete consumer response at both endpoints,`.25 <= |beta| <= 4`, cosine at least`.85`, and both
directional relative residuals at most`.55` for all three relations.

No best-k truncation is allowed. If more than32 distinct fixed-or-sparse groups pass all discovery controls, the
representation is declared non-identifiable at this grain and confirmation does not open. Otherwise every passer is
retained.

## Multiplicity and planted-identifiability controls

Sixteen fixed controls use seeds51410:51426. Within each batch and branch subset, independently permute the flattened
response coordinates for P, Z7, and Z8 while applying the same permutation to all terms of one action. This preserves
each action's complete internal term algebra and term magnitudes but destroys coordinatewise relations between
actions. The full fixed and sparse banks are searched under every control.

For a relation on a test window define

`margin = min(cosine - .85, .55 - maximum directional residual)`.

A group's score is the minimum margin across all three relations and both independent A/B searches. In addition to
the absolute gates above, a real group must exceed the largest familywide control score by at least`.02`. This fixed
comparison protects against finding an apparently good signed sum merely because113,562 groups were searched.

Before model outcomes count, the same exhaustive search must recover the exact planted support and signs, after
canonical overall-sign normalization, as the unique accepted program in each of eight fixed synthetic problems with
seeds51400:51408. Supports alternate between size2 and3 and contain a known shared signal plus term-specific nuisance.
Failure on any seed makes A false and routes only to instrument repair; it cannot be repaired by looking at model
results.

## Fresh-document prediction

For every discovery group, refit only its three cross-action scalars on the pooled discovery documents500:748, while
keeping its exact support, signs, and object class fixed. Test those scalars on untouched documents752:1000. All three
relations must be material and must achieve cosine at least`.75` and both directional residuals at most`.65` in each
confirmation half. No support change or coefficient refit is allowed.

## Physical removal and substitution

At most32 confirmed groups proceed. For a group response `t_x` in target action x, donor response`t_y`, and frozen
scale beta, run both:

- removal: `consumer_x -> consumer_x - t_x`;
- substitution: `consumer_x -> consumer_x - t_x + beta*t_y`.

Run four reusable removals per group and all three relations in both directions, for ten consumer patches per group.
Attention programs patch the complete attention11 residual write; MLP programs patch the MLP11 residual write. For
every direction and both confirmation halves, removal must have copy-task norm at least`.00025` nat and held-out-
circuit RMS at least`.0005` nat; substitution damage must be at most`.50` of removal damage for both the four copy
cells and30 held-out circuit effects; absolute off-target CE damage must be at most`.002` nat.

## Registered predictions and routes

- **A — exact live and identifiable instrument:** all hashes, calibrations, source relations, branch edits, corner
  replays, closures, joint-Gram symmetries, window counts, permutation controls, call counts, and any physical patches
  pass; all eight planted problems recover exactly one correct signed support.
- **B — constrained multi-term discovery:** between1 and32 fixed-or-sparse groups pass both independent searches,
  all absolute gates, and the familywide permutation margin.
- **C — held-out identification:** at least one B group passes all three relations in both untouched confirmation
  halves with support/signs fixed.
- **D — causal interchange:** at least one C group passes all six bidirectional physical substitutions and their
  matched removal controls.
- **E — reuse:** the same fixed factor name or exact canonical sparse support passes D for at least two different
  MLP10 branch subsets.

`strong_null = not (A and B and C and D)`. E is the stricter reuse result.

Frozen routes:

- A false: repair only the named statistics, permutation, planted-recovery, or patch instrument.
- B false: neither fixed factor allocations nor two/three-term signed programs identify the shared variable; move to
  a prospectively task-conditioned nonlinear downstream reader of the exact terms, with the62 circuits used as
  held-out outcomes rather than returning to rank or widening this support search.
- more than32 passers: treat the representation as non-identifiable and strengthen observations, not selection.
- C false: retain only a discovery screen and diagnose which fixed response relation changes across documents.
- D false: the combination is not operationally interchangeable; split at the first downstream reader that rejects
  its substitution.
- D true, E false: validate the branch-specific multi-term circuit on fixed out-of-distribution code.
- D and E true: validate the reusable program jointly across branches and on out-of-distribution code.

No outcome permits support size above3, continuous coefficient fitting inside a program, lower thresholds, rank or
SAE sweeps, selecting a best few after the32-group stop, or calling Shapley attribution a circuit without D.

## Literal price

The bank contains42 fixed groups and113,520 sparse groups. Search and16 controls use only joint Gram arithmetic.
Discovery costs2,108 full forwards,47,616 local attention corners,5,952 local MLP corners, and0 backwards.
Confirmation, if B holds, costs the same again.

If q groups confirm, physical execution costs`1,798 + 620q` full forwards. Because the identification gate stops when
q exceeds32, the total maximum is`4,216 + 1,798 + 620*32 = 25,854` full forwards,0 backwards, at most96 fitted
cross-action scalars,0 deployed parameters added, and0 parameters saved. The physical factor reconstruction adds at
most47,616 attention corners and5,952 MLP corners. These are research costs, not claimed deployment savings.

## Frozen evidence

- rung513 result SHA256: `043dd563baa5ffe5bda57c7774dc76e4727add3b4351699cb997ecfd563179d5`;
- rung513 bundle SHA256: `06118d18594c4b167a3f3d46a2aa282969f6b061835f83a3b3d62b5ca72b8d8a`;
- rung513 source SHA256: `dda9c2636a99f76a2298e5cebccea1b1e8bd503c415f073ca93c984e8713fc98`;
- rung513 preregistration SHA256: `b895d1aefdac4c7deee0477c260a5e1ec087477925e841d0d2b8ebb4a02670aa`;
- checkpoint weights SHA256: `680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3`.
