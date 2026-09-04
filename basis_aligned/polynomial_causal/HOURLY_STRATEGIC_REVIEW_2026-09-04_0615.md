# Hourly strategic review — 2026-09-04 06:15 UTC

## Circuit interpretation targets

A useful decomposition must eventually satisfy all seven targets below.  Native attention-head and MLP boundaries are
candidate implementation boundaries, not assumed semantic variables.

1. **Computational specification:** say what information is read, what operation is performed, what is written, and
   which later computations use it.
2. **Grouping and splitting across module boundaries:** merge pieces of different heads/MLPs when later computation
   treats them as one variable, and split one module when its pieces serve different operations.
3. **Held-out and OOD prediction:** predict activations and behavioral effects on unseen prompts, task variants, and
   shifted data.
4. **Extraction or sufficiency:** an executable circuit, or a precise interface plus declared background computation,
   reproduces the target computation or signed causal effect.
5. **Selective manipulation:** removals, swaps, or edits change the intended behavior while preserving unrelated
   behaviors, with redundancy and interactions measured rather than assumed away.
6. **Composition and reuse:** shared computations work across tasks/modules, and their joint behavior is predictable
   when installed with task-specific branches.
7. **Stable identification:** the units survive data splits, reasonable gauge changes, and fitting restarts, or are
   defined operationally by indistinguishability to later readers.

The program goal remains a smaller executable tensor program that is jointly predictive on fresh/OOD inputs,
composable, selectively manipulable, and simpler under literal storage, compute, edge, state, and program costs.
Compression, variance explained, low rank, or low CE damage alone does not establish a circuit.

## What changed since 05:15

### The first strict behavior correctly failed

Task 17's prospective positional-list FIT assay ran through the reviewed hash-bound queue.  The model achieved only
43.75% base and 40.63% donor forced-choice accuracy; the worst cell was 29.17%, and both mean answer margins were
negative.  This is a valid scientific `hard_abort`, not a broken instrument: all 8 calls and 192 row-side evaluations
were present, every price and provenance check passed, every later phase stayed closed, and all localization outputs
were null.  The result rules out this exact strict prompt family without relaxing thresholds or selecting easier rows.

### The next authority was repaired before freeze

The successor build initially called local token repetition “task 18,” randomly sampled token roles, and incompletely
recomputed row semantics.  CPU review found and corrected four distinct issues before any GPU access:

- behavior-bank task 18 already means named-field retrieval, so the new strict unit is task 21;
- its canonical behavior ID is the existing `verbatim_repeat.copy`, not a reversed new registry key;
- 84 verified tokens now form four disjoint 21-token phase pools, and a cyclic assignment uses every token exactly once
  as target, alternative, control replacement, and every filler-position role;
- mutation tests that previously passed now reject mismatched text versus structured tokens, wrong causal-effect
  labels, impossible word counts, and the wrong OOD prompt shape.

The repaired draft has 21 linked A1/A2/P/C panels per phase.  Its FIT compiler currently produces exactly 8 forwards,
168 row-side evaluations, and 1,344 raw numeric bytes, with no later-phase bytes.  Synthetic evidence checks confirm
that 18/21 successes passes the 0.85 cell bar, 17/21 fails closed with null projections, and incomplete or duplicated
evidence is rejected.  The builder reports 69/69 current CPU tests; final hashes, checked-in dry run, broad tests, and
commit are still pending and therefore no execution is authorized yet.

### Diagnostic work supplied constraints, not adoption evidence

The constant-write lineage found that replacing attention 1 and attention 5 by fixed vectors costs 0.377 nat while
deleting them costs 3.019 nats.  Ranking by `deletion cost - replacement cost` avoided both earlier mistakes: ratios
favored tiny denominators, while replacement cost alone favored components that did not matter.  Joint fixed-vector
errors were sub-additive for the saving-ranked set, so intervention errors can cancel as well as compound.

These are useful implementation diagnostics, but they do not name a computation, extract a task circuit, or establish
selective manipulation.  They therefore do not update the adoption ledger.  The durable methodological rule is to
score both endpoints and the actual difference made by keeping an approximation; it is not a license to replace
components by constants based only on aggregate document CE.

The diagnostic bank grew from 16 to 21 behaviors with old rows verified bit-identical.  Only one of five new hand-picked
behaviors cleared its capability threshold.  A separate screen found that round numeric starts switch the model from
“last value plus one” to genuine step continuation.  Thus prompt demonstrations and pooled numeric accuracy can hide
mixtures of computations.  Future numeric authorities must stratify roundness prospectively.  None of these one-run
bank findings is four-phase circuit evidence.

## Is the current route still highest-information?

**Yes, conditionally.**  The bottleneck is not another component ranking; it is obtaining at least one behavior with a
clean authority that can survive capability, FIT localization, held-out interchange, selective removal, and joint
composition.  Task 21 is the cheapest way to exercise that complete protocol after task 17's valid null.

There is an important limit.  In every task-21 row the answer is the immediately preceding prompt token.  A direct
path from the final token embedding through the residual stream can solve the task without remote retrieval or
induction.  A capability pass would validate the dataset and pipeline, but would not by itself make task 21 a rich
attention circuit.  Any localization phase must compare proposed attention/MLP paths with that direct residual path.
`A2` is helpful because it leaves an older conflicting target visible; `C` changes repeat strength while preserving
identity.  Still, a component effect may encode confidence or generic token identity rather than an attention-based
copy operation.

## Confound audit

- **Post-selection:** old bank results chose the behavior only.  New task-21 rows and thresholds remained outcome-blind.
- **Token difficulty:** fixed by exact role balance rather than merely more examples.
- **Shared rows:** A1/A2/P/C share one base by design.  Later uncertainty estimates and decisions must group by linked
  panel; duplicate base evaluations cannot be treated as independent examples.
- **Restricted metric:** “accuracy” means answer logit greater than the maximum registered prompt-word foil, not full
  vocabulary top-1 accuracy.  All claims must retain that definition.
- **Direct path:** final-token embedding/residual information is an explicit competing explanation, not background to
  omit from a circuit diagram.
- **Frame/gauge mixing:** no learned basis exists yet.  If localization opens, causal equivalence under downstream
  readers must determine grouping; head coordinates alone are insufficient.
- **Nonlinear composition:** single-component margin effects cannot predict joint removal or installation.  The recent
  constant-write results show both cancellation and compounding are possible.
- **Precision and dead controls:** capability has no intervention precision issue.  A later interchange must retain
  applied-versus-planned hook checks, active P/C controls, and complete call coverage.
- **Numeric mixture:** irrelevant to task 21's word vocabulary, but now mandatory to control in numeric successor tasks.

## Genuinely different next paths

1. **Finish task 21 as a strict protocol anchor.**  It can advance held-out prediction, selective manipulation, and
   stable identification only after capability passes and direct-path controls are registered.  Kill it if the final
   CPU review finds a closure/semantic defect or if native capability fails.
2. **Repair task-14 subject–verb agreement.**  This is scientifically richer because the controlling noun is not the
   final token, archived capability was 1.00, and prior authority can be adapted.  It should become next if task 21
   validates the pipeline, or immediately if task 21 fails.  Kill it if the repaired A1/A2/P/C grammar cannot make
   head noun, attractor, and surface changes independent.
3. **Repair the induction/copy-successor interchange instrument.**  This directly tests remote selector-payload
   composition and cross-head reuse, but R593's applied-versus-planned numerical mismatch must be understood first.
   Kill a repaired run if the tripwire cannot distinguish harmless floating-point accumulation from a wrong edit.
4. **Downstream-reader-defined factorization of exact bilinear interactions.**  Once a strict task identifies live
   writes, group directions from different modules when their interchanges are indistinguishable to the same later
   quadratic readers.  This targets cross-module grouping and gauge-stable identification.  Kill it if the equivalence
   classes do not transport to held-out panels or selective edits.
5. **Fixed-write compilation.**  Keep attention 1/5 constant replacement only as an implementation control or priced
   baseline.  It becomes circuit evidence only if task-conditioned interventions show what variable the fixed write
   supplies and unrelated tasks are preserved.  Aggregate document CE cannot promote it.

## Ranked decision and action

1. Complete and independently review task 21's frozen CPU authority/compiler.
2. If approved, build the smallest model-facing FIT capability adapter and authorize one 8-forward run through the
   managed queue.
3. On a pass, preregister FIT localization with an explicit direct-final-token residual baseline and linked-panel
   analysis; on a fail, close task 21 unchanged and move to repaired subject–verb agreement.
4. Keep induction-instrument repair and downstream-reader factorization as the next independent mechanism routes.

This is not a rank-reduction program.  The live action is already concrete: the task-21 builder is freezing hashes,
adding checked-in dry-run verification, and running the broad CPU suite.  No GPU run is authorized at this boundary.
