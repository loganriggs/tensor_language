# Hourly strategic review — 2026-09-04 00:22 UTC

## Circuit interpretation targets

A useful decomposition must eventually provide all seven of the following:

1. **Computational specification:** what information is read, what operation is
   performed, what is written, and which later computations use it.
2. **Grouping across modules and splitting within modules:** join pieces of
   different heads or MLPs when downstream computation treats them as one
   variable, and split a native module when its pieces serve different tasks.
3. **Held-out and OOD prediction:** predict activation and causal effect on new
   examples, tasks, and distributions.
4. **Extraction or sufficiency:** run the proposed computation in isolation or
   through a specified background and reproduce the target signed effect.
5. **Selective manipulation:** remove, swap, or edit the computation while
   preserving unrelated behaviors, explicitly accounting for redundancy and
   interactions.
6. **Composition and reuse:** predict how shared and task-specific pieces behave
   together across several circuits.
7. **Stable identification:** preserve the claimed variables across data splits,
   fitting restarts, and equivalent changes of basis, or define them by
   downstream operational equivalence.

The program-level goal remains a smaller transparent tensor program that is
predictive, composable, manipulable, and simpler under literal storage, compute,
edge, state, and program prices. Lower rank or lower cross-entropy alone cannot
satisfy this goal.

## What changed in the last hour

The parent-plus-two-agent circuit bootstrap moved from design to one complete
review/repair/execution cycle on the R591 diagnostic:

- Exact cross-review blocked the first R591 candidate because causal labels used
  the wrong comparisons, selected IDs were omitted, dry-run authority parsed
  older outcomes, and the managed path could execute changed bytes after hashing.
- Exact cross-review blocked the first R590 candidate because executable
  dependencies could run before being pinned.
- The shared handoff grew from 23 to 26 tested lessons covering classification
  authority, transitive outcome blindness, and immutable execution.
- A centered bilinear intervention was derived and independently reviewed. Its
  algebra is correct, but the review established a scope limit: it is a partial
  projected-output factor intervention, not automatically a normalized full
  attention pattern, a realizable query/key state, literal removal, or
  sufficiency. This became shared lesson 27.
- Repaired R591 commit `a5e1dd022` passed a different agent's exact review in
  commit `d80ce3f39` and was executed through the managed queue.
- R591 completed exactly 234 FIT-only forwards. It found a mixed failure: the
  current replay hook reaches $1.811981201171875\times10^{-5}$ final-logit error,
  while native padding changes reach
  $2.8848648071289062\times10^{-5}$. Fixed-shape membership and observation-only
  comparisons were exactly zero. The total discrepancy decomposed into hook plus
  batch/padding to $4.55\times10^{-13}$ residual.
- R590's first repair was blocked again: one nested executable still reopened a
  mutable dependency, its dry run read prior outcome artifacts, and provenance
  could bind path bytes rather than executed snapshots. Its second repair is now
  active. No R590 model call or outcome has been opened.

Failures have therefore improved shared tools rather than being erased. There is
still no R585 scientific result.

## Is the current path still highest information?

Yes for the next decision, with a limit on further wrapper work. R591 has now
answered the only question blocking the induction factor experiment: both
contraction order and physical padding matter at the frozen threshold. A
prospective fixed-geometry centered-factor experiment can directly test the
semantic selector/content hypothesis. Another broad component sweep or rank
search would not resolve that question.

R590 remains useful because it tests the same evidence interface on a genuinely
different behavior. But if one more exact review discovers another recursive
execution-closure failure, the higher-information response is a small clean-room
producer with no historical executable imports, not another layer of wrappers
around the old code.

## Confound audit

- **Baseline subtraction:** R591 shows that comparing logits across different
  padding shapes is not exact enough. Every successor arm needs a native baseline
  on identical token tensors, membership, order, and padding.
- **Frame mixing:** the selector/content object spans four sites. Factors must be
  captured before intervention and all inserted deltas frozen, so an early edit
  cannot redefine a later supposedly fixed factor.
- **Nonlinear loss composition:** causal claims must be computed from paired
  row-level logits/CE; sums of mean CE effects are not an interaction
  decomposition. The bilinear mixed finite difference is defined at the inserted
  tensor, with behavioral interactions measured separately.
- **Shared token difficulty:** paired counterfactual groups and answer-preserving
  active controls remain required. No unpaired token-average attribution is
  promoted.
- **Leakage/post-selection:** R591 opened FIT only. The R585 successor must freeze
  its repair and pass exact review before reopening SELECT; FINAL and OOD remain
  closed.
- **Dead controls:** observer-only and fixed-shape membership controls were live
  measurements and exactly zero. Future answer-preserving controls must also have
  nonzero inserted tensors.
- **Precision floor:** the old $10^{-5}$ bar stays fixed. We change the operational
  computation and paired geometry rather than enlarging the tolerance.
- **Normalization/realizability:** a partial equality-coefficient swap may not be
  a normalized or query/key-realizable attention pattern. It is named only as an
  output-factor mediator intervention.

## Genuinely different routes

1. **Centered output-factor route:** use
   $B(E',U')-B(E,U)$ with fixed padding and exactly paired native baselines.
   This directly tests the proposed semantic mediator.
2. **Head-space literal route:** construct the complete equality-supported
   128-dimensional head contribution, combine it with the native nonequality
   remainder, and apply $W_O$ once. This is closer to literal remove-and-insert
   and is the stronger later sufficiency/removal test.
3. **Realizable query/key route:** fit or solve for a state change whose native
   Q/K projections produce the desired attention change while respecting the
   full pattern. This asks whether the mediator corresponds to an internal
   attention computation rather than only an output-space intervention.
4. **Downstream causal quotient:** group factor changes that every registered
   downstream reader treats identically, including across heads. This may yield a
   gauge-stable variable even when Q/K or value coordinates are nonunique.
5. **Clean-room experiment route:** rebuild the minimal row authority,
   intervention, evidence writer, and validator without importing historical
   experiment executables. This is now the fallback if R590's recursive wrapper
   closure continues to dominate research time.
6. **Distinct-behavior wave:** after the interface survives, run pending-opener
   state and numbered-list successor in parallel. Agreement across those very
   different behaviors would identify reusable methods; disagreement would expose
   behavior-specific assumptions.

## Ranked next moves and kill evidence

1. **Freeze the R585 successor design from R591.** Target changes: computational
   specification, within-head splitting, and interchange. Measurement: exact
   zero self-delta, matched-geometry native equality, and opposing selector-only
   versus content-only causal effects. Kill it if the full-state ceiling fails or
   active controls change comparably to targets.
2. **Finish and cross-review R590's second execution-closure repair.** Target
   changes: trustworthy selective manipulation evidence on a second behavior.
   Kill the wrapper approach after another recursive-closure failure and move to
   a clean-room producer.
3. **After a held centered factor, run the head-space literal removal arm.** Target
   changes: extraction/sufficiency and selective removal. Kill the claim that the
   factor is a removable native subcircuit if literal deletion cannot reproduce
   the centered causal effect selectively.
4. **Launch the two distinct circuit tracks only after this interface cycle
   closes.** Target changes: cross-module grouping, reuse, and OOD prediction.
   Kill a proposed shared method if it succeeds only on one prompt family or one
   native module boundary.

## Anti-rank-drift decision

No selected next move optimizes rank, variance, reconstruction error, storage, or
aggregate CE as its scientific objective. The current decisions concern which
semantic factor is causally active, whether it can be exchanged and removed, and
whether the same interface transfers across behaviors. Rank may later price an
already identified executable circuit; it is not being used to discover one.

The live continuation is the R585 successor design plus the active R590 repair;
the completed R591 run is not a stopping boundary.
