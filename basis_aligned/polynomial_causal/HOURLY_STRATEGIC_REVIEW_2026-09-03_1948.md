# Hourly strategic review — 2026-09-03 19:48 UTC

## Circuit interpretation targets

A useful circuit decomposition must eventually provide all seven kinds of evidence:

1. **Computational specification:** what information is read, what operation or composition is performed, what is
   written, and which later computations use it.
2. **Cross-module grouping and within-module splitting:** merge parts of different attention heads or MLPs when later
   computation treats them as the same variable, and split one native module when its parts perform different tasks.
3. **Held-out and out-of-distribution prediction:** predict activations and behavioral effects on unseen inputs and
   shifted task variants.
4. **Executable extraction or sufficiency:** run the isolated computation, or a precisely specified interface plus
   background, and reproduce the target computation or signed causal effect.
5. **Selective manipulation:** remove, swap, or edit the circuit while preserving unrelated computations, including
   explicit redundancy and interaction checks.
6. **Composition and reuse:** predict joint behavior when shared subcomputations are combined with task-specific uses.
7. **Stable identification:** survive data splits, plausible gauge changes, and fitting restarts, or be defined by
   operational equivalence under downstream readers.

The full program goal is a smaller transparent tensor program that is predictive, composable, manipulable, and simpler
under literal storage, compute, edge, state, and program pricing. Compression measurements are engineering prices, not
substitutes for circuit interpretation.

## What changed since the 18:48 review

The highest-information changes came from the causal circuit lane, not another rank sweep:

- R576's exact numbered-list cached-value contribution was necessary but not selectively removable. R582 therefore
  split its finite effect inside later bilinear MLPs into an exact background-cross term and contrast-self term.
- R577 established an audited complete-state null for broad numeric swaps. Several sites transferred every target
  direction but failed active controls. This blocks more whole-state or low-rank searches at the same sites.
- R578 repaired the induction counterfactuals; R580's native FIT/SELECT behavior held scientifically. R581 independently
  reproduced all measurements and all 86 bootstrap cells, but its formal audit failed because `next_step` was a JSON
  list rather than the frozen scalar string.
- R584 implemented the exact downstream MLP split. Cross-lane adversarial review blocked execution before GPU use:
  missing complete groups were accepted, normal nulls could emit Infinity, per-row/site exactness and position evidence
  were incomplete, and provenance was insufficient.
- A generic `result_contract.py` now rejects exact-membership, field-type, finite-JSON, split, price, mutation, and hash
  failures. Fourteen focused tests pass. Applied to the real R580 artifact, every configured non-type contract check
  passes over 3,240 rows, while the strict contract rejects exactly `next_step: expected declared string, got list`.
- R586 is now a fully model-free, prospective clean replication package: it delegates the scientific computation to
  hash-pinned R580 code, writes a new result namespace, integrates the generic contract, preserves 95 forwards/zero
  backwards/no updates, and tests held and scientific-null fixtures. Its independent R587 audit must be frozen before
  any R586 result is opened.
- The R585 induction red team found a scientific, not merely administrative, correction: score-only and value-only
  attention arms must use factors cached before intervention. Otherwise an earlier score edit changes a later live
  value and the nominal single-factor arms leak into one another.

Claude's §2703 result also held: a second-order Fisher calculation prices joint rank-32 changes in blocks 11–17 within
roughly a factor of two and captures their large pairwise interactions. This is a useful post-identification price tool.
It does not discover a circuit, split a module by behavior, or pass selective manipulation, so it does not displace the
causal tracks below.

## Confound audit

- **Baseline subtraction and nonlinear loss:** R584 and R585 save logits, CE, full-vocabulary change, and factorial
  interaction terms. A donor-direction effect must improve donor CE, not merely damage the recipient answer.
- **Multiple mediators and frame mixing:** R585 will cache native recipient/donor score and value factors before any
  intervention, subtract the live term, and insert a frozen factorial combination at each later site.
- **Shared token difficulty:** counterfactuals are paired within semantic groups, and uncertainty resamples groups rather
  than individual rows.
- **Data leakage and post-selection:** FIT chooses under a fixed order; SELECT opens conditionally; FINAL_TEST/OOD remain
  closed. Token banks and semantic groups are split-disjoint.
- **Dead interventions:** control claims require nonzero intervention norms relative to matched target norms. Structural
  no-ops are labeled as algebra checks rather than selectivity evidence.
- **Precision and incomplete replay:** R584 repair must save all-row, all-site float32 exactness rather than a first-batch
  smoke test. Standard JSON must contain no NaN or Infinity.
- **Bookkeeping Goodharting:** a scientific pass cannot override a failed result contract. R580/R581 remain immutable and
  formally failed even though their scientific values reproduce.

## Genuinely different routes

1. **Exact downstream-use response basis (R584).** Split one broad carrier by what later bilinear MLPs compute with it.
   This directly tests computational specification, within-module splitting, selective manipulation, and reuse across
   three numeric formats. Kill it if the repaired exact instrument finds no component passing successor-versus-copy and
   active-null gates.
2. **Frozen attention score/value factorial (R585).** Treat the four equality-gated terms as a distributed computation,
   not four head units. Opposing selector-only and payload-only predictions test a real factorization. Kill it if the
   complete joint factor fails transfer or active controls, or if the single arms do not separate as predicted.
3. **Downstream operational equivalence.** If R585 holds, group score/value features across heads by identical responses
   of registered downstream consumers, then test interchange and composition on held-out prompts. Kill it if equivalence
   is unstable across consumers or data splits.
4. **Predictive-state causal quotient.** Define internal states as equivalent when every registered continuation and
   intervention produces the same output distribution. This is more global than the current hand-selected factors, but
   needs a tractable separating set of continuations. A counterexample continuation kills any proposed merge.
5. **Signed composition algebra.** Use Möbius interaction terms to represent how removals combine instead of summing
   individual importance scores. This becomes useful if exact factors are identified; it is killed as a compact account
   if higher-order residuals remain large on held-out compositions.
6. **Fisher subset pricing.** Use §2703 only to price already identified late-site implementations. It is demoted now
   because a cheaper rank subset alone changes none of the seven circuit targets.

## Ranked next actions

1. **Freeze R587, then run R586 through the managed GPU queue.** This is only a 95-forward prospective replication, but
   it is the hard dependency for the more informative R585 factor experiment. R587 failure blocks R585 rather than being
   normalized away.
2. **Finish the R584 contract repair and cross-lane re-review.** If approved, enqueue it independently of R586/R587; its
   result directly decides whether later MLP use separates successor from generic numeric carriage.
3. **Freeze R585's exact four-arm computation while outcomes remain closed.** Execution waits for R587, but the algebra,
   semantic coordinates, opposing predictions, and active-control census can be made audit-ready now.
4. **Only after a held R585 factor result, test shared cross-head/downstream equivalence and translate it to weights.**
5. **Do not spend the circuit lane on another rank, PCA, or reconstruction sweep.** §2703 may price a later candidate;
   it cannot supply identification.

The current causal direction survives the alternatives because it is the only live route that directly asks whether a
broad known computation can be split by behavior and selectively manipulated. The immediate action is already underway:
R586's package is complete, R584's repairs are active in a separate lane, and R587 is the next parent-owned freeze before
any new model outcome.
