# Hourly strategic circuit review — 2026-09-07 19:14 UTC

## Circuit interpretation targets

A useful decomposition must provide all seven: (1) a computational specification of what is read,
computed, written, and consumed downstream; (2) cross-boundary grouping and within-module splitting
instead of assuming heads or MLPs are semantic atoms; (3) held-out and out-of-distribution prediction;
(4) an executable sufficiency interface or extracted circuit; (5) selective removal, swapping, and
editing with redundancy and interactions measured; (6) predictable composition and reuse across
tasks; and (7) stable identification across documents, constructions, gauges, and fitting restarts.

The full goal remains a smaller transparent tensor program that is jointly predictive, composable,
manipulable, and simpler under literal storage, compute, edge, state, and program prices.  A causal
screen, low-rank response basis, or exact activation replay does not by itself meet that goal.

## What changed since 18:14

1. The original adaptive greedy and five-piece attention lattice were corrected after finding that
   legacy P/C base and donor sequences had unequal token lengths.  Their target-side A1/A2 results
   remain valid, but their selectivity terminals are retracted.  A reusable fail-closed alignment
   contract now checks every row's token count and terminal semantic position before an absolute-
   index patch.
2. A deterministic repaired authority preserves all A1/A2 rows exactly and supplies equal-length P
   and C controls.  Its capability-only managed run used two forwards, opened no causal outcome, and
   found all four P/base, P/donor, C/base, and C/donor cells 16/16 correct; each panel is 16/16 jointly
   capable.
3. The frozen 32-mask attention lattice was replayed at exactly 35 forwards.  Target metrics reproduce
   within `2.92e-6`.  Only the full mask reaches the A1 target bar (`.80535`; A2 `.86829`), but it has
   P median KL `.11881` with five flips and C median KL `.01156` with three flips.  No complete-head
   subset is selective under the separately reported control rule.
4. Singleton control attribution shows L8H1 is the largest P collision (`.03106`, four flips), with
   L9H1 and L11H3 each flipping three; complete attention 15 alone flips neither P nor C.  This turns
   the null into a within-head splitting problem rather than a reason to add more complete modules.
5. A two-fold, opposite-parity response split compared DIM rank one, task-SVD rank eight, and versions
   orthogonalized against aligned P.  Task-SVD preserves A1 (`.80380`) but transfers only `.57337` on
   A2 and retains four P flips.  P-complement SVD removes all P/C flips and lowers P KL to `.00913`,
   but destroys target transfer (`.57591/.40321`).  DIM is also construction-specific
   (`.72501/.43223`).  The complement objective succeeds at its stated loss but falsifies the assumed
   linear separation.
6. The exact attention-pattern/value/interaction factorial is preregistered, implemented, tested,
   hash-bound, and queued behind the live shared family-separability run.  It factors the four heads
   by `P'V'-PV=(P'-P)V+P(V'-V)+(P'-P)(V'-V)` while retaining the clean attention-15 branch.

## Confound and validity audit

- **Baseline subtraction:** target response and control KL/top-one changes are relative to each native
  base; all-base intervention closure is zero in both completed corrected experiments.
- **Frame mixing and leakage:** A1 selects lattice masks; A2 does not.  The linear split fits A1 and P
  only on opposite group parity, so a row never fits its own projector; A2/C never fit.  A2 did
  influence the older site pool and remains a construction check rather than pristine identification.
- **Nonlinear loss composition:** every causal arm executes through the full native suffix.  The
  complement result is not inferred from covariance alone: its zero-flip gain and target loss are
  measured interventions.
- **Shared token difficulty:** repaired P changes cue construction at fixed tense, noun, answer, and
  token count; C is unrelated and independently reported.  Perfect native capability rules out a
  weak-answer artifact, though both remain one prospective control design rather than universal
  invariance.
- **Post-selection:** the four heads and attention 15 were selected on v15 target evidence.  Current
  results are localization/splitting screens; a new target/control family is mandatory after a pass.
- **Dead knobs and tripwires:** hashes, exact row equality, per-row alignment, derived denominators,
  parity disjointness, ranks, closures, exact forward prices, and separate P/C rules all fired.
- **Precision/noise:** target replay error is below `3e-6`; causal differences are far larger.  The
  pattern/value run includes reconstruction and factor-closure checks rather than trusting a hook.
- **Optimization overfit:** complement removal was evaluated cross-fitted and still failed A2.  This
  supports the user's memorization diagnosis and shows that regularization alone cannot rescue an
  observation target that demands an impossible orthogonal split.

## Competing approaches and ranked next moves

1. **Exact attention pattern/value/interaction factorial (active).**  It changes computational
   specification, within-head splitting, selective manipulation, and weight-level translatability.
   Kill it if no factor subset reaches `.75` target transfer with zero P/C flips, or if full-factor
   closure cannot reproduce the parent.
2. **Source-position-by-factor split** if one operation is useful but still nonselective.  Partition
   its exact attention sum into cue, prefix, and local source terms before any learned projection.
   This changes what the head reads and writes.  Kill it if no source group has a held-construction
   target/control advantage under native execution.
3. **Regularized nonlinear DAS on the finite factor-response operator** if every exact factor subset
   fails.  DIM, task-SVD, full factors, and step zero remain fixed baselines; train on one parity and
   A1/P blocks, validate on the opposite parity, and keep A2/C sealed.  This changes stable
   identification and selective intervention.  Kill it if inner-fold selection again mispredicts A2
   or sacrifices a required factor/reader block.
4. **Direct QK/OV weight translation** follows a selective factor/source pass.  Contract its response
   covectors through Q/K/V/O weights to rank upstream writers and downstream readers, then test those
   edges causally.  Kill an edge when its weight prediction fails source/destination interchange.
5. **Adding complete MLPs or increasing rank** is demoted.  Complete MLPs were the largest collateral
   source, and rank eight already shows that target-only capacity preserves A1 while failing A2/P.

The operation factorial is now the highest-information route because the linear projection premise
has been causally falsified.  It exposes the bilinear computation needed for weight translation and
does not substitute compression or reconstruction for circuit evidence.

## Throughput and systems audit

The corrected control capability landed about 14 minutes after the alignment defect claim; the
licensed lattice followed eight minutes later; the cross-fitted response split followed ten minutes
after that.  Median scientific claim-to-result spacing is ten minutes.  GPU execution itself took
three, nine, and six seconds; most elapsed time was deterministic authority construction, runner
implementation, interpretation, commits, and shared-queue serialization.  Parse/gate/dry-run tests
were seconds and did not dominate.  The factor runner reuses the existing attention reconstruction
library rather than adding a new framework.  The broader fast-screen lane moved the distinct-circuit
count from 50 to 54 during the same interval, with its current family-separability recheck serialized
ahead of the factor run.

`CIRCUIT_FOCUS: PASS` — the hour repaired an invalid instrument, produced a valid exhaustive
selectivity null, causally tested a within-head linear split, and opened an operation-level circuit
test.

`CEREMONY_BUDGET: PASS` — focused contracts and tests were smaller than the scientific design and
native screens; no bespoke audit framework or rerun ceremony was added.

`NOVELTY_LESSON_GATE: PASS` — prior pattern/value machinery was searched and reused; unequal-token
controls, control-family masking, A2 post-selection, complete-module atom assumptions, and DIM/DAS
construction overfit were all incorporated explicitly.

The immediate continuation is the managed pattern/value factorial already in the queue.  Its result
must be interpreted into either an exact source/weight branch or the registered finite-response DAS
branch without pausing at the result boundary.
