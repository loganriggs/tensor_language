# Hourly circuit and systems review — 2026-09-04 17:00 UTC

## Circuit interpretation targets

A useful circuit decomposition must eventually provide all seven of the following:

1. **Computational specification:** what information is read, what operation is performed, what is written, and what later computation uses it.
2. **Grouping and splitting across native modules:** merge pieces of different heads or MLPs when they implement one downstream variable, and split a native module when its pieces do different jobs.
3. **Held-out and out-of-distribution prediction:** predict activation and causal effects on unseen documents and shifted task variants.
4. **Extraction or sufficiency:** an executable circuit, or an explicit interface plus declared background, reproduces the target computation or signed causal effect.
5. **Selective manipulation:** removal, interchange, or editing changes the intended behavior while preserving unrelated behavior, including tests for redundancy and interactions.
6. **Composition and reuse:** shared subcomputations work across tasks or modules and combine predictably with task-specific parts.
7. **Stable identification:** the units survive data splits, fitting restarts, and plausible changes of basis, or are defined by downstream operational equivalence.

The program-level goal is a smaller transparent tensor program that is predictive, composable, and manipulable and has a literal storage, compute, edge, state, and intervention price. Rank or compression alone does not meet any circuit target.

## What changed since 16:00

The head-11.3 route advanced in two different ways.

First, a complete-head interaction control jointly interchanged all eight other heads in attention 11. Across the four answer-changing subject–verb-agreement cells, their recovered effect was 0.66%, 0.96%, 0.64%, and 1.79%; their interaction with head 11.3 was 0.27%, -0.29%, 0.63%, and -1.07%. This cleanly supports treating head 11.3, rather than the native nine-head module, as the object to split on this task. It remains a screen on the FIT authority rather than held-out identification.

Second, a downstream-reader screen induced the head-11.3 donor effect and restored each later attention or MLP output separately to its native recipient value. No single module met the preregistered shared-reader bar. MLP12 was the closest common reader: it removed 16.06%, 6.21%, 13.73%, and 6.11% of the induced effect across the four task cells, with a 10.53% mean absolute effect and 1.18%/1.36% movement on the two controls. MLP11 was direction-dependent: -1.33%, 18.39%, -5.12%, and 12.96%, with 0.84%/1.23% control movement. Attention modules 12–17 were individually negligible. The registered result is therefore **inconclusive**, not a reader identification and not a null. A one-module restoration can miss redundancy or nonlinear joint use.

The deeper causal-projector implementation also received an independent code audit. The production backend, exact SELECT-cell coverage, healthier failure semantics, conditional rank-8 license, finite checks, and chance-corrected stability reporting now exist in the shared worktree and pass the focused CPU tests. A final audit found that its rank-zero and rank-128 replay flags are still hard-coded rather than measured. Program A must not run until those endpoint checks are real and included in its prospective price.

## Repository-timestamp throughput audit

| UTC | receipt or implementation boundary | serial interpretation |
|---:|---|---|
| 16:02:44 | natural copy/select control ended in a native-capability null | valid cheap terminal |
| 16:14–16:48 | projector preregistration, spectral initializer, data boundary, objective, price, and atomic claim | deep identification track; deliberately parallel work, not a fast screen |
| 16:33:06 | complement runner preflight failed | invalid instrument; no scientific reading |
| 16:35:29 | corrected complement screen completed | 0.51 seconds of model time |
| 16:49:42 | downstream-reader candidate atomically claimed | authoritative start of this candidate |
| 16:51–16:58 | runner and focused tests completed | about four minutes of authoring and review |
| 16:59:17 | downstream-reader result completed | 1.48 seconds of model time; 9.59 minutes claim-to-result |
| 17:00:28 | result scored, indexed, and claim released | 10.77 minutes claim-to-terminal bookkeeping |

The last fast screen was essentially on the ten-minute target; GPU compute was only 1.48 seconds. The avoidable cost is repeated runner/scorer/receipt authoring and post-run indexing. The new atomic claim gate and authority-plus-fast-ledger prior-art search reduce duplicate work. The next systems improvement is to make the four-corner interaction screen declarative over the existing cached-module runner instead of creating another general compiler.

The first run of `circuit_latency.py --since 16:00` is itself invalid as an hourly measurement: it assigned both new candidates the subject–verb family's earliest 15:17 files and reported false 77.7- and 101.5-minute serial times. The exact claim ledger says the reader began at 16:49:42. Until the join is repaired to prefer exact candidate claims and request IDs over fuzzy family filenames, the raw command output must not be used to judge throughput.

## Confound audit

- **Baseline subtraction:** the reader screen measures the loss of the already-induced head effect relative to the same recipient baseline; no raw donor preference is relabeled as recovery.
- **Frame mixing:** the current screen uses full native module outputs, so it does not identify a rotated semantic coordinate. The projector track is explicitly responsible for that question.
- **Nonlinear composition:** a single-module restoration includes interactions with all untouched downstream modules but cannot assign pairwise MLP11–MLP12 compensation. The next factorial measures that missing term directly.
- **Shared token difficulty:** A1/A2 directions are reported separately, and noun-identity and distractor-number controls are active. MLP11's sign reversal across directions prevents a pooled-reader claim.
- **Leakage and post-selection:** the reader candidates were fixed before the run and use only FIT rows. Projector Program A still cannot open outer VALIDATION, but its process can read metadata authorities; call this a token boundary, not complete filesystem isolation.
- **Precision and health:** native replay passed to about $1.1\times10^{-5}$ in the reader screen. Projector endpoint replays remain a blocker because the present booleans are not measurements.
- **Dead controls:** the complement's environment-style preflight initially executed the model. That output is explicitly invalid; the repaired wrapper exercises the real managed preflight contract.

## Ranked next moves and falsifiers

1. **MLP11 × MLP12 downstream-reader factorial.** This advances grouping/splitting, selective manipulation, and computation specification. Measure four exact suffix states after inducing head 11.3: restore neither MLP, restore MLP11 only, MLP12 only, or both. A stable positive joint effect or a large interaction across all task cells would identify a downstream reader pair; small joint effect or control movement comparable to task movement kills the pair hypothesis.
2. **Causal projector inside head 11.3.** This advances stable identification, held-out prediction, and selective interchange—not compression for its own sake. Real rank-0/rank-128 replay, healthy matched random/permuted controls, and exact SELECT coverage are prerequisites. Healthy failure through the licensed dimensions kills the small-interchange-subspace hypothesis.
3. **Weight translation after projector identification.** Contract the identified output projector through head 11.3's output projection and the measured MLP readers. This advances computational specification and extraction. Failure to predict finite held-out effects from the contracted weights kills that proposed weight-level explanation even if the activation projector transfers behavior.

The current route survives alternatives because the complement screen rules out a same-layer head coalition, while the reader screen supplies a concrete, falsifiable MLP11/MLP12 interaction hypothesis. The projector remains valuable as a parallel identification track, but it must not serialize the ten-minute circuit loop or drift into a rank sweep.

## Hourly systems rule propagated

Claude's circuit-only instruction remains active on the shared board: at the first safe boundary each UTC hour, inspect authoritative repository timestamps, measure serial time to a screen/null, remove the largest repeated systems cost, search prior results, and start no compression-only work. The 17:00 correction adds that fuzzy family-file timestamps are not authoritative when an exact atomic claim exists.
