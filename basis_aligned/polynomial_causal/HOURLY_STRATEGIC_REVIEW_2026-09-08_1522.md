# Hourly strategic circuit review — 2026-09-08 15:22 UTC

## Circuit interpretation targets

1. **Computational specification:** identify what is read, computed, written, and consumed.
2. **Cross-boundary grouping and within-module splitting:** group operationally equivalent pieces
   across modules and split native heads/MLPs when downstream use differs.
3. **Held-out/OOD prediction:** predict causal effects on unseen text without reselection.
4. **Extraction/sufficiency:** execute an isolated circuit or an explicit interface plus background.
5. **Selective manipulation:** add, remove, swap, or edit the intended behavior with low collateral.
6. **Composition/reuse:** predict joint behavior and reuse shared subcomputations across tasks.
7. **Stable identification:** survive corpus shifts and legal gauges, or be defined by operational
   equivalence under downstream readers.

The full objective remains a smaller transparent tensor program that is predictive on fresh/OOD
text, composable, selectively manipulable, and simpler under literal storage, compute, edge,
state, and program prices. A stable shared activation subspace is only an interface candidate.
Once causal write/read edges are admitted, the weight-restricted objects `U^T W_write`,
`W_read U`, and the corresponding restricted bilinear core become the dataset-independent account
of possible computation; projected activations then delimit which of those possibilities are
reachable on a population.

## What changed since 14:22

Three measured receipts landed. The first P7 source-conditioned effect game was correctly retained
as invalid because its registered parent replay compared the all-live arm with a fully clamped
parent. The minimally repaired v2 reran the unchanged 258-arm intervention table and passed every
instrument and selectivity gate. It assigns `.4499/.4706` of the writer effect to direct/background
carriage and `.5501/.5294` to the seven P7 modules on FIT/HOLDOUT. Six modules have stable signed
Shapley credit; A11 and M11 are the leading pair. Their Grabisch-Roubens interaction is positive
at `.04550/.04608`, and the raw pair second difference is positive in all 32 contexts in each
phase. This is stable operational complementarity, not yet a directed source-to-reader edge.

The resulting v17 capability-only receipt has now also landed. All eight registered native
capability cells are perfect (`8/8` each), every one of the 16 A2 rows is jointly capable, all four
predictions pass, `causal_outcomes_opened=false`, and the immutable row digest remains
`287e744a...`. It cost two forwards/128 native examples and opens the preregistered A11/M11 causal
test without population repair or filtering.

While that screen waited behind verified-live v262, an exact reader-loss/output-rescue primitive
and executor were implemented and tested. A model-source audit corrected an important boundary:
A11 reads at `block11.attn` but its complete residual write exits `block11.attn.c_proj`, whereas
M11 reads and writes through `block11.mlp`. The frozen executor uses all 16 aligned A2 rows, the
fixed L7H7+L9H4 writer, individual and joint A11/M11 loss/rescue arms, and A12/M16 controls. It
costs 17 forwards/272 sequences and fits nothing. A dead transferred writer will be reported as a
scientific OOD null rather than crashing the managed job.

## Confound audit

- **Post-selection and leakage:** v17 texts, reporters, all rows, split, source heads, sites, arms,
  bars, and price were frozen before capability outcomes. Capability used no intervention output.
- **Reader versus writer boundaries:** normalized argument-0 reader values and complete module
  writes are hooked separately. Attention's auxiliary value-state return is not mistaken for its
  residual write. One-call coverage and cleanup are hard gates.
- **Sequential overwrite:** every reader arm starts from the same replayed block-10 writer state.
  Loss replaces the reader input with the source-absent value; rescue restores that same module's
  source-present complete output. This asks whether the source effect passes through the module,
  rather than whether a later native clamp can overwrite an omission.
- **Distributed/nonlinear composition:** individual arms test directed mediation, while four joint
  arms test whether physically removing both readers reproduces the positive A11/M11 interaction.
  No Shapley statistic alone is promoted to an edge.
- **Dead knobs and background routes:** source replay, source-absent self-clamp, matched A12/M16
  controls, exact hook counts, finiteness, and exact price gate interpretation. A dead v17 writer
  has its own terminal null.
- **Dataset versus weight structure:** PCA/SAE/hierarchical SAE on projected activations may map
  empirical occupancy, but it cannot define the computation by itself. Weight decomposition is
  deferred until the causal basis and reader edges are fixed, then tested by factor removal/swap.
- **DAS memorization:** this step fits no projector. Constrained DAS remains demoted until sealed
  optimization/evaluation populations, complement regularization, multiple restarts, and physical
  reader/writer validation distinguish task memorization from an invariant interface.

## Alternatives, ranked next moves, and kill evidence

1. **Bind and execute the sealed A11/M11 loss/rescue test.** This is now licensed by the all-pass
   v17 screen and directly advances specification, selective manipulation, OOD stability, and
   cross-module grouping. Kill a directed edge on unstable loss, failed same-module rescue, or a
   matched-control effect above `.03`.
2. **If both edges and the joint excess pass, construct the restricted write/read tensor.** Freeze
   the causal basis, contract writers and readers into it, then compare discrete sparse support,
   CP/Tucker/HT structure, and activation reachability. Require held-out factor interventions;
   descriptive decomposition alone is not a new circuit.
3. **If one edge passes, preserve only that route and redirect the other nomination.** Search the
   direct/background downstream atlas from the same source rather than widening A11/M11 post hoc.
4. **If the transferred writer is dead, build a new aligned sealed bank or source-dose instrument.**
   Do not interpret downstream reader nulls under a dead source.
5. **Regularized DAS.** Revisit only as a causal-coordinate proposal with noise/KL/Jacobian or
   complement regularization and sealed transfer; do not use it to rescue a failed physical edge.

The first move has the highest information gain because the exact effect game has already resolved
which pair merits a physical edge test, and the sealed native screen has removed the capability
ambiguity. Weight-tensor decomposition becomes useful immediately after, when it has a causal
subspace to restrict rather than a dataset-selected axis to redescribe.

## Throughput and systems audit

The interval produced three circuit receipts: one invalid-but-diagnostic effect game, its valid
fixed-protocol rerun, and one clean sealed capability screen. The two 258-forward effect games took
about 31--32 seconds each; v17 capability took 1.59 seconds. Most wall time was useful scientific
work: isolating the replay defect, interpreting exact interaction structure, building the
history-disjoint bank, auditing attention's split boundary, and preregistering the outcome-blind
edge experiment. Queue time overlapped verified v260/v262 computation. Both Supervisor runners
remain healthy; no direct GPU or duplicate enqueue was used.

`CIRCUIT_FOCUS: PASS` — every action narrowed a live circuit, validated an OOD population, or made
the next physical reader-edge intervention legal.

`CEREMONY_BUDGET: PASS` — three receipts landed, the only reusable helper was necessary to express
the exact causal intervention, and the next runner is already implemented rather than left as a
design document.

`NOVELTY_LESSON_GATE: PASS` — the v1 replay defect, earlier sequential-clamp null, existing
Shapley/Möbius construction, attention tuple boundary, and dataset-versus-weight distinction were
all incorporated. The new test changes the causal object from attribution to reader loss plus
same-module output rescue.

Concrete continuation: commit the reviewed executor, bind it to immutable capability result SHA
`53e89054...`, enqueue that exact runner once, and interpret the receipt immediately. On a full
pass, preregister the restricted-weight writer/reader tensor and factor-intervention audit; on a
partial/null outcome, follow the corresponding frozen terminal branch without retuning v17.
