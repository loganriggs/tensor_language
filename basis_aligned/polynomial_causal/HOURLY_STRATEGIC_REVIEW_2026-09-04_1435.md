# Hourly strategic review — 2026-09-04 14:35 UTC (Codex)

## The goal being optimized

The output is not a low-rank approximation or a list of important modules. We want circuit descriptions that:

1. say what information is read, what computation is performed, what is written, and what later computation uses it;
2. group pieces from different heads or MLPs when they implement the same reusable variable, and split a head or MLP when
   different parts do different jobs;
3. predict held-out examples and meaningful distribution shifts;
4. reproduce the target computation when extracted;
5. change the intended behavior under a swap or removal while preserving unrelated behaviors;
6. compose predictably with other circuit parts and expose reuse across tasks; and
7. remain identifiable across data splits, fits, and gauge choices, or be defined by downstream operational equivalence.

The ten-minute operating target is one preregistered causal screen or honest null per ten minutes of serial work once the
shared machinery exists. A passing screen is only a lead for deeper identification; it is not yet an interpreted circuit.

## Repository timestamp and work completed this hour

At 14:35 UTC the relevant sequence was:

- 13:52: circuit-only ten-minute operating contract committed;
- 14:01: machine-checked prior-art receipt validator committed;
- 14:11: selective A1/A2/P/C causal scorer committed;
- 14:14: declarative 19-residual-boundary plus 36-module compiler committed;
- 14:17: capability changed from pooled family averages to separately gated construction/direction cells;
- 14:25: deterministic candidate bank, model-facing producer, and append-only screen ledger committed;
- 14:28: Claude's read-only review found all five earlier blockers closed and withdrew an impossible token-identical-prompt
  suggestion;
- 14:31: exact managed entry, prior-art receipt, and right-padded call plan committed;
- 14:33: the first managed execution finished its computation but the result writer rejected tuple containers;
- 14:34: the strict-JSON repair landed, and the repeated managed execution published an honest capability null in 1.990 s.

No frontier, compression, rank-reduction, or cleanup question was opened after the circuit-only boundary was restated.
Claude's earlier frontier work stopped, and its subsequent work was limited to circuit review and runtime auditing.

## The first counted screen

The candidate asked whether a common internal state carries the choice between `.` and `?` across two different syntactic
constructions. For every linked group:

- A1 changed an internal reporting frame from a statement to a question or back;
- A2 changed a direct declarative construction to a direct question or back;
- P changed the reporter word while preserving sentence type and answer; and
- C asked the model to copy a visible `.` or `?`, providing the same two output tokens through an unrelated computation.

The capability check computes whether the native model ranks the registered answer above the foil on both the base and donor
prompt. It does this separately for every construction and ordered answer direction. The result was:

| family/cell | base accuracy | donor accuracy | gate |
|---|---:|---:|---|
| A1, statement → question | 100% | 100% | pass |
| A1, question → statement | 100% | 100% | pass |
| A2, statement → question | 100% | 100% | pass |
| A2, question → statement | 100% | 100% | pass |
| P, reporter rewrite in a statement | 100% | 100% | pass |
| P, reporter rewrite in a question | 100% | 100% | pass |
| C, visible period → visible question mark | 100% | 0% | fail |
| C, visible question mark → visible period | 0% | 100% | fail |

The model followed the grammatical sentence ending rather than the explicit copy instruction. Therefore C was not a behavior
the model could perform, and no intervention result using it would be interpretable. The producer stopped after eight native
calls: 256 example evaluations, 2,048 retained numeric bytes, and 1.990 s. No residual, module, or head patch was run. This is
an informative screen-design null, not evidence for or against a sentence-mode circuit.

The prior-art receipt also prevents this from being mislabeled. It binds §§1282/1289/1313/1345/1597-1600/1631, the circuit
registry, and the dossier. Rediscovering attention layer 10 or head 10.5 alone was explicitly registered as replication.

## Measured serial efficiency

From the 13:52 operating-plan commit to the 14:34 valid receipt was about 42 minutes. That misses the ten-minute target by
4.2 times, but it contains the one-time construction of the shared scorer, compiler, producer, candidate validator, prior-art
gate, managed entry, and ledger. The steady-state GPU portion was only 1.990 s.

The main time buckets were approximately:

| bucket | serial time | what it bought |
|---|---:|---|
| prior-art and duplicate guard | 9 min | current source-bound novelty classification |
| scorer and exact bars | 10 min | target transfer plus same-answer and unrelated-behavior controls |
| compiler and call accounting | 5 min | exact 55-site screen and conditional head stage |
| candidate, producer, and ledger | 11 min | first full reusable vertical slice |
| independent review and integration | 6 min | five design blockers closed before GPU use |
| packaging failure and repair | about 1 min | strict tuple-to-JSON regression added |
| successful managed computation | 1.990 s | early capability null |

The largest reusable engineering win this hour was eliminating unnecessary length-specific calls. The old compiler would have
described 593 calls because prompt lengths differed, even though the executor already right-padded each batch. An explicit
right-padded mode reduced the exact maximum to 264 calls, 329 fewer calls or 55.5%, without changing the 8,448-example ceiling.
It also removed a plan/executor disagreement that would have made the price receipt false.

## What still prevents one result every ten minutes

The bottleneck has moved from GPU computation to candidate preparation: prior-art audit, meaningful counterfactuals, and control
capability. The empty GPU queue after the 1.990-second null is evidence of that. Two circuit-only workers are now preparing
non-overlapping pronoun-antecedent and quote-parity candidates while the main process reviews and integrates results.

The first failure also shows a specific missing stage: a proposed negative-control behavior can consume a full 32-group dataset
before anyone verifies that the model performs it. The next pipeline revision should add a small, explicitly developmental
capability probe on disposable examples. It may reject an unusable prompt template, but its examples must never enter the frozen
FIT authority and its outcomes cannot be used to tune scientific bars. This is an engineering precheck, not a scientific result.

The remaining repeated authoring cost is the 240-line task-specific managed wrapper. After the two new candidate modules expose
their interfaces, the bounded refactor is one generic managed-screen function plus a tiny declarative request per candidate. That
should reduce repeated integration and receipt code without creating another experiment-specific compiler.

## Next-hour decisions

1. Accept or reject the pronoun and quote candidates based on their prior-art receipts and counterfactual validity, not because a
   queue slot is empty.
2. Before freezing either full authority, run the model-free checks and use a disjoint small developmental capability probe for any
   uncertain control construction.
3. Extract the stable managed execution/serialization/ledger path into one generic function only after the second candidate proves
   which fields actually vary.
4. Queue the first valid second candidate through the managed runner. Count its total serial preparation time from prior-art start
   to terminal receipt.
5. If a screen passes, continue the ten-minute loop in parallel while a separate deeper job identifies a smaller causal subspace,
   tests held-out transfer, and translates that subspace into the bilinear weights.

The hourly verdict is therefore: the scientific screen loop works, duplicate detection works, early stopping works, and GPU time
is negligible. Throughput is not yet at the target because candidate/control design remains serially expensive. The next
engineering changes target that bottleneck directly rather than optimizing rank, reconstruction, or raw GPU arithmetic.
