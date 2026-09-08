# Hourly strategic circuit review — 2026-09-08 02:15 UTC

## Circuit interpretation targets

1. **Computational specification:** identify reads, operations, writes, and downstream consumers.
2. **Cross-boundary grouping and within-module splitting:** group or split by causal interchange rather than native module names.
3. **Held-out and OOD prediction:** predict unseen text, constructions, and shifts.
4. **Extraction or sufficiency:** execute an isolated circuit or explicit circuit-plus-background interface.
5. **Selective manipulation:** change the intended behavior while preserving controls and measuring redundancy/interactions.
6. **Composition and reuse:** predict how shared and task-specific subcomputations behave jointly.
7. **Stable identification:** survive corpus splits, gauges, fitting restarts, or operational downstream equivalence.

The full goal remains a smaller transparent tensor program that is predictive on fresh/OOD text, composable under joint replacement, selectively manipulable, and literally simpler in storage, compute, edges, state, and program price. Rank and reconstruction remain controls or post-identification prices, not substitutes for these circuit targets.

## What changed since 01:15

Six causal receipts landed in the temporal branch.

1. The complete layer-12--17 attention/MLP write bank mediates only `.141-.207` reset and `.127-.210` rescue. No stable singleton exists; MLP13 is the largest positive singleton and MLP17 opposes it. This closes greedy downstream-write union and establishes direct residual carry.
2. Complete residual-state swaps from entry12 through post-MLP17 all mediate exactly `1.0`. This identifies the complete entry12 residual tensor as the causal interface and shows recurrent `v1` carries no independent difference, but the flat deterministic curve does not name a later consumer.
3. A cross-fitted rank-two A1/A2 DIM union at entry12 is sufficient (`.795-.881`); pooled rank one fails A1. Exact full-P-span complementation removes all P/C collisions but destroys target (`.023-.058`). The complement loss succeeds at invariance and fails because the target overlaps the nuisance response space.
4. The gold three-branch A1/A2/off finite mixture composes the union with zero control effect. A router using only native base state fails because paired A1/P rows have identical base text and therefore identical features.
5. A source-delta nearest-centroid router predicts off everywhere. It is selective but carries no target.
6. A class-balanced dual-ridge router selected by nested leave-one-group-out CV achieves perfect training-parity CV, then also predicts off everywhere on held parity. Even/odd parity exactly reverses present-to-past versus past-to-present, exposing signed-direction memorization despite L2 regularization.

The current object is therefore precise: two construction coordinates are jointly sufficient and compose under a correct finite gate, but branch identification must be invariant to transformation direction and must observe source/background information.

## Highest-information route

The next bounded test is the fixed six-scalar Gram router. It replaces raw signed 2304-dimensional deltas with semantic-token and prefix-averaged squared norms and cross-inner-products of the two proposed expert writes. Those features are invariant to simultaneous sign reversal and directly test the interaction needed by the observed parity failure. This advances stable identification, computational specification, composition, and selective manipulation. It is not a rank or compression experiment.

If it succeeds, the finite program receives OOD router evaluation before promotion, followed by exact weight translation of both state coordinates and the router features to candidate upstream writers/downstream readers. If it fails, response-signature routing closes and the alternatives are an explicit token-level source router or a prospectively richer nonlinear circuit—not another ridge penalty grid.

## Confound audit

- **Baseline subtraction:** every causal program is scored against the same upstream-off execution, with full target effects defined by fixed expert on-minus-off states.
- **Frame mixing:** A1/A2 experts remain separate; the union is shared but each held panel is scored separately. No pooling can hide a failed expert/parity cell.
- **Nonlinear composition:** control KL/flips stay cellwise. The gold router is an upper-bound arm, never a selection input.
- **Shared token difficulty:** rowwise effects precede aggregation; group parity is disjoint. The newly discovered parity/direction equivalence is now an explicit tripwire.
- **Leakage:** learned routers cannot read held labels, answer IDs, logits, or outcomes. They may read both source and background because the causal replacement operator itself receives both.
- **Dead knobs/post-selection:** the Gram representation and centroid rule have no grid. All six features and their order are frozen before execution.
- **Precision/noise:** replays are exactly zero so far; substantive target bars are hundreds of times larger than numerical error.
- **Memorization:** the nested ridge failure proves within-parity leave-group-out CV is insufficient for direction generalization. The next representation encodes the required sign gauge rather than relying on regularization to learn it.
- **Tautology:** complete-state equality gives deterministic suffix equality and is not reused as consumer localization evidence.

## Alternative routes and kill evidence

1. **Six-scalar sign-invariant Gram router (current).** Kill if held macro accuracy remains below `.875`, any P/C row routes to target, or routed target projection misses `.75` despite the gold pass.
2. **Explicit token-level source router.** Read the changed temporal cue/auxiliary token pair using a prospective finite token rule or localized attention interface. Kill if it does not transfer to fresh lexical/construction banks or merely encodes row IDs.
3. **Richer nonlinear causal-response router.** Use bilinear/tensor interaction features only after the six-scalar lower-complexity object fails. Kill if nested complete-direction holdouts fail or control leakage remains.
4. **Weight-interface translation of the rank-two union.** Defer until a selective gate exists. Static reader rankings without causal selective admission are demoted by the L15H5 lesson.
5. **Direct final unembedding read decomposition.** Useful to price the carried state after identification, but it cannot solve branch selectivity and is therefore lower priority now.

## Throughput and systems audit

Authoritative run logs show receipts at approximately 01:23, 01:35, 01:45, 01:56, 02:02, and 02:14: six causal screens/nulls in one hour, approximately ten-minute mean spacing. GPU execution totaled only about 15 seconds; scientific specification, reusable contracts, publication, and hash-bound handoff dominated. The serial median is near the ten-minute operating target and materially improved from the prior hour because state execution, basis fitting, routing, and scoring reused shared components. The queue repeatedly drained because each GPU screen lasts 2--4 seconds, but every gap was active implementation rather than idle waiting.

Validation remained smaller than scientific design/execution: focused suites reused 3--10 tests, no bespoke compiler or post-outcome audit was built, and OOD was correctly deferred until a candidate passes. The next engineering block adds only the six-feature contract and a thin reuse of the 24-forward router runner.

`CIRCUIT_FOCUS: PASS`

`CEREMONY_BUDGET: PASS`

`NOVELTY_LESSON_GATE: PASS`

The route changes representation, not objective: from signed raw deltas that fail direction reversal to a sign-invariant quadratic response signature. The concrete continuation is implementation and managed execution of the preregistered Gram router.
