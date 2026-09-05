# Hourly circuit and systems review — 2026-09-05 07:55 UTC

## Circuit interpretation targets

The immediate goal remains a reusable codebase and organized evidence for hundreds of high-quality circuits. A circuit explanation must eventually specify what is read, the operation performed, what is written, and what later computation uses it; split native modules or group pieces across modules when causal behavior demands it; predict held-out and OOD cases; support an executable extraction; allow selective manipulation; compose with other circuits; and remain stable across data splits and gauge choices. The full program goal is still a smaller transparent tensor program that is predictive, composable, manipulable, and simpler under literal storage, compute, edge, state, and program prices. Compression, rank, and reconstruction are not substitutes for these targets.

## What changed since 06:54

- The exact L13H8 semantic-opener contribution was split without fitting into a triplet mean $$\mu$$ and centered delimiter difference $$\delta$$. Natural delimiter swaps and removals showed that both are causal, but a clean hand-written type/common output-axis separation failed.
- A fresh two-factor removal factorial found only 6–7% normalized $$\mu\times\delta$$ interaction. The factors combine approximately additively; the failed clean axis is oblique, not hidden by a large nonlinear interaction.
- A prior-art audit prevented a duplicate 41-site atlas: R549 had already identified structured downstream responses. We tested causal mediation instead.
- Restoring MLP15 after removing either factor recovered only 1.3–2.2% of the centered closer-logit effect and made median correct-answer CE worse.
- Restoring the other three R549-eligible attention heads recovered at most 0.12%; restoring all three jointly was also negligible, with joint-minus-sum interactions around $$10^{-6}$$ to $$6\times10^{-5}$$.
- Restoring eight complete downstream module writes found no mediator. The largest target recovery was MLP14 for $$\delta$$ at 6.3–6.8%, and every median CE rescue was negative.
- This produces a new canonical lesson: structured downstream activation response is localization evidence, not causal reader or mediator evidence. Exact downstream rescue is required.
- The hourly throughput tool was wrong across midnight and when two experiments finished in one minute. Both parser errors were fixed with regression tests. It now counts nine scientific terminals in the preceding hour rather than two.
- Claude repaired sentence-terminal and quote-parity controls, producing causal carriers at residual layer 18. Claude also established that the current same-answer control threshold cannot fail and must not be described as a strong selectivity test.

## Throughput and time allocation

Authoritative terminals after the prior checkpoint were at 06:57, 06:58, 07:12, 07:20, two at 07:34, 07:43, 07:49, and 07:54: ten results including the just-landed complete-module null, or nine in the rolling sixty-minute window at measurement time. The corrected median serial interval was 7.0 minutes, within the ten-minute target. GPU computation remained about 6–8 seconds per basic screen; authoring and interpretation dominate.

The largest avoidable cost in this block was not safeguards but repeated successor-specific dispatch code. The scientific sequence itself stayed small: 5, 6, 6, 10, 6, and 20 forwards for the main L13H8 screens. Focused tests took under one second each. One review caught and reverted an attempted edit to an already-run parent before enqueue; the successor now keeps plural restoration local and parent hashes remained unchanged.

`CIRCUIT_FOCUS: PASS` — every scientific run tested a below-head factor, causal interaction, or downstream use; the only systems edit repaired circuit throughput accounting.

`CEREMONY_BUDGET: PASS` — three focused tests and one shared static/dry-run gate accompanied each downstream rescue; scientific design and execution dominated, with no bespoke compiler or backup suite.

`NOVELTY_LESSON_GATE: PASS` — R549/R551 were searched before coding, a duplicate atlas was rejected, MLP15 was not repeated in the module scan, and the activation-response-versus-mediation failure is now in `LESSONS.md`.

## Confounds and current interpretation

- Native capability is 100% in all four prompt families and exact replay error is zero.
- Both factor removals have large, positive correct-answer CE damage, so the rescue nulls are not dead interventions.
- High rescue cosine with tiny projection recovery is not mediation: a very small vector can point in the right direction while recovering almost no magnitude.
- Restoring a module's native write can make CE worse if that module adapts to or compensates for the changed residual state. Negative CE rescue therefore supports a response/compensation interpretation, not absence of reading.
- The persistent residual connection provides a bypass around every individual module write. Individual-write nulls do not show that downstream modules never read the factor; they show their changed writes are not substantial single mediators.
- Current controls test construction stability, not strong behavioral selectivity. No promotion claim uses the weak fixed control threshold.

## Alternatives and ranked next moves

1. **Residual-path versus grouped downstream-write factorial.** Cache every downstream write under native and factor-removed states. Cross upstream factor presence with the complete downstream write bank. This directly separates the persistent residual route, downstream write response, and their interaction. It changes computational specification and composition evidence and is the highest-information next step. It dies if the hybrid interventions fail exact replay or do not distinguish routes.
2. **Restore grouped attention and grouped MLP banks separately.** If the full bank matters, split it into attention versus MLP responses. This is premature if the entire downstream bank has negligible causal contribution.
3. **Translate the persistent residual factor through final normalization and unembedding.** Compare its direct closer-logit effect with the full model effect and then include exact Jacobian or finite counterfactual terms. This connects the circuit to weights, but should follow the grouped causal factorial so it does not mistake a local linear readout for the whole causal path.
4. **Held-out/OOD promotion of the $$\mu/\delta$$ factorization.** Valuable only after downstream use is better specified; opening more data now would validate a factor whose reader remains unclear.

The next action is therefore one compact $$2\times2$$ causal factorial for each of $$\mu$$ and $$\delta$$: upstream factor native/removed crossed with all downstream writes cached from the native/removed run. It is an interaction test motivated by the multiple-mediators problem, not another site or rank scan.
