# Hourly circuit-only review — 2026-09-05 17:18 UTC

## Controlling goal and circuit targets

The controlling goal remains a reusable pipeline for hundreds of nonduplicated causal circuits, followed by a smaller transparent tensor program built from their shared structure. A mature circuit record should state what information is read, what operation is performed, what is written, and which later computation uses it. It should predict held-out or OOD behavior, support extraction or sufficiency, allow selective removal or interchange without unrelated damage, expose reuse and composition, and remain stable under relevant data changes and gauge freedoms.

Native attention heads and whole MLPs remain localization handles rather than assumed semantic units. Rank, quantization, activation variance, and activation reconstruction are not circuit evidence and were not used as scientific targets this hour. Task CE and answer margins were retained alongside exact causal interventions and lexical controls.

## What changed in the preceding hour

The Task14 subject-number circuit was split twice below the MLP8 module boundary.

First, the raw pre-MLP8 subject state was decomposed as

$$
x=E+A+U+V,
$$

where $E$ is embedding/skip history, $A$ is attention history, $U$ is MLP0--3 history, and $V$ is MLP4--7 history. A complete $2^4$ intervention factorial showed that $V$ is the stable dominant MLP source for the previously identified embedding--MLP and attention--MLP interactions. The winner is the same in both grammatical directions and both CE/margin. The valid result was published as canonical Task14 v20.

Second, $V$ was split as

$$
V=W+X,
\qquad
W=\mathrm{MLP4}+\mathrm{MLP5},
\qquad
X=\mathrm{MLP6}+\mathrm{MLP7},
$$

while $E$, $A$, and $U$ remained live factors in a complete $2^5$ factorial. Neither half is a global dominant winner; the preregistered distributed outcome holds. $X$ is larger in almost every parent aggregate, but the full-response $W\times X$ interaction is strongly direction dependent: about $-23\%$ for plural-to-singular and $+49\%$ for singular-to-plural. Therefore $W$ cannot be dropped. This result is canonical Task14 v21, with the v13--v21 publisher suite at 47 passing tests.

Claude independently added `preposition_selection.on_vs_of`, a valid selective causal site at `resid:17`. Together with the animacy capability null, this prospectively strengthens the current boundary: seven grammatical/function-word behaviors have passed the shared screen, while three lexical-semantic behaviors have failed cross-construction capability. These are separate circuit-family results, not evidence for MLP8.

## Throughput and systems audit

Between 16:17 and 17:18 UTC, eight distinct terminal receipts landed:

1. OOD Task14 MLP8 native capability: pass;
2. OOD Task14 MLP8 polarized intervention: valid causal screen;
3. animacy cross-construction capability: honest null;
4. fresh Task14 MLP8 E/A/M input-source factorial: valid causal screen;
5. preposition selection: valid causal screen;
6. Task14 E/A/U/V float32 regrouping attempt: numerical invalid, preserved separately;
7. repaired Task14 E/A/U/V depth factorial: valid causal screen; and
8. Task14 E/A/U/W/X MLP4--7 factorial: valid causal screen.

This is one terminal per 7.6 serial minutes. The causal GPU runs took roughly 6--12 seconds. The dominant cost was scientific authoring and exact replay bookkeeping, not compute. The E/A/U/V runner exceeded its intended serial budget because two required parent replay fields (`HR`, then `M0_3/MR`) were omitted and a later float32 regrouping changed a known parent corner. The repaired shared pattern now has three explicit rules: preserve the complete downstream replay-state contract, use authoritative parent aggregates for same-role corners, and form a new grouped sum only for genuinely mixed counterfactuals. The next runner's focused tests directly cover those rules.

No broad backup suite was added. The two execution failures are engineering events rather than scientific nulls; the float32 result is separately labeled invalid and its thresholds were not relaxed.

## Confounds and falsifiers

- **Nonlinear outcomes:** Boolean-lattice Möbius terms in CE and answer margin include RMSNorm and downstream nonlinear computation. They are causal set-function interactions, not direct coefficients of MLP8's weight tensor.
- **Signed cancellation:** inclusive source recoveries can exceed 100% or become negative because suppressive interactions are retained. They are not probabilities.
- **Cross-world states:** donor/recipient source mixtures are explicit path interventions. Same-role corners must reproduce native parent corners exactly; mixed corners are not claimed to occur naturally.
- **Lexical leakage:** every factorial includes the complete same-number/different-noun control. The latest maximum is 23.4%, below the frozen 25% number-specificity bar but close enough that the next split must retain it unchanged.
- **Direction averaging:** the large sign change in the $W\times X$ full-response interaction would disappear under an average over transfer direction, so all next predictions remain bidirectional.
- **Post-selection:** MLP6--7 is chosen because the frozen $2^5$ result made it the larger distributed contributor. MLP6-vs-MLP7 alternatives and thresholds are frozen before their outcomes.
- **Control liveness:** native capability is hash-bound to all registered direction/template cells, exact parent corners can fail independently, and the same-number lexical donor is capable of producing a large effect. These are not loose always-pass thresholds.

## Strategic alternatives

The current recursive interaction split remains highest information because it directly improves within-module splitting, computational specification, and later weight translation. Three alternatives remain live but are ordered behind it:

1. Contract the causally identified input differences through MLP8's $L$, $R$, and $D$ tensors. This can translate a causal source into weights, but doing it before MLP6/MLP7 localization risks explaining a mixture.
2. Use downstream equivalence across the growing circuit corpus to group MLP8 output directions that later readers treat identically. This targets reuse and gauge-stable identification, but requires more completed circuit-response signatures.
3. Promote the grammatical-versus-lexical fast-screen boundary with new held-out constructions. This broadens the circuit corpus and continues in Claude's lane without duplicating the Task14 decomposition lane.

The recursive route is killed if MLP6 and MLP7 remain distributed with a large signed interaction that cannot be isolated, or if lexical controls cross the frozen bar. In that case the sibling pair becomes the causal unit and the next action switches to downstream-equivalence or exact weight contraction rather than another marginal layer split.

## Gates and continuation receipt

`CIRCUIT_FOCUS: PASS.` The hour added five valid causal screens, one circuit-family null, two canonical Task14 revisions, and a finer within-MLP interaction graph.

`CEREMONY_BUDGET: PASS WITH ENGINEERING CORRECTION.` Scientific GPU runtime remained seconds. Repeated replay-contract omissions cost several minutes, so the next implementation reuses the corrected runner and tests the complete state contract instead of adding more checks.

`NOVELTY_LESSON_GATE: PASS.` The Task14 dossier, current-value depth result, equality MLP dossiers, claims, invalid receipts, and canonical registry were searched. The new interventions occur at raw pre-MLP8 input and do not repeat later current-value localization, rank reduction, or reconstruction work.

The concrete continuation is active: `subject_verb.number_agreement.head11_3_fresh_matched_subject_mlp8_mlp6_7_source_factorial_v1` is claimed, its prior-art receipt and derivative capability license are committed, and its runner implementation is underway. It retains $E$, $A$, $U$, and $W$ while splitting MLP6 from MLP7 in a complete $2^6$ factorial. Frozen price: 4 model forwards, 12,224 example evaluations, 6,048 interventions, no gradients, and no parameter updates.
