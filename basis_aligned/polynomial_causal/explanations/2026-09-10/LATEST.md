# Research update since the last requested explanation

## Latest follow-up — 10 September, 17:04 UTC

**The local formula transfers to native states, but its proposed replacement
fails.** Reusing one context gives25–26% selected-response error; borrowing the
next verb's context also fails the joint criterion. Even exact native context
leaves99–101% full-vocabulary effect error when the output is represented by
the two chosen readers. This closes that behavioral replacement proposal while
preserving the exact local equation. [Native test and limits](gerund_scalar_writes_and_live_feedback.md).

**The mathematical review produced a small composable response program.**
Two initialized context scalars predict the last MLP's response to repeated
grammatical-direction edits for two chosen output readers. Direct-weight and
saved-program checks pass. Equal-state context pairs nevertheless differ in
other output responses, giving an 11–24% error lower bound for that restricted
state. This is local conditional extraction, not full-circuit extraction or
natural-text OOD evidence. [Program, proof and limits](gerund_scalar_writes_and_live_feedback.md).

**MLP consumers explain a substantial part of the missing response.** Holding
the fixed grammatical scalar at its original value inside each MLP reduces
target recovery by 29 percentage points in both constructions. It removes 70%
of the agreement damage on the tested base endpoint. The scalar-only predictor
is much closer on this altered computation, but still fails its joint error
bound. This is conditional consumer evidence, not independent extraction or a
selective repair. [Updated math, controls and results](gerund_scalar_writes_and_live_feedback.md).

**New verbs and cues transfer, but broader selectivity fails.** The fixed
unembedding-derived grammatical direction recovers 95%/86% of the new target
effects. Removing it also damages subject–verb agreement by .548 nats, so the
closer control rejects a separately removable gerund circuit. All native
capability and replay checks hold. The [updated explanation](gerund_scalar_writes_and_live_feedback.md)
also spells out the two paths: individual token readers, and shared structured
readers plus their token-specific remainders, folded through the same weights.

The requested attention weight audit and both backward-unembedding views have now run. [The full new explanation](unembedding_token_and_hierarchy_backward_folds.md) defines the fold and reports its limits.

- **Distributed grammatical writes:** swapping the fixed scalar in attention and MLP outputs recovers 94%/102% of the cue effect and passes the registered preservation/removal tests. But a scalar-only predictor misses 65–67% of the full-vocabulary effect: later computations change the rest of the state. This is a useful distributed intervention, not an independently extracted one-scalar circuit. [New backward-source and feedback result](gerund_scalar_writes_and_live_feedback.md).

- **Shared grammatical component:** an unembedding-derived bare/-ing direction transfers 59–62% of the cue effect on held-out verbs; cross-verb cue changes work similarly. Removal damages the target while preserving the registered unrelated control. The 80% sufficiency test fails, and most of the state is carried into the last MLP from earlier computation. [New causal result](shared_gerund_component_from_unembedding.md).

- **Token functions after folding:** no qualifying shared pair among 518 sampled readers. Closest scaled substitutes leave a median 97% quadratic-function error; old cluster means leave 96%. The only close pair was a literal duplicate involving a padded output ID. This rejects proportional whole-token functions as the tested source of new sharing, while shared subterms remain open. [New weight-function analysis](token_readers_as_bilinear_functions.md).

- **QK1/QK2/value test:** the two tasks do not show opposite dependence on the stored score halves. Holding recipient values loses .78–.89 of native cue recovery, while holding either current score loses .013–.066. Contextual values dominate this intervention, but full-vocabulary factor interactions remain large. [New factor analysis](attention_qk1_qk2_value_dependencies.md).
- **Attention input sharing:** in 19 of 26 selected heads, the complementary branch can read the saved scalar’s full value-input function. The same within-head QK factors serve both branches. This is per-head access, not global semantic equivalence; cross-head cancellations matter. [Weight audit](attention_ov_input_reader_overlap.md).
- **Individual token readers:** the exact MLP16–MLP17 fold predicts live vocabulary effects within 4.6–5.6% error on the three reused panels, with CE-change prediction error below .0034 nats. The edited MLP16 contribution slightly opposes the task, so this is not target-circuit extraction.
- **Unembedding hierarchy:** a fixed 16-leaf hierarchy has recognizable groups but its shared means leave 94–98% effect error. Most token-specific response remains necessary. This does not rule out other structured reader decompositions.
- **Prior-work correction:** MLP17 calibration, quadratic output readers and context-gated normalized responses were already documented. The module dossiers and startup now make those checks explicit; the new tests extend prior knowledge.

All native weights remain required. The full four-property goal is still open.

## Consolidated report through 14:17 UTC

**10 September 2026.** This continues the [previous consolidated update](../research_update_2026-09-09.md), which ended at 23:06 UTC on 9 September. It covers the subsequent work through the native experiment completed at 14:17 UTC today, plus the CPU audits of those results. It summarizes the research sequence rather than claiming uninterrupted experimentation throughout that interval.

**We have made progress on explicit computations and causal tests, but we have not found the smaller, independently executable collection of circuits that meets all four requirements.** The strongest new results are an explicit two-consumer test of a previously known calibration scalar, and a partition inside shared attention heads that transfers two different behaviors separately. Both still depend on the original model, and stronger independence tests have exposed limitations.

**Prior-work clarification, 10 September:** calibration, quadratic folding and normalized MLP17 response analysis predate this update. See [the updated module dossier](../MLP17_CURRENT_UNDERSTANDING.md). The new claim is the specific two-consumer test and its later limitations.

## The high-level rundown

The work moved through five stages:

1. **We pursued the original handoff's weight-based ideas beyond the stagnant is/was refinement.** Several exact weight folds worked, but the proposed simpler contextual query and value paths did not explain enough of the real behavior. This established that algebraic correctness and explanatory sufficiency are different requirements.
2. **We tried more explicit behavioral operations.** Noun-number selection and induction copying gave useful causal components, but the clean proposed algorithms or their selective controls failed. We retained the observed components without calling them complete circuits.
3. **We extended an existing calibration-scalar finding from the last bilinear MLP.** Its weight-derived formula predicts a calibration effect, and the same scalar serves both vocabulary scoring and normalization. However, its fitted direction was insufficiently stable and simpler proposed producers failed.
4. **We built a replayable, weight-folded correlative interface.** A correlative is a paired construction such as “both … and” or “neither … nor.” A saved set of directions across 26 attention heads transfers this cue on new word combinations. We can now express and execute its scalar reads and writes directly from the weights.
5. **We tested whether that interface separates into reusable parts.** Its first/local value sources and routing/value operands are individually insufficient. A different split—saved correlative directions versus the rest of the same heads—does transfer two behaviors separately. But selective mean replacement and additive final effects fail.

There was also a material correction: an earlier report's “32/32 pairs kept” counted a measurable cue effect, not correct answers at both endpoints. A matched longer-sentence test exposed this difference. I corrected the interpretation and added an explicit endpoint-accuracy helper.

The overall lesson is more specific than “everything interacts.” We now have executable operations, examples of shared use, and tests that distinguish **information that can be swapped** from **computations that can be independently removed or reused**. The missing work is explaining their contextual producers and downstream consumers well enough to build the simpler program.

## Terms used below

| Term | Meaning here |
|---|---|
| Native model | The original trained model, with its actual weights and operations. |
| Residual stream | The vector carried through the network; attention and MLPs add contributions to it. Its width is 1,152 in bilin18. |
| MLP | A transformation at each token position. This model's bilinear MLP multiplies two learned sets of input features. |
| Producer / consumer | A producer computes a quantity; a consumer reads or uses it. One producer can have several consumers. |
| Interface / port | An explicitly specified place and representation through which a computation reads or writes information. An interface can still depend on opaque native computation. |
| Weight folding | Algebraically combining adjacent linear maps to expose the computation a particular reader uses. It need not reduce the remaining arbitrary weights. |
| Donor swap | Replace a specified recipient component with the component from another sentence, then recompute subsequent layers. |
| Live replacement | Subtract the recipient's **current** component, including changes caused by earlier interventions, before adding the donor component. |
| Mean replacement | Replace the specified component by its saved fitting-set mean. This is our registered removal operation, not a proof of universal feature erasure. |
| Logit / margin | A logit is a next-token score. An answer margin is the intended answer's score minus the alternative answer's score. |
| Cross-entropy, CE | Negative log probability of the correct token. Positive CE change is damage; negative change is improvement. Natural-log units are called nats. |
| OOD | Out-of-distribution relative to a stated test or fitting distribution. New word combinations and a different text corpus are specific shifts, not proof of generalization to every kind of input. |

For recent correlative tests, raw recovery is computed separately for each sentence pair and then averaged:

    recovery = mean((base_margin - patched_margin)
                    / (base_margin + donor_margin)).

Each native margin points toward that endpoint's own intended answer. Thus 1 means reproducing the native cue-induced margin change; it does **not** mean explaining 100% of the model or achieving 100% answer accuracy. Some older reports further divide this recovery by the full selected-head recovery; that is explicitly labeled **normalized recovery**.

We also compare the change across all 50,304 vocabulary scores. The relative error is the Euclidean norm of the candidate-minus-reference effect divided by the reference effect norm, after subtracting each vector's vocabulary mean. A good answer margin can coexist with a poor full-vocabulary prediction.

## 1. What happened before the current correlative work

The original [handoff](../bilinear_circuit_reconstruction_codex_handoff.md) and [pilot report](../bilinear_reconstruction_pilot_report.md) remained the authority. The pilot had already validated alternative execution of bilinear attention and rejected particular local-sharing proposals. We did not treat repeating that recurrence or lowering tensor rank as a new discovery.

The overnight weight analysis considered two circuits reading the same MLP. Their downstream readers can be folded into the MLP weights, giving exact quadratic functions. This made some local writes independently controllable, but stronger native controls and weight-removal tests failed. Other tests ruled out fixed or direct-token-only query descriptions and showed that contextual routing/value interactions could not simply be added away. Those negative results remain recorded in the [weight-tensor report](../weight_tensor_two_circuit_math_2026-09-09.md) and the query/value notes in this folder.

The noun-number investigation supplied a more explicit task: determine which noun's number controls a reflexive. The model did not reliably implement the initially proposed grammatical selector. Later factorial tests—varying several sentence factors in all combinations—found a context-dependent combination of noun-number information. A local mixed-value component changed 16–25% of that measured interaction under removal while largely preserving other tested factors. But direct residual carry did not reproduce its downstream effect. This was partial localization, not extraction of the complete selector. See [noun-number selection](noun_number_selection_and_cross_token_interactions_2026-09-10.md).

The induction investigation tested “find an earlier occurrence and copy what followed it.” We separated source selection from the carried payload. Selector swaps recovered roughly 86–91% of the mean answer-margin change and payload swaps 77–86%, but joint answer-preserving edits and filler controls failed. Replacing the **current** contribution rather than adding a difference from an old cached state improved some loss measurements, yet the broader vocabulary and control failures remained. That fixed interface was closed rather than promoted. See [selector–payload tests](induction_selector_payload_interchange_2026-09-10.md) and [live replacement](induction_live_replacement_and_context_2026-09-10.md).

## 2. A positive weight-derived computation: one scalar, two uses

The last bilinear MLP supplied a useful intermediate result. Write its computation as

    M(u) = D[(L u) * (R u)] + b.

Here u is its normalized input, L and R produce learned features, * multiplies matching entries, D writes the products back to the residual stream, and b is a bias. For a chosen output direction w, its scalar coefficient is

    c(u) = wᵀ M(u) / (wᵀ w) = uᵀ Q u + beta,
    Q = sym(Lᵀ diag(Dᵀw / (wᵀw)) R).

`diag` places a vector on a matrix diagonal; `sym(A)=(A+Aᵀ)/2`. This is an exact local weight contraction, up to floating-point evaluation differences. It exposes which input-pair products create the scalar without constructing a huge three-index tensor.

Removing this scalar increased rare-token loss by about0.500 nats on held-out FineWeb rows and 0.573 on Pile, while improving frequent-token loss by 0.287 and 0.213 nats. Pile was a corpus shift, not certified absent from training. The scalar was initially proposed using frequency correlation, but the simple “it is a log-frequency bias” interpretation failed a sanity test.

The scalar has two consumers. It changes the vocabulary-score numerator and the shared normalization denominator. If h=g+cw, the denominator contains

    mean((g+cw)²) + epsilon
      = mean(g²) + 2c*mean(g*w) + c²*mean(w²) + epsilon.

Therefore changing c rescales the retained background as well as changing its direct vocabulary contribution. The joint readout formula reproduced native removals; either consumer alone was insufficient. Replacing c by a constant or another text's value also hurt prediction, showing that its pairing with context mattered.

The limitation is substantial: separately fitted directions had similar geometry but insufficiently stable causal effects. Direct-token, reader-energy and selected bilinear-pair simplifications also failed. We kept the original conditional result without declaring its semantic identity or independent producer solved. See [the two-consumer scalar](calibration_scalar_and_two_consumers_2026-09-10.md) and [stability follow-up](calibration_context_dependence_and_stability_2026-09-10.md).

## 3. The current positive: an explicit correlative read/write interface

At13:09 we reconstructed and saved the existing correlative fitting recipe because the older report had not preserved its fitted tensors. The new fit reproduces the old summary metrics, but that does not prove identity with the missing old tensors. We now save the actual head order, directions, means and rows for replay.

There are26 selected heads across 14 layer blocks, with one scalar direction per block. **This is 14 scalars, not one scalar for the whole model.** Folding those directions into the native value and output matrices gives

    c_l = sum over selected heads h and source tokens j:
          attention_score(l,h,j)
          * [local_value_read(l,h,j) + first_layer_value_read(l,h,j)],

    residual_edit_l = output_writer_l * (donor_c_l - live_c_l).

The actual normalization, positional rotation, signed value mixing and native context remain explicit. The folded executor reproduced the original donor and mean interventions to floating-point tolerance. On new reporter/noun combinations relative to the original source bank, full raw recovery was0.969 in the bare frame and 0.899 in the report frame, with small registered controls.

This is meaningful progress on an executable interface. It is not independent token-to-answer extraction: the original network still produces its contextual inputs and executes its consumers. The folded core contains 76,032 coefficients, alongside all 545,902,902 native parameters. There is no achieved whole-model structural saving. See [replay and folding](correlative_replay_and_folded_program_2026-09-10.md) and [new combinations](correlative_new_combinations_and_value_sources_2026-09-10.md).

## 4. What the attempted simplifications taught us

First, the first-layer value contribution and the current-layer contextual value contribution were each insufficient alone. Keeping their exact sum reproduced the full operation, but adding their separately measured final effects did not.

Next, we separated routing from value content. Let P be attention scores and U the complete scalar value reads, so c=sum(P*U). Swapping donor P or donor U separately gave:

| Operation | Bare-frame recovery | Report-frame recovery | Relative vocabulary error, bare/report |
|---|---:|---:|---:|
| Complete routing–value swap | .969 | .899 | 0 / 0 by definition |
| Routing only | .159 | .119 | .829 / .860 |
| Values only | .853 | .780 | .209 / .221 |

Neither partial operation met the registered combination of recovery>=.8 and vocabulary error<=.15 in both frames. Values carry most of the target effect, but “most” is weaker than sufficiency.

The math makes the missing term explicit:

    delta_c = sum(delta_P * U_base)
            + sum(P_base * delta_U)
            + sum(delta_P * delta_U).

The third term is the routing–value interaction. In a norm that weights each scalar by its actual output writer and stacks the separate block writes, omitting it loses38% and 30% of the native write change in the two frames. This is a local/native-state quantity. It is not a percentage attribution of the final behavioral effect, because multi-block interventions follow different live trajectories. See [the routing–value report](correlative_routing_value_product_2026-09-10.md).

We also tested a matched longer construction containing “either the rope or the lamp,” compared with “only the rope or the lamp.” Cue position, token count and final token were fixed. The native model often still preferred “and” after “neither”; the bare frame had 0/16 pairs correct at both endpoints in either context. Earlier positive-effect counts could not establish the claimed grammatical capability. We retained that failure and corrected the metric interpretation. See [context math and capability correction](correlative_context_math_and_capability_correction_2026-09-10.md).

## 5. Two behaviors use different parts of the same heads—but not independently in every sense

The latest experiment split the selected head coordinates into the saved directions and their orthogonal remainder. For a unit direction v, the projector is Π=vvᵀ and the remainder is I−Π. Their sum is the identity. The remainder has 3,314 coordinates and remains a large unexplained object.

The donor-swap results are the clearest recent answer to your shared-module question:

| Behavior | Saved directions | Orthogonal remainder |
|---|---:|---:|
| both/neither, bare frame | .969 recovery | −.004 |
| both/neither, report frame | .899 | .052 |
| either/not → or/but | −.025 | .982 |

Each part transfers one behavior strongly and the other weakly. This is a **double dissociation under swapping**: evidence for distinct information within the same native heads.

But mean replacement fails selective preservation. Removing the saved directions changes the disjoint task by an average absolute 0.172 nats, above the 0.1 limit. The signed average is only +0.051 nats because 10 examples worsen and 6 improve. The report-only CPU audit decomposes this exactly:

    average harm          = 0.1115 nats,
    average improvement   = 0.0603 nats,
    signed average change = harm - improvement = 0.0513,
    absolute average      = harm + improvement = 0.1718.

Those harm/improvement averages divide by all 16 examples. Replacing the remainder also changes the bare correlative task too much under the preservation criterion, mostly through improvement rather than damage. We should not misdescribe that as large average harm; it is a failure to leave the other behavior stable.

The joint final-vocabulary effect also differs from the sum of the singleton effects by 21%, 27% and 12% in the three panels. Orthogonal coordinates do not guarantee independent downstream consumers or additive final behavior. This does not rule out compositionality: a correct composed program could explicitly compute the interaction. We have not yet extracted that program. See [shared heads and selective swaps](correlative_shared_heads_selective_swaps_2026-09-10.md).

## Did the three-hour mathematical cycles help?

**Yes, mainly by making the right computations and falsifiers available. They have not yet produced the desired complete discovery.** There were five scheduled mathematical reviews after the previous consolidated report:

| Review, UTC | Concrete contribution | What it changed |
|---|---|---|
| [01:49](../../THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-10_0149.md) | An exact contextual-query source expression retaining both normalizations. | Enabled a 19-source causal atlas. No single source explained a complete panel; adding separate omission effects missed the joint effect by 73–81%. |
| [04:49](../../THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-10_0449.md) | Correct transport for rounded position rotations, plus a metric for overlapping downstream readers. | Prevented treating an ideal rotation identity as exact in deployment, and distinguished answer-direction errors from the rest of a head's vocabulary effect. |
| [07:49](../../THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-10_0749.md) | Separated a bilinear MLP's newly created product interaction from interaction already present in its inputs. | Directed an actual source intervention rather than another generic normalization/Gram-matrix replay. It did not justify calling one source the whole circuit. |
| [10:49](../../THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-10_1049.md) | An exact counterexample distinguishing a shared producer from its consumer-specific ports. | Showed why a perfectly factorized copying computation can still fail full-output invariance when another consumer reads source context. Our induction failure remained a failure. |
| [13:49](../../THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-10_1349.md) | A four-corner lower bound for additive cue/context representations. | Directly tested and rejected additive separation of the fixed 14 correlative coordinates on the matched contexts. |

The last result is especially simple. If a scalar were `cue_part(c)+context_part(k)`, then

    D = s(c1,k1) - s(c0,k1) - s(c1,k0) + s(c0,k0)

would be zero. On exactly these four inputs, the best additive approximation has squared error D²/4; some corner must have error at least |D|/4. All 448 tested layer/example contrasts exceeded their measured folding discrepancies. This rules out that additive description of the fixed coordinates, not every nonlinear encoding or alternative circuit.

Other mathematical work also helped between scheduled cycles. A trained-weight counterexample showed that identical folded query–key numerator tensors can still produce 53–55% different normalized attention scores. The missing normalization terms matter to whether two computations are genuinely shared. An exact memory compiler also made the pilot's distinction operational: shared feature arithmetic can require separate editable histories for different consumers. Neither control alone counts as trained-model circuit discovery.

The candid assessment is that the math has improved the **quality of the questions and the validity of the conclusions** more than it has improved independent extraction. It has ruled out misleading shortcuts and enabled the current weight-folded interventions. We should judge subsequent cycles by whether they identify and test a concrete producer/consumer operation, not by the number of formulas or review files.

## Where this leaves the four properties

| Property | New evidence since the previous update | Still missing |
|---|---|---|
| OOD prediction | Correlative transfer on specified new word combinations; calibration effects on a corpus shift. | Broad prediction of unseen constructions from an explicit extracted operation. The longer correlative construction failed native capability. |
| Extraction | Exact weight-derived scalar producers/readers and faithful conditional intervention executors. | Independent input producers and consumers with a smaller structural description. |
| Removal/interchange | Strong donor transfer and a shared-head double dissociation. | Selective mean removal failed; an intervention map satisfying the full intended scope remains unproven. |
| Composition/reuse | A scalar with two explicit readout uses; exact coupled routing–value operation; distinct information inside shared heads. | Independently explained reusable subcomputations and correct joint execution across the required contexts. Non-additive effects must be modeled, not ignored. |

All 545,902,902 native parameters remain in the current full-model executors. Fewer bytes, a saved direction, or a renamed large remainder would not satisfy the appended requirement for fewer independently specified computations.

The most recent native screen finished at 14:17. The next downstream-MLP investigation has begun with prior-art reading and derivation only: it asks how the first MLP consumer combines the two writes, and how much interaction comes from multiplication versus normalization. No new protocol, queued GPU experiment or native result for that question exists yet. The already available four-corner algebra can be reused rather than rebuilt.

The 14:14 hourly review is complete. It recorded four managed receipts in the preceding hour, with a median gap of 15.05 minutes—above the 10-minute target—even though the recent GPU executors each took about 1.6–1.9 seconds. Authoring, reasoning and reporting dominate turnaround. We now reuse the intervention executors, explicit endpoint checks and paired-bootstrap helper. Bootstrap intervals resample the authored groups; they describe uncertainty in the finite panel, not universal generalization.

The next scheduled checkpoints are 15:14 for strategy and 16:49 for mathematics. Both managed runners were healthy at the latest check. The durable research goal remains open.
