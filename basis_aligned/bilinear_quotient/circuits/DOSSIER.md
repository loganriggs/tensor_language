# Circuit dossier — bilin18

Assembled from the frozen census (source note: 2026-08-30 by Claude, circuit task (Logan)) plus current version-2 records. **62 census response regions and 6 task-defined behavior circuits/shared subroutines**. Each census region was localised by two independent causal interventions over the 256,000-position census grid.

`concentration` = mean|dCE| on the circuit's members / mean|dCE| off its slice, when the named component is ablated. **mean** replaces the component output with its grid mean; **interchange** replaces it with its output at a random other position (seed 20260830).

Sources: `circuits/BATTERY.json` (localisation), `circuits/DAS.json` (learned subspace, where run), and each circuit's own file (story, examples, certification). Nothing here is recomputed.


## Behavior circuits, shared subroutines, and counterfactual identification

These version-2 records are task-defined behaviors or cross-module subroutines, not assumed aliases of census leaves. Their events include failed/null/invalid evidence so the same causal question is not silently repeated.

| circuit | kind | status | declared variable | families | negative events | next missing evidence |
|---|---|---|---|---:|---:|---|
| `task.subject_verb_number_agreement` | behavior_circuit | program_screened | `complete_subject_number_at_final_position` | 4 | 7 | retain the interface-simple 10-vector upstream + two-reader program and its asymmetric cross-program robustness, but do not call it end-to-end simple; compile or remove its externally supplied selector/context dependencies; fixed mediator gains and possessive reuse are closed nulls, and rank/reconstruction sweeps remain closed |
| `subroutine.induction.equality_score` | shared_subroutine | site_live | `cross_head_equality_score` | 5 | 5 | materialize the text-edit and matched-natural answer-changing families plus the payload-preserving invariance family; then measure complete-state query/key/MLP7 ceilings with identical patch semantics before fitting a shared subspace |
| `task.bracket.pending_opener` | behavior_circuit | program_screened | `pending_opener_state_three_value_candidate` | 5 | 9 | retain the OOD-screened six-vector L13H8 ordered-pair program and norm-matched cross-program robustness; next compile or remove its externally supplied ordered-pair selector; retain the pair-centered selective-necessity null |
| `task.increment.state` | behavior_circuit | proposed | `increment_state` | 4 | 0 | freeze cross-format rows; require number-word transfer and nonincrement numeric controls |
| `task.induction.selector_payload` | behavior_circuit | proposed | `induction_selector_payload` | 5 | 1 | freeze two-valid-source and payload-swap rows; measure selector and value site ceilings |
| `task.successor.pointer` | behavior_circuit | proposed | `successor_pointer_state` | 4 | 2 | expand families and test shared-plus-private projectors against failed cross-family transfer |

### `task.subject_verb_number_agreement` — path_grouped

**Read:** the grammatical number of the complete subject, including a nearby prepositional phrase or relative clause. **Operation:**
carry that number to the final-token prediction and combine it with the copula choice. **Write:** evidence favoring ` is` or ` are`.
**Endpoint:** signed answer-versus-foil logit-margin recovery under donor interchange.

| family | role | status |
|---|---|---|
| opposite-number transfer in a prepositional phrase | interchange | measured |
| opposite-number transfer in a relative clause | interchange | measured |
| noun identity with answer held fixed | control | measured |
| attractor number with answer held fixed | control | measured |

**Current causal picture:** attention head 11.3 carries about 60% of the donor answer effect in both syntactic constructions. Later
attention modules contribute little to the downstream response. MLP11–17 collectively change the transported effect, and within the
late path MLP15 and MLP17 form a useful causal group: together they reproduce the MLP15–17 effect within 11.6% relative error, MLP16
contributes only 0.56% RMS, and the MLP15-by-MLP17 interaction is only 0.655% RMS. This groups module contributions to this task; it
does not yet identify the internal features in either MLP.

The L11H3 source-side interface is now more explicit. Grouped MLP6/7 E/A/U/W factor replacements form a complete 16-subset causal
lattice. A target-free midpoint JVP predicts exact-minus-base margin amplitudes prospectively at cosine `>0.99999999`; one central
row reader predicts all 512 lattice effects at cosine `0.99996`. Two fixed 1,152-D direction readers then transfer to a disjoint corpus
without target-tail execution at cosine `0.96983`, and reader-chosen gains set a requested signed `0.04` margin edit with median
absolute error `0.01013`.

A fixed upstream program has also passed a prospective screen. Ten 1,152-D displacement vectors indexed only by answer direction and
factor cardinality were exported from the second corpus and installed on 32 new noun forms in unseen near/beyond syntax. Their causal
effects substitute each target text's native displacement at cosine `0.86301` with `0.93945` sign agreement; the frozen readers predict
the installed effects at cosine `0.96927` with perfect signs. Cardinality reduces SSE `29.72%` versus a two-direction control. All ten
literal writes preserve frozen numbered-list and bracket behaviors (worst median normalized collateral `0.000715`, zero answer flips)
after a documented FP32 tripwire repair. This is a **screened reusable interface/program**, not adoption: broader selectivity and joint
composition/reuse remain open.

Cross-task reuse is bounded by a valid null. The preselected cardinality-0 writes do not transfer to adjacent possessive agreement with
`their`/`his` outputs: median donorward change is `0.000499`, one direction×construction cell has chance-level sign, and the frozen
correct-versus-swapped advantage misses. Thus the current operational unit is Task14/copula-specific at this interface, not a generic
local grammatical-number state. No alternate cardinality or scale is licensed on the opened possessive rows.

The fixed program now has an explicit downstream composition boundary. Clamping MLP15+17 to their matched base-program outputs removes
`21.63%` of the full program-effect norm with cosine `0.95849` and perfect signs; this recurs in every direction×template and cardinality
group. An exact singleton factorial shows a distributed, approximately additive pair: MLP15 carries `29.52%` and MLP17 `92.02%` of the
joint-mediation norm, while their cancelling interaction is `21.61%`. Both singleton effects remain material in all four
direction×template groups and all five cardinalities, so the evidence does not license dropping MLP15 even though MLP17 dominates.
The mediator effects themselves admit a smaller operational readout: six direction-only scalar gains applied to the frozen reader
predict row-held-out MLP15, MLP17, and interaction responses, reconstructing joint mediation at cosine `0.95337`, relative L2
`0.30185`, and perfect signs. A 30-scalar direction×cardinality extension improves SSE by only `0.1018%` and is rejected. However,
the six gains fail a temporally sealed fourth corpus: joint cosine/error are `0.89345/0.48050`, and the apparent overall MLP15 pass
collapses in singular-to-plural under/beyond syntax (cosine `0.06627`, signs `0.42969`). The gains remain a retrospective screen only;
the ten-vector upstream program and fixed reader themselves continue to transfer there at cosine `0.94456` with perfect signs.

Selectivity is now broader but still empirical rather than universal. All ten writes preserve four additional native-capable behaviors—
polarity, narrative tense, preposition selection, and voice frame—under per-behavior/per-write bars. Across those 40 new cells, the
worst median normalized effect is `0.00405` and there are zero answer flips. Together with numbered-list and bracket panels this gives
six measured collateral behaviors. The first broad receipt's `5e-5` FP32 tripwire was exceeded by `5.984e-5`; an append-only,
zero-rerun audit at the repository's existing `1e-4` activation tolerance preserves all scientific values.

Literal pricing now separates selected runtime state from controls. The selected ten vectors contain `11,520` FP32 scalars and the two
readers contain `2,304`, for `13,824` scalars (`55,296` bytes) total; the two direction-only vectors in the export are controls and add
`2,304` scalars only to the research artifact. A causal install costs `1,152` additions; including the optional dense reader costs
`3,455` scalar arithmetic operations. This saves `64.706%` of storage versus a literal 32-vector direction-by-subset table with the same
readers (`68.75%` for vectors alone). But the current harness still needs externally supplied direction and intervention cardinality,
counterfactual role prompts, native contextual base construction, and the native suffix/logit path. It eliminates **zero** native blocks
and **zero** native parameters. The licensed verdict is therefore `interface_simple_not_end_to_end`, not runtime acceleration or a
standalone model. This also corrects the 23:35 reviews' arithmetic: their `13,824` upstream-vector count included the two controls; the
selected upstream vectors are `11,520`, and `13,824` is the selected vectors-plus-readers total.

Cross-program composition is asymmetric. In an exact four-corner stress test with the independently frozen bracket program, bracket
vectors are strongly live on the Task14 panel (`2.2515×` the isolated Task14-effect norm), yet Task14 effects are preserved at cosine
`0.99983`, relative L2 `0.01863`, and perfect signs; the interaction is only `1.863%` of Task14-effect norm. This one-sided gate is a
valid composition screen. The reverse is not licensed: Task14-vector stress on bracket prompts is only `0.0004896×` bracket-effect norm,
so its excellent numerical preservation is vacuous. The parent two-sided verdict remains inconclusive, and neither result establishes a
single natural prompt that jointly instantiates both semantic variables.

An outcome-free norm-matching audit resolves only the reverse panel's robustness question. Scaling the Task14 direction-cardinality
vectors by the frozen median bracket-vector/Task14-vector norm ratio (`137.6339`) makes foreign stress material on bracket prompts
(`0.31572×` isolated bracket-effect norm). Under that stress the bracket program retains cosine `0.98621`, relative L2 and
interaction/own norm `0.17838`, norm ratio `0.91971`, and perfect signs across all 144 endpoints. This is a robustness screen at a
declared artificial norm-matched scale. It does not upgrade the original-gain reverse test, establish a natural jointly instantiated
prompt, or license arbitrary scaling as part of either program.

The two programs are now compiled into one hash-bound dispatcher package. It contains exactly the ten selected Task14 vectors and six
bracket vectors (`18,432` FP32 scalars, `73,728` bytes), reproduces dispatch for all 512 prospective Task14 cells and 144 OOD bracket
targets, and returns exact zero for all 216 licensed bracket self-pair controls. Selection uses only
`(recipient_number, donor_number, cardinality)` or `(recipient_closer_id, donor_closer_id)`; it no longer needs a recipient/donor role
prompt, row, template, family, or model call to choose a vector. This removes prompt-lookup ceremony, not the substantive boundary: an
external intervention specification, native base activation, and native downstream suffix/logit path are still required. The package
is a compiled intervention dispatcher, not an autonomous predictor.

A retrospective cross-construction diagnostic shows that bracket donorward effects may admit a much smaller suffix-free readout. Six
ordered-pair scalar means transferred symmetrically between the direct-type and completed-then-reopened OOD families at cosine
`0.98181`, relative L2 `0.19025`, norm ratio `0.99373`, and perfect signs across 144 endpoints. The two directional folds separately
reach cosine `0.99767` and `0.98960`, again with perfect signs. This costs six FP32 scalars and a lookup, but by itself was feasibility
evidence on already opened outcomes, not a promoted predictive program; the following untouched-corpus test supplies the required seal.

That prospective test now passes on a third construction containing an identical completed distractor delimiter before the active
pending opener. Six scalars frozen from the original direct-type family predict all 72 fresh program effects at cosine `0.99626`,
relative L2 `0.10378`, norm ratio `1.05380`, median absolute error `0.46449`, and perfect signs. Native accuracy and positive program
effects are `1.0` overall and in every ordered pair; every pair's median absolute error is at most `1.03691`. Thus the bracket package
has a prospective six-scalar reader for donorward program-effect magnitude without native suffix execution. It still needs the native
base activation to install its displacement and does not predict full logits.

A second prospective dependency reduction replaces the native L13H8 opener term rather than adding a displacement to it. Three
1,152-D closer-conditioned absolute terms were exported from balanced SELECT endpoints and installed directly on 72 untouched endpoints
in a fourth construction. They reproduce exact within-row donor-term swaps at cosine `0.99876`, relative L2 `0.11076`, norm ratio
`0.89979`, and perfect signs; native accuracy and positive program effects are `1.0` overall and in every ordered pair. The edited
replacement does not read the recipient's native opener term and stores only `3,456` FP32 scalars. This removes that one local base-term
dependency, not all upstream context or the native suffix needed for causal execution. The earlier pair-centered selective-necessity null
remains unchanged.

The absolute program also survives a fresh-intervention audit on both established OOD target families: cosine `0.99654`, relative L2
`0.09265`, norm ratio `1.03749`, and perfect signs across 144 endpoints. The direct and completed-then-reopened families separately reach
cosine `0.99853` and `0.99612`; all six ordered pairs have positive-effect and sign-agreement fractions `1.0`. All 216 answer-preserving
endpoints dispatch their native term unchanged, with exactly zero maximum logit change and no answer loss. This licenses replacing the
six displacement vectors with three absolute closer terms in the compiled bracket interface while retaining the no-edit path.

The consolidated predictive/manipulable package now stores ten Task14 displacements, ten compile-time reader effects, three bracket
absolute terms, and six bracket effect scalars: `14,992` FP32 scalars (`59,968` bytes). Compiling Task14's two dense readers into ten
fixed dot products is exact within `7.03e-9`; exhaustive dispatch covers ten Task14 edits, six bracket edits, and three bracket no-edits.
Relative to the previous complete operational inventory (both older vector banks, Task14 readers, and bracket scalars), storage falls
`27.7215%`. Effect prediction for both programs is specification-only and suffix-free. Causal execution still needs model execution to
the installation site and afterward; Task14 alone still reads its native L11H3 base term, while bracket needs the semantic opener position.

The direct Task14 absolute-term analogue is a decisive null. Ten second-corpus direction-by-cardinality absolute L11H3 heads fail across
all 512 third-corpus cells: cosine `0.19725`, relative L2 `2.72039`, norm ratio `2.73485`, and sign agreement `0.59375`. One
direction-by-template cell is anticorrelated (`-0.19765`), no cardinality passes, and closure error is exactly zero. Thus Task14's native
base head contains indispensable prompt context that cannot be removed by a ten-entry absolute table. This does not weaken the
displacement or scalar-effect program; it bounds causal execution. The next allowed compression is reader-aligned scalar context, not a
post-hoc template table or unconstrained rank sweep.

Keeping only the native base deviation along the already frozen direction-specific reader is also a registered null, though informative:
overall cosine improves to `0.68452`, relative L2 to `0.76755`, norm ratio to `0.92471`, and signs to `0.82813`, but it misses the frozen
overall bars and two direction-by-template recurrence cells. The reader-coordinate identity holds within `5.96e-8` and all closures are
zero, so this is a scientific failure rather than an instrument failure. One scalar captures substantial output-relevant context but not
enough for causal execution. The only licensed next extension is the two-dimensional span of both pre-existing frozen readers; no learned
rank, new direction, template table, or threshold change is allowed.

**Append-only evidence ledger:**
| event | stage | test | verdict | lifecycle | result artifact |
|---|---|---|---|---|---|
| `agreement_head11_3.legacy` | complete | mean replacement | **non-necessary/redundant** | active | explanation §1548 |
| `agreement_full_state.v2` | complete | donor interchange | **held** | active | `task14_subject_verb_agreement_full_state_v2_result.json` |
| `agreement_cross_syntax.v1` | complete | cross-syntax interchange | **held** | active | `task14_subject_verb_agreement_cross_syntax_v1_result.json` |
| `agreement_head11_3_complement.v1` | complete | head/complement interaction | **held** | active | `task14_attention11_head_complement_factorial_v1_result.json` |
| `agreement_downstream_modules.v1` | complete | single-module restoration | **inconclusive** | active | `task14_head11_3_downstream_module_reader_screen_v1_result.json` |
| `agreement_attention_vs_mlp_path.v1` | complete | path restoration | **MLP path held** | active | `task14_head11_3_attention_mlp_path_factorial_v1_result.json` |
| `agreement_mlp11_12_vs_13_17.v1` | complete | grouped MLP interaction | **inconclusive** | active | `task14_head11_3_early_late_mlp_factorial_v1_result.json` |
| `agreement_mlp13_14_vs_15_17.v1` | complete | grouped MLP interaction | **inconclusive** | active | `task14_head11_3_late_mlp_halves_factorial_v1_result.json` |
| `agreement_mlp15_17_vs_16.v1` | complete | grouped MLP interaction | **MLP15+17 held** | active | `task14_head11_3_mlp15_17_vs_mlp16_factorial_v1_result.json` |
| `agreement_mlp15_by_mlp17.v1` | complete | exact derived interaction | **additive** | active | `task14_head11_3_mlp15_mlp17_interaction_v1_result.json` |
| `agreement_head11_3_projector_program_a.v1` | complete | learned causal subspace | **instrument invalid** | active | `task14_head11_3_causal_projector_program_a_v1_receipt.json` |
| `agreement_mlp6_7_background_composition.v1` | complete | factor composition and continuous manipulation | **held** | active | `task14_fresh_fronted_mlp6_7_background_composition_transfer_v1_result.json`; `task14_fresh_fronted_mlp6_7_continuous_background_gain_manipulation_v1_result.json` |
| `agreement_mlp6_7_midpoint_reader.v1` | complete | sealed prospective causal prediction | **held** | active | `task14_prospective_mlp6_7_downstream_midpoint_margin_jvp_amplitude_v1_result.json` |
| `agreement_mlp6_7_complete_lattice_reader.v1` | complete | 16-subset composition prediction | **held** | active | `task14_prospective_mlp6_7_single_reader_full_lattice_v1_result.json` |
| `agreement_mlp6_7_fixed_direction_readers.v1` | complete | no-target-tail cross-corpus extraction | **held** | active | `task14_fixed_direction_reader_cross_corpus_transfer_v1_result.json` |
| `agreement_mlp6_7_reader_guided_edit.v1` | complete | absolute signed manipulation | **held** | active | `task14_fixed_reader_guided_margin_edit_v1_result.json` |
| `agreement_mlp6_7_upstream_cardinality_program.v1` | complete | prospective upstream substitution | **screen** | active | `task14_mlp6_7_direction_cardinality_prototype_causal_validation_v1_result.json` |
| `agreement_mlp6_7_program_collateral.v1` | invalid | projected-write selectivity | **instrument invalid** | superseded by v2 | `task14_mlp6_7_direction_cardinality_prototype_cross_circuit_collateral_v1_result.json` |
| `agreement_mlp6_7_program_collateral.v2` | complete | projected-write selectivity | **held narrowly** | active | `task14_mlp6_7_direction_cardinality_prototype_cross_circuit_collateral_v2_result.json` |
| `agreement_mlp6_7_possessive_reuse.v1` | invalid | cross-task upstream reuse | **native gate invalid** | superseded by v2 | `task14_mlp6_7_cardinality0_upstream_cross_task_possessive_reuse_v1_result.json` |
| `agreement_mlp6_7_possessive_reuse.v2` | complete | cross-task upstream reuse | **null** | active | `task14_mlp6_7_cardinality0_upstream_cross_task_possessive_reuse_v2_result.json` |
| `agreement_mlp6_7_program_mlp15_17_mediation.v2` | complete | fixed-program downstream mediation | **held** | active | `task14_mlp6_7_direction_cardinality_program_mlp15_17_mediation_v2_result.json` |
| `agreement_mlp6_7_program_mlp15_vs_17.v1` | complete | exact mediator factorial | **additive distributed pair** | active | `task14_mlp6_7_direction_cardinality_program_mlp15_vs_mlp17_mediation_v1_result.json` |
| `agreement_mlp6_7_program_mediator_gain.v1` | complete | row-held-out scalar composition | **cardinality extension inconclusive** | superseded by v2 | `task14_mlp6_7_direction_cardinality_program_loo_mediator_gain_v1_result.json` |
| `agreement_mlp6_7_program_mediator_gain.v2` | complete | row-held-out scalar composition | **six-scalar screen** | active | `task14_mlp6_7_direction_program_loo_mediator_gain_v2_result.json` |
| `agreement_mlp6_7_program_mediator_gain_transfer.v1` | complete | sealed fourth-corpus composition | **null** | active | `task14_direction_mediator_gain_fourth_corpus_causal_validation_v1_result.json` |
| `agreement_mlp6_7_program_mlp15_component.v2` | complete | preregistered component audit | **null** | active | `task14_direction_mlp15_gain_fourth_corpus_component_v2_result.json` |
| `agreement_mlp6_7_program_broad_collateral.v1` | invalid | four-behavior selectivity | **FP32 tripwire invalid** | superseded by v2 | `task14_mlp6_7_direction_cardinality_program_broad_collateral_v1_result.json` |
| `agreement_mlp6_7_program_broad_collateral.v2` | complete | four-behavior selectivity | **held narrowly** | active | `task14_mlp6_7_direction_cardinality_program_broad_collateral_v2_result.json` |
| `agreement_mlp6_7_program_literal_price.v1` | complete | storage/compute/dependency audit | **interface simple, not end-to-end** | active | `task14_direction_cardinality_program_literal_price_v1_result.json` |
| `task14_bracket_program_stress_composition.v1` | complete | two-sided four-corner composition | **inconclusive: reverse stress inert** | superseded in one direction by v2 | `task14_bracket_fixed_program_stress_composition_v1_result.json` |
| `task14_under_bracket_program_stress.v2` | complete | one-sided four-corner composition audit | **held** | active | `task14_under_bracket_program_stress_composition_v2_result.json` |
| `bracket_under_norm_matched_task14_stress.v3` | complete | outcome-free norm-matched reverse robustness | **held at artificial matched norm** | active | `bracket_under_norm_matched_task14_program_stress_v3_result.json` |
| `task14_bracket_compiled_dispatcher.v1` | abandoned | pre-execution specification audit | **contradictory self-pair rule** | superseded by v2 | none |
| `task14_bracket_compiled_dispatcher.v2` | complete | exact combined dispatch and dependency boundary | **screen** | active | `task14_bracket_compiled_dispatcher_v2_result.json` |
| `bracket_ordered_pair_suffix_free_scalar_feasibility.v1` | complete | bidirectional leave-family-out scalar transfer | **retrospective feasibility** | active diagnostic | `bracket_ordered_pair_suffix_free_scalar_feasibility_v1_result.json` |
| `bracket_suffix_free_scalar_fresh_corpus.v1` | complete | frozen six-scalar prediction on untouched third construction | **predictive screen** | active | `bracket_suffix_free_scalar_fresh_corpus_validation_v1_result.json` |
| `bracket_l13h8_closer_absolute_term_program.v1` | complete | three absolute terms on untouched fourth construction | **program screen** | active | `bracket_l13h8_closer_absolute_term_program_v1_result.json` |
| `bracket_absolute_term_program_ood_controls.v2` | complete | two OOD targets plus three no-edit control families | **program screen** | active | `bracket_absolute_term_program_ood_control_validation_v2_result.json` |
| `task14_bracket_compiled_predictive_dispatcher.v3` | complete | exact strongest-program packaging and price | **predictive/manipulable interface** | active | `task14_bracket_compiled_predictive_dispatcher_v3_result.json` |
| `task14_direction_cardinality_absolute_head_program.v1` | complete | ten absolute heads on 512 third-corpus cells | **null: native base context required** | active | `task14_direction_cardinality_absolute_head_program_v1_result.json` |
| `task14_absolute_head_reader_scalar_context.v2` | complete | one frozen-reader coordinate of native base context | **null: material but insufficient** | active | `task14_absolute_head_reader_scalar_context_v2_result.json` |
| `task14_absolute_head_two_reader_context.v3` | complete | final fixed two-reader span of native base context | **null: reader-span compression closed** | active | `task14_absolute_head_two_reader_context_v3_result.json` |
| `task14_bracket_counterfactual_margin_actuator.v4` | complete | common baseline-margin plus 16-effect lookup | **suffix-free margin actuator screen** | active | `task14_bracket_counterfactual_margin_actuator_v4_result.json` |
| `task14_bracket_margin_actuator_composition_contract.v5` | complete | typed immutable-baseline state algebra | **exact composition screen** | active | `task14_bracket_margin_actuator_composition_contract_v5_result.json` |
| `task14_bracket_native_baseline_semantic_linear_feasibility.v1` | complete | row-disjoint 6/5-coefficient baseline models | **retrospective feasibility screen** | active diagnostic | `task14_bracket_native_baseline_semantic_linear_feasibility_v1_result.json` |
| `task14_bracket_native_baseline_semantic_linear_prospective.v2` | complete | 27-scalar standalone program on two untouched corpora | **null: bracket near-cancellation exposes baseline error** | active | `task14_bracket_native_baseline_semantic_linear_prospective_v2_result.json` |
| `task14_bracket_counterfactual_error_budget_audit.v3` | complete | exact baseline/effect error decomposition | **bracket baseline is 95.1% of error norm** | active diagnostic | `task14_bracket_counterfactual_error_budget_audit_v3_result.json` |
| `task14_standalone_bracket_conditioned_hybrid.v6` | complete | 22-scalar strongest-boundary package | **predictive/composable/manipulable hybrid screen** | active | `task14_standalone_bracket_conditioned_hybrid_v6_result.json` |
| `task14_bracket_transparent_margin_program_release.v7` | complete | hash-bound importable/CLI 22-scalar program | **executable release** | active | `task14_bracket_transparent_margin_program_release_v7_result.json` |
| `bracket.pending_opener.native_baseline_l13h8_causal_ceiling_newest.v1` | complete | exact semantic-opener and whole-head zero removal on newest construction | **causal ceiling screen** | active | `bracket_native_baseline_l13h8_causal_ceiling_newest_v1_result.json` |
| `bracket.pending_opener.l13h8_direct_readout_baseline_program.v1` | complete | six-scalar causal decomposition on outcome-sealed archive construction | **null: cancellation precision not reached** | active | `bracket_l13h8_direct_readout_baseline_program_v1_result.json` |
| `task14_bracket_transparent_program_boundary_certificate.v8` | complete | hash-bound release/null boundary | **empirical minimality certificate** | active | `task14_bracket_transparent_program_boundary_certificate_v8_result.json` |
| `task14_bracket_text_selector_program_release.v9` | complete | exact controlled-text delimiter stack over 2,088 prompts | **selector-compiled executable screen** | active | `task14_bracket_text_selector_program_release_v9_result.json` |
| `bracket.pending_opener.circuit_source_selector_release.v10` | complete | exact pending state plus L13H8 source token over 2,088 prompts | **internal-selector compiler screen** | active | `bracket_circuit_source_selector_release_v10_result.json` |
| `subject_verb.number_agreement.text_direction_selector_program_release.v11` | complete | controlled-text subject-number read over 96 endpoints | **direction-compiled executable screen** | active | `task14_text_direction_selector_program_release_v11_result.json` |
| `cross_behavior.task14_bracket_selector_compiled_margin_program_release.v12` | complete | hash-bound unified raw-text selector and margin-program release | **selector-compiled controlled-domain release** | active | `task14_bracket_selector_compiled_margin_program_release_v12_result.json` |

**Important negative result:** the Program-A optimizer improved its objective by 0.025–0.047 across nine fits, below the registered
minimum improvement of 0.05. It therefore cannot answer whether a small causal subspace exists. Do not repeat the same optimizer or
reinterpret its target/control scores as a subspace null. The corrected receipt and bundle are internally hash-consistent; the first
publication pair is retained only under `artifact_invalid` filenames.

**Next:** keep both fixed internal programs and their bounded cross-program robustness, but reject fixed MLP15/17 gains and any
whole-model simplicity claim. Counterfactual role-prompt lookup is eliminated. One- and two-reader Task14 base repairs are null, so
reader-span compression is closed. At the answer-margin boundary, the v4 actuator now removes all stored intervention vectors and all
post-input model execution: one native unedited margin plus an edit specification selects one of sixteen frozen effects and performs one
addition. The typed v5 contract now covers sequential/no-edit operations exactly without new learned state: same-slot edits overwrite
from the immutable baseline and independent slots commute; donorward effects are never summed. The remaining frontier is therefore the
native baseline margin itself. Minimal six/five-coefficient baseline generators transfer prospectively, and the Task14 standalone
counterfactual passes, but the combined 27-scalar program is a null: bracket absolute-counterfactual cosine is 0.80754 and relative L2
is 1.02870. The bracket effect still transfers at cosine 0.99816/relative L2 0.06068; baseline error is amplified where native margin and
edit effect nearly cancel. Retain the baseline-conditioned v4 actuator and close semantic-linear standalone removal without adding
pair interactions, templates, lexical features, row identity, rank, or outcome-conditioned rescue.

The exact post-result budget preserves that null and sharpens the boundary: bracket baseline error is 95.1% of total error norm,
whereas intervention-effect error is 15.8%; median cancellation amplification is 5.80x and reaches 46.15x. Task14 cancellation is
negligible and its prospective standalone margins pass. The next package should therefore promote only the preregistered Task14
component, retain bracket baseline conditioning, and preserve the exact typed composition contract.

The v6 hybrid does exactly that in 22 scalars: Task14 is standalone on untouched data, bracket effect prediction transfers to the newest
construction, and all typed composition cases remain exact. Its sole model-valued runtime dependency is the native unedited bracket
margin. The next circuit-first frontier is a fixed readout of the already localized native bracket term/state, evaluated for the precision
needed under counterfactual cancellation; semantic feature expansion remains closed.

The v7 release makes this boundary executable rather than documentary. Its hash-bound CLI/import API passes all 64 Task14 and 45
bracket conformance cases, rejects malformed specifications, and imports no model/training/network dependency. The release manifest
binds the prospective evidence, exact composition counts, 22-scalar price, combined standalone null, and remaining bracket baseline
dependency. It is the current simplest honest transparent program, not a whole-model replacement.

The newest-construction causal ceiling now localizes a material part of that remaining dependency. Exact semantic-opener removal at
L13H8 damages the correct closer on all 72 endpoints, explains 44.83% of native-margin norm, aligns with native margin at cosine 0.84532,
and agrees with complete-head damage at cosine 0.97512. Every one of the six ordered delimiter pairs recurs. This licenses exactly one
fixed direct-readout compression evaluated at the counterfactual-cancellation precision boundary; failure closes local L13H8 baseline
compression without feature, site, rank, or reconstruction rescue.

That sole direct-readout test is now a valid null. The checkpoint-fixed scalar readout transfers the semantic-term contribution at
cosine 0.98622 and relative L2 0.16672, and the six-scalar causal decomposition improves native-baseline relative L2 from 0.28794 to
0.20019. It nevertheless misses the frozen baseline precision and three ordered-pair bars; native/edit cancellation amplifies the
remaining error to counterfactual cosine 0.77108, relative L2 0.75136, and sign agreement 0.73611. Local L13H8 bracket-baseline
compression is closed. The v7 native-margin dependency remains the simplest honest boundary.

The v8 certificate makes that boundary append-only and hash-bound: v7 remains minimal only within the tested semantic-linear and
single fixed L13H8-readout classes. It does not claim universal minimality, free-form support, or whole-model replacement.

The v9 executable removes a different dependency rather than retrying baseline compression. A transparent delimiter-stack reader
infers the pending closer directly from controlled-domain raw text with zero learned scalars. It matches all 2,088 labeled endpoints
across five frozen corpora and all 6,264 corresponding v7 equations exactly, while rejecting balanced, multiply-pending, mismatched,
and malformed inputs. The bracket API now retains only raw text, one native unedited donorward margin, and the desired closer edit.

The v10 circuit selector extends that read to internal execution: the same controlled text plus native token IDs identifies the exact
L13H8 semantic-opener token position as well as the recipient closer on all 2,088 endpoints. Thus neither circuit selector must be
supplied externally; internal causal execution still retains native prefix/base activation and suffix computation.

The unified v11 executable removes Task14 direction as an external input too. Controlled raw subject text yields the labeled number
and opposite-number direction on all 96 frozen endpoint texts, and the resulting API matches 3,072 v7 equations exactly across every
E/A/U/W subset and native/edit arm. E/A/U/W membership and edit/no-edit remain explicit intervention specifications, not inferred state.

The v12 release binds those selectors to the strongest honest executable boundary. Its 22 stored FP32 scalars (88 bytes) pass 9,336
frozen selector/equation checks: 96 Task14 endpoint texts, 3,072 Task14 equations, 2,088 bracket text/source endpoints, and 6,264 bracket
equations. Task14 is standalone in the controlled domain; bracket prediction remains conditioned on one native unedited donorward
margin. The release is therefore predictive, composable, and manipulable at the margin interface, but neither free-form nor a
whole-model replacement.

### `aspectual_anchor.has_vs_had` — module circuit localized

| candidate | status | test | verdict | result artifact |
|---|---|---|---|---|
| `aspectual_anchor.has_vs_had` | complete | 55-site A1/A2/P/C fast screen | **residual carrier from resid:10** | `aspectual_anchor_has_vs_had_v1_result.json` |
| `aspectual_anchor.has_vs_had.layer8_9_module_factorial_v1` | abandoned pre-run | timestamp integrity | **no model execution** | none |
| `aspectual_anchor.has_vs_had.layer8_9_module_factorial_v2` | complete | exact 16-arm L8/L9 module factorial plus nine L9 heads | **composable module circuit screen** | `aspectual_anchor_layer8_9_module_factorial_v2_result.json` |
| `aspectual_anchor.has_vs_had.fresh_construction_transfer_v1` | invalid | frozen two-construction module/head transfer | **native capability gate failed** | `aspectual_anchor_fresh_construction_transfer_v1_result.json` |
| `aspectual_anchor.has_vs_had.l9h1_h4_source_term_factorial_v1` | complete | exact L9H1/H4 source-term interchange | **null: heads do not read raw cue directly** | `aspectual_anchor_l9h1_h4_source_term_factorial_v1_result.json` |
| `aspectual_anchor.has_vs_had.l9h1_h4_downstream_source_bank_v1` | invalid | exact multi-source H1/H4 compression | **BF16 closure tripwire** | `aspectual_anchor_l9h1_h4_downstream_source_bank_v1_result.json` |
| `aspectual_anchor.has_vs_had.l9h1_h4_downstream_source_bank_v2` | complete | sole BF16-corrected, science-identical source bank | **contextual source-bank screen** | `aspectual_anchor_l9h1_h4_downstream_source_bank_v2_result.json` |
| `aspectual_anchor.has_vs_had.contextual_source_state_onset_v1` | complete | exact source-bank residual-depth sweep | **source carrier first sufficient at resid:5** | `aspectual_anchor_contextual_source_state_onset_v1_result.json` |
| `aspectual_anchor.has_vs_had.block4_contextual_source_writer_factorial_v1` | complete | exact carried/attention4/MLP4 Boolean factorization | **MLP4 contextual writer screen** | `aspectual_anchor_block4_contextual_source_writer_factorial_v1_result.json` |
| `aspectual_anchor.has_vs_had.mlp4_bilinear_response_factorial_v1` | invalid | exact left/right/interaction response factorial | **BF16 intermediate closure tripwire** | `aspectual_anchor_mlp4_bilinear_response_factorial_v1_result.json` |
| `aspectual_anchor.has_vs_had.mlp4_bilinear_response_factorial_v2` | complete | tolerance-only corrected response factorial | **two-term MLP4 bilinear subprogram** | `aspectual_anchor_mlp4_bilinear_response_factorial_v2_result.json` |
| `aspectual_anchor.has_vs_had.mlp4_to_l9h1_h4_path_mediation_v1` | null | exact two-term-writer to source-bank/head mediation | **specific partial path; misses frozen sufficiency by 0.24 points** | `aspectual_anchor_mlp4_to_l9h1_h4_path_mediation_v1_result.json` |
| `aspectual_anchor.has_vs_had.mlp4_induced_l9_head_sweep_v1` | null | fixed all-head singleton plus leave-one-out mediation | **H1/H4 dominate; no missing direct L9 head** | `aspectual_anchor_mlp4_induced_l9_head_sweep_v1_result.json` |
| `aspectual_anchor.has_vs_had.mlp4_induced_block9_crossing_factorial_v1` | abandoned pre-run | BF16 instrument audit | **no model execution** | none |
| `aspectual_anchor.has_vs_had.mlp4_induced_block9_crossing_factorial_v2` | complete | exact carried9/attention9/MLP9 factorial | **attention9-dominant resid10 crossing** | `aspectual_anchor_mlp4_induced_block9_crossing_factorial_v2_result.json` |
| `aspectual_anchor.has_vs_had.mlp4_induced_final_query_onset_v1` | abandoned pre-run | exact-budget audit | **no model execution** | none |
| `aspectual_anchor.has_vs_had.mlp4_induced_final_query_onset_v2` | null | fixed resid5–9 final-query depth sweep | **observed onset resid6, not predicted resid9** | `aspectual_anchor_mlp4_induced_final_query_onset_v2_result.json` |
| `aspectual_anchor.has_vs_had.mlp4_induced_block5_crossing_factorial_v1` | complete | exact carried5/attention5/MLP5 factorial | **attention5 initiates final-query transport** | `aspectual_anchor_mlp4_induced_block5_crossing_factorial_v1_result.json` |
| `aspectual_anchor.has_vs_had.mlp4_induced_attention5_head_sweep_v1` | null | fixed singleton plus leave-one-out attention5 sweep | **transport is distributed across heads** | `aspectual_anchor_mlp4_induced_attention5_head_sweep_v1_result.json` |
| `aspectual_anchor.has_vs_had.mlp4_induced_attention5_four_head_factorial_v1` | complete | exact 16-arm H7/H1/H6/H8 factorial | **four-head attention5 transporter** | `aspectual_anchor_mlp4_induced_attention5_four_head_factorial_v1_result.json` |
| `aspectual_anchor.has_vs_had.mlp4_to_attention5_four_head_source_identity_v1` | complete | exact attention5 source-term identity | **MLP4 bank-to-four-head path closed** | `aspectual_anchor_mlp4_to_attention5_four_head_source_identity_v1_result.json` |
| `aspectual_anchor.has_vs_had.explicit_path_lexical_holdout_v1` | invalid pre-outcome | prospective lexical/recombination path transfer | **token alignment failed before model forward** | run log |
| `aspectual_anchor.has_vs_had.explicit_path_lexical_holdout_v2` | complete | token-audited prospective path transfer | **explicit path transfers prospectively** | `aspectual_anchor_explicit_path_lexical_holdout_v2_result.json` |
| `aspectual_anchor.has_vs_had.transparent_path_program_release_v1` | released | zero-forward hash/graph/metric/scope audit | **typed paired-causal tensor program** | `aspectual_anchor_transparent_path_program_release_v1_result.json` |
| `aspectual_anchor.has_vs_had.attention9_h1h4_lexical_holdout_v1` | complete | sealed prospective downstream-bank transfer | **H1/H4 bank transfers prospectively** | `aspectual_anchor_attention9_h1h4_lexical_holdout_v1_result.json` |
| `aspectual_anchor.has_vs_had.blocks6_8_crossing_factorials_lexical_holdout_v1` | complete | shared-capture exact three-boundary factorials | **prospective intermediate route closed** | `aspectual_anchor_blocks6_8_crossing_factorials_lexical_holdout_v1_result.json` |
| `aspectual_anchor.has_vs_had.block9_crossing_factorial_lexical_holdout_v1` | complete | sealed prospective block9 factorial | **attention9-dominant resid10 crossing** | `aspectual_anchor_block9_crossing_factorial_lexical_holdout_v1_result.json` |
| `aspectual_anchor.has_vs_had.transparent_path_program_release_v2` | released | executable API/equation/evidence audit | **zero-fit tensor-equation program to resid10** | `aspectual_anchor_transparent_path_program_release_v2_result.json` |
| `aspectual_anchor.has_vs_had.suffix_depth_adaptive_factorial_lexical_holdout_v1` | complete | frozen 16/16 suffix selection/confirmation | **block11 crossing confirmed without leakage** | `aspectual_anchor_suffix_depth_adaptive_factorial_lexical_holdout_v1_result.json` |
| `aspectual_anchor.has_vs_had.block15_crossing_confirmation_v1` | complete | disjoint secondary suffix factorial | **block15 crossing confirmed** | `aspectual_anchor_block15_crossing_confirmation_v1_result.json` |
| `aspectual_anchor.has_vs_had.attention11_15_head_compression_split_v1` | complete | split singleton/necessity plus compact-set test | **four-head suffix sets transfer disjointly** | `aspectual_anchor_attention11_15_head_compression_split_v1_result.json` |
| `aspectual_anchor.has_vs_had.attention11_15_single_head_confirmation_v1` | complete | unopened disjoint singleton compression | **block11 H3 and block15 H5 suffice compactly** | `aspectual_anchor_attention11_15_single_head_confirmation_v1_result.json` |
| `aspectual_anchor.has_vs_had.attention11h3_15h5_source_compression_split_v1` | invalid | split six-role source compression | **audit compared query-only deltas over the full sequence; rows preserved but unreleased** | `aspectual_anchor_attention11h3_15h5_source_compression_split_v1_result.json` |
| `aspectual_anchor.has_vs_had.attention11h3_15h5_source_projection_query_diagnostic_v1` | complete diagnostic | frozen confirmation query-index audit | **v1 failure localized exactly to off-query indices** | `aspectual_anchor_attention11h3_15h5_source_projection_query_diagnostic_v1_result.json` |
| `aspectual_anchor.has_vs_had.attention11h3_15h5_source_compression_release_v1` | released | zero-forward immutable evidence-class audit | **three-role 11H3/15H5 banks released with post-outcome-repair label** | `aspectual_anchor_attention11h3_15h5_source_compression_release_v1_result.json` |
| `aspectual_anchor.has_vs_had.transparent_path_program_release_v3` | released | executable API/equation/evidence audit | **source-resolved paired-causal program through block15** | `aspectual_anchor_transparent_path_program_release_v3_result.json` |
| `aspectual_anchor.has_vs_had.mlp11_15_bilinear_compression_split_v1` | invalid, superseded | split exact bilinear-response factorials | **used all six source roles instead of frozen three-role banks** | `aspectual_anchor_mlp11_15_bilinear_compression_split_v1_result.json` |
| `aspectual_anchor.has_vs_had.transparent_path_program_release_v4` | invalid, superseded | executable API/equation/evidence audit | **compiled invalid v1 MLP authority; v3 remains valid release** | `aspectual_anchor_transparent_path_program_release_v4_result.json` |
| `aspectual_anchor.has_vs_had.mlp11_15_bilinear_compression_v1_design_audit` | invalid | zero-forward design audit | **A/B diagnose mismatch; C exact-string assertion was malformed** | `aspectual_anchor_mlp11_15_bilinear_compression_v1_design_audit_result.json` |
| `aspectual_anchor.has_vs_had.mlp11_15_bilinear_compression_v1_design_audit_v2` | complete diagnostic | exact-string-only audit correction | **v1 and program v4 superseded invalid** | `aspectual_anchor_mlp11_15_bilinear_compression_v1_design_audit_v2_result.json` |
| `aspectual_anchor.has_vs_had.mlp11_15_bilinear_compression_split_v2` | complete | sole-correction split bilinear factorials | **two-term MLP11/15 responses transfer in exact released source contexts** | `aspectual_anchor_mlp11_15_bilinear_compression_split_v2_result.json` |
| `aspectual_anchor.has_vs_had.transparent_path_program_release_v5` | released | valid-lineage executable audit | **corrected source- and MLP-resolved paired-causal program through block15** | `aspectual_anchor_transparent_path_program_release_v5_result.json` |
| `aspectual_anchor.has_vs_had.sparse_suffix_recurrence_confirmation_v1` | null | confirmation-only sparse resid10-to-resid18 recurrence | **blocks11+15 necessary but retain 78.99%, below 85% sufficiency** | `aspectual_anchor_sparse_suffix_recurrence_confirmation_v1_result.json` |
| `aspectual_anchor.has_vs_had.sparse_suffix_missing_block_compression_split_v1` | invalid | split singleton/necessity screen over omitted suffix writes | **dense control compared compressed block11/15 recurrence against full block11/15 writer; block12+14 signal remains unreleased** | `aspectual_anchor_sparse_suffix_missing_block_compression_split_v1_result.json` |
| `aspectual_anchor.has_vs_had.sparse_suffix_missing_block_v1_design_audit` | complete diagnostic | zero-forward semantic control audit | **v1 failure is localized to an incommensurate compressed-vs-full program comparison** | `aspectual_anchor_sparse_suffix_missing_block_v1_design_audit_result.json` |
| `aspectual_anchor.has_vs_had.sparse_suffix_missing_block_compression_split_v2` | complete, post-outcome repair | sole-control-repair replication | **blocks12+14 recover 77.97% of missing increment and 96.43% of all-omitted total; not prospective** | `aspectual_anchor_sparse_suffix_missing_block_compression_split_v2_result.json` |
| `aspectual_anchor.has_vs_had.suffix_block12_14_component_factorial_split_v1` | complete, conditional | exact four-component Shapley lattice plus split confirmation | **MLP12+MLP14 retain 90.70% of selected-block increment and 98.78% of full total** | `aspectual_anchor_suffix_block12_14_component_factorial_split_v1_result.json` |

The fixed L8/L9 module bank recovers 68.33% of the native donor effect: A1 65.81%, A2 70.85%, with perfect directional recurrence.
Its P and canonical-C effects are 11.69% and 7.77%. Removing attention layer 9 loses 38.72 recovery points, and exact factorial Shapley
attribution assigns it the largest contribution (40.27 points). The preregistered non-adaptive head sweep localizes L9H1 and L9H4 as
individually sufficient partial carriers. This is a screen on the discovery corpus; the frozen bank and heads require a sealed fresh
construction before promotion to a predictive circuit.

The first frozen construction transfer cannot provide that promotion. Although the four-module bank moves the target by 74.20%, the
H1/H4-reduced bank retains 94.34% of that movement, and all target directions recur, A1, A2, and one P native capability cell fail; A2
falls to 0.625/0.5625. Those intervention numbers are therefore disclosed only as invalid diagnostics. The prompts are closed without
post-outcome retuning. The next circuit-first question is what token/source relation L9H1 and L9H4 read on the valid discovery corpus.

That exact source-term test rejects the simplest direct-cue story. The manual attention path matches native scored logits exactly and
reconstructs native head vectors to 3.8e-6, while the complete H1/H4 pair recovers 38.39%. Replacing only the `since/by` term recovers
2.42% (6.30% of the pair); the unchanged downstream `last` and period positions carry 8.96% and 15.19%. Thus the heads read a contextual
trace propagated into later source-token states, not the raw temporal preposition. The frozen next question is whether a small
downstream source bank (period plus determiner) composes most of the pair effect.

It does. The first run was invalid only because repeated BF16 additions missed an unrealistic 1e-4 scored-logit closure tolerance by
0.02095; no causal threshold or arm was interpreted. The sole corrected run uses a fixed 0.125 BF16-scale closure quantum with identical
science. Period+determiner retains 72.00% of complete H1/H4 recovery, and adding `last` retains 96.03%, with perfect direction in both
constructions. `cue+self` retains just 3.44%. On this valid discovery population, the read is therefore a three-source contextual bank
over `last`, the period token, and `the`; it is not a direct edge from the raw `since/by` token.

The contextual carrier is now depth-localized. Exact replacement of the `last`+period+`the` residual states is inert at embeddings,
rises from 31.20% recovery at resid:3 to 43.15% at resid:4, and first passes at resid:5 (A1 52.87%, A2 58.56%, both direction
1.0). The raw cue state instead falls from 35.12% at embeddings to 3.11% at resid:5. This places the sufficient contextual write five
boundaries before the final-subject onset at resid:10 and licenses an exact carried/attention/MLP factorization of block 4.

That factorization closes to direct resid:5 replacement within 7.6e-6 scored logit. MLP4 is the decisive new writer: its exact
three-factor Shapley contribution is 33.58 recovery points, versus 6.03 for attention4 and 16.11 for carried resid:4 state. Removing
MLP4 from the full arm loses 33.56 points in A1 and 35.89 in A2. MLP4 alone recovers 31.89%; carried+MLP4 reaches 49.19%. The next
factorization target is therefore the bilinear MLP4 response at the three contextual source positions.

The response factorization resolves that target. V1 was invalid solely because its FP32 reconstruction differed from the native
BF16 intermediate by 0.001160, narrowly outside a 0.001 tolerance, even though the independently scored full-factor/direct-MLP4
closure was 7.6e-6. A preregistered tolerance-only v2 (0.002; all rows, arms, factors, behavioral gates, and computations unchanged)
passes every gate. The exact right-change and left-change terms have Shapley contributions of 17.17 and 14.19 recovery points;
their fixed pair recovers 33.38%, or 104.67% of the complete three-term MLP4 response, with direction fraction 1.0 in A1 and A2.
The mixed bilinear interaction contributes only 0.52 points and is dispensable on this population. The transparent writer is thus
`Down((Left_d-Left_b)*Right_b + Left_b*(Right_d-Right_b))` at `last`+period+`the`. The next frozen test asks whether this two-term
write is specifically mediated through those positions' L9H1/H4 attention-source terms, closing the writer-to-reader path.

That path is real and sharply specific, but the preregistered whole-path claim is null. The two-term writer recurs at 33.38%, while
transplanting its induced `last`+period+`the` source terms through L9H1/H4 recovers 13.27%: 39.76% of the writer, just below the frozen
40% sufficiency bar. No threshold was rounded or retuned. The bank nevertheless captures 102.02% of the complete all-source H1/H4
effect, both families have direction fraction 1.0, and cue+self carries only 2.05% in absolute terms. Thus this is a licensed partial
edge, not a licensed complete route. A fixed all-nine-head L9 sweep is required to locate the remainder of the MLP4-induced signal.

The non-adaptive all-head sweep rules out that tempting rescue. All nine L9 heads at the final-subject query together mediate 14.87%
recovery, only 44.54% of the MLP4 writer and below the frozen 50% bar. H4 and H1 dominate the endpoint attribution at 7.42% and
5.56%; the largest remaining heads, H8 and H7, contribute only 0.96% and 0.91%, and no additional head passes the preregistered joint
sufficiency/necessity criterion. The unexplained writer effect therefore does not sit in another direct L9 final-query head. The next
circuit boundary is resid:10 itself: factor the writer-induced final-query crossing into carried resid:9, attention9, and MLP9 terms.

The corrected block9 crossing (v1 was abandoned before execution after a BF16-tolerance audit) closes to direct resid:10 replacement
within 5.7e-6 scored logit and screens. The complete final-query crossing retains 76.53% of the two-term MLP4 writer: A1 23.61%, A2
27.48%, both direction 1.0. Attention9 is dominant with exact factorial Shapley 13.32 points; carried resid:9 contributes 9.38 and
MLP9 2.84. Removing attention9 loses 11.60 points in A1 and 15.10 in A2. Notably, attention9's attribution agrees with the independently
measured H1/H4 contextual-bank mediation (13.27%), tying that explicit reader edge to the dominant crossing term. The remaining
secondary branch is already present in carried resid:9, so the next exact depth sweep traces when the writer reaches the final query
through blocks5-8 before factoring the onset block.

The frozen depth prediction is null but the curve is more informative than expected. Resid5 final-query replacement is exactly inert,
confirming that MLP4 changed only the three source positions. The writer signal first passes immediately at resid6, after block5
(5.53% recovery; A1 direction 0.9375, A2 1.0), rather than at predicted resid9. It then accumulates monotonically: resid7 7.29%,
resid8 9.57%, resid9 10.92%. The carried branch is therefore initiated by block5 and strengthened through blocks6-8; the next exact
factorization targets block5's carried/attention/MLP crossing at the final query.

That onset factorization closes within 6.7e-6 scored logit and screens. It exactly reproduces the resid6 recovery (5.5313%). Attention5
is the first transport operation, with Shapley 4.8411 points; MLP5 adds 0.6902 and carried resid5 is exactly zero, as required by the
source-only MLP4 intervention. Removing attention5 loses 4.94 points in A1 and 4.81 in A2. The explicit circuit now branches from the
two-term MLP4 write into an immediate attention5 source-to-final-query edge, followed by accumulation through blocks6-8, alongside the
separately closed attention9 H1/H4 contextual-bank edge. The next frozen decomposition resolves which attention5 heads carry the onset.

No single attention5 head owns that onset. The all-head intervention exactly recurs at 5.5313%, but the preregistered 3-point
single-head localization gate is null. The stable sufficiency/necessity endpoint ranking is H7 1.339%, H1 1.270%, H6 1.077%, H8
0.839%, and H0 0.630%; remaining heads are negligible. The top four sum to roughly 82% of the all-head effect but this arithmetic is
not yet a causal compression claim. A frozen 16-arm H7/H1/H6/H8 factorial is the next efficient test.

The exact factorial licenses that compression. H7/H1/H6/H8 recover 4.5575%, or 82.40% of all-head attention5 transport, with A1/A2
direction fractions 0.9375/1.0. All four exact Shapley values are positive (H7 1.338%, H1 1.267%, H6 1.078%, H8 0.875%); H7 is
largest and its removal lowers both families. Because the upstream MLP4 intervention changes only `last`+period+`the` at resid5,
attention5's first read now admits a stringent source-term identity test: those three changed terms should reconstruct the licensed
four-head output difference without invoking any other source.

It does, to numerical identity. The `last`+period+`the` source-term arm retains 99.9997% of the complete H7/H1/H6/H8 effect and differs
by at most 6.2e-6 scored logit; the all-source arm is identical to the same precision. Cue and final-query self terms are exactly zero.
Individually, period carries 2.119%, the determiner 1.424%, and `last` 0.954% recovery. This licenses the explicit discovery-corpus path
`Down((Left_d-Left_b)*Right_b + Left_b*(Right_d-Right_b)) @ MLP4(last,period,the)` -> the same three source terms ->
`attention5 heads {7,1,6,8}` -> final query. The route is manipulable and composable, while downstream blocks6-9 still supply measured
amplification/parallel transport rather than a complete standalone predictor.

The explicit path now has prospective support. V1 failed closed before any model forward because four holdout agents tokenized into two
tokens; an append-only v2 replaced only those agents using model-free tokenizer metadata and preserved every scientific arm and bar.
On the 64-row sealed authority, every A1/A2/P capability cell is perfect and both canonical controls pass. The two-term writer transfers
at 28.36% mean recovery (A1 26.88%, A2 29.83%); H7/H1/H6/H8 transport 3.966% with perfect direction, retain 92.74% of all attention5
heads, and are reproduced 99.9999% by the fixed `last`+period+`the` source bank. This promotes lexical/recombination stability within
the two validated syntactic constructions. It does not reopen the failed fresh-construction claim or imply free-form transfer.

The evidence is now compiled rather than left as a narrative chain. The v1 typed artifact binds the exact two-term MLP4 equation,
three source/write positions, attention5 heads H7/H1/H6/H8, carried resid5–9 curve, and discovery-only attention9 H1/H4 observation
to seven immutable hashes. Its zero-forward release audit passes all authority, graph, metric, price, dependency, and negative-scope
checks. The price is zero fitted scalars and vectors, but execution still requires checkpoint weights plus paired base/donor states.
Accordingly this is a predictive prospective circuit-effect program within the two validated constructions—not a native-margin,
full-logit, free-form, new-construction, or whole-model program. The null-terminal onset and H1/H4 receipts remain null in the release.

The previously discovery-only attention9 branch now has its own unopened-outcome prospective test on that same immutable authority.
H1/H4 recover 10.761% of the native donor effect, or 37.95% of the prospectively recurrent two-term writer, with A1/A2 direction
fractions 0.9375/1.0. `last`+period+`the` recover 11.038% and 102.57% of the all-source H1/H4 effect; cue+self account for only
2.45% in absolute proportion. All five frozen predictions pass in 16 forwards and 256 example evaluations with no fit. This promotes
the H1/H4 bank branch to lexical/recombination stability in the two known-capable syntaxes, while leaving new-construction and
standalone prediction boundaries unchanged.

One shared-capture prospective run now closes the intermediate final-query route without three redundant model loads. Starting from
the open all-head attention5 value of 4.277%, the exact full crossings rise monotonically to 5.749% at resid7, 7.342% at resid8,
and 8.762% at resid9. Carried state is dominant at every boundary (Shapley 3.616%, 5.474%, 6.478%), while attention6/7/8 adds
1.074%, 1.332%, and 1.763% and MLP6/7/8 adds 1.058%, 0.537%, and 0.522%. Removing each attention term damages both A1 and A2;
all residual tensor errors are below 0.0004 and all scored-logit closures below 6.7e-6. The run passes all frozen predictions in
62 forwards and 992 evaluations, prospectively turning the earlier depth curve into a component-resolved compositional route.

The final measured crossing into resid10 also transfers prospectively. Its full carried9+attention9+MLP9 arm recovers 21.410% of the
native donor effect and retains 75.50% of the two-term writer, with perfect direction in both families. Exact Shapley attribution again
makes attention9 dominant at 11.205%, versus carried9 8.594% and MLP9 1.612%; removing attention9 damages A1 by 9.081 points and A2
by 13.530. The full factorial closes to direct resid10 replacement within 4.8e-6 scored logit. Thus the entire MLP4 source-bank route
through attention5, blocks6–8, and the block9 crossing is now prospectively component-resolved within the two capable constructions.

The v2 release makes that route executable as framework-agnostic tensor equations rather than only a JSON graph. Its API implements
the two-term MLP4 hidden response and Down projection, per-head source-term deltas for the frozen attention5 and attention9 banks,
arbitrary carried/attention/MLP crossing subsets, and functional query writes. A zero-forward audit passes 22 deterministic synthetic
equation and rejection cases and binds every prospective value above to immutable evidence. The module has no model-loading import and
stores no fitted scalar or vector. It remains explicitly paired-causal: checkpoint weights, paired captures, and the native blocks10–17
suffix are dependencies, so this release is not presented as standalone native-margin, full-logit, free-form, or whole-model prediction.

A leakage-controlled 16/16 split now opens the native suffix. On selection rows, direct-query recovery rises from 20.663% at resid10
to 27.218% at resid18, which reproduces the writer exactly. The frozen rule selects block11's 3.454-point jump over the secondary
block15 jump of 2.125 points. On disjoint confirmation rows, block11's carried/attention/MLP Shapley values are 18.918%, 4.487%, and
3.444%; removing attention or MLP damages both A1 and A2, and the full arm (26.850%) exceeds carried alone (18.858%). Residual closure
is 2.9e-6 and all five preregistered checks pass. Block11 is therefore a prospectively confirmed suffix amplifier; block15 remains a
selection-only secondary candidate until its still-unopened confirmation factorial is run.

That block15 confirmation now screens on the untouched half. Its exact carried/attention/MLP Shapley values are 26.030%, 2.502%,
and 0.654%; the full crossing reaches 29.186% versus 26.084% for carried alone. Removing attention damages A1/A2 by 2.624/2.330
points and removing MLP damages them by 0.462/0.773, so both new components recur across families. Closure is 2.9e-6. The suffix's
two material selection-half gains—block11 and block15—are therefore both component-resolved on disjoint confirmation rows.

Their attention terms also admit compact head sets. Selection ranks block11 H3/H7/H2/H6 and block15 H5/H1/H4/H6; on disjoint
confirmation rows those four-head sets retain 96.23% and 101.57% of the respective all-head-minus-no-head increments, with positive
increments in A1 and A2. Projection error is below 0.00035 and crossing tensor error below 0.00391. The selection rankings are highly
concentrated: block11 H3 accounts for a 3.388-point endpoint score versus 0.117 for H7, while block15 H5 accounts for 1.964 points
versus 0.380 for H1. Because confirmation outcomes for the singleton sets remain unopened, a frozen H3/H5 one-head compression test
is licensed before source-term decomposition.

That final compression also screens. On the same disjoint confirmation half but previously unopened singleton outcomes, block11 H3
retains 81.44% of the validated four-head attention increment (3.535% versus 4.341%), and block15 H5 retains 83.22% (2.094% versus
2.516%). Both heads have positive no-head-relative increments in A1 and A2; all control values recur exactly. The material suffix
attention circuit is therefore two heads total—L11H3 and L15H5—before source-term identity is tested.

The first source-term compression artifact remains formally invalid and is not silently relabeled. Its six-role split nevertheless
selected determiner+period+self for L11H3 and period+determiner+self for L15H5; those banks retained 90.34% and 87.03% on disjoint
confirmation rows, and the all-source logits matched the frozen singleton-head controls within 3.8e-6. The failed audit statistic
compared a delta constructed only at the semantic query against dominant-head deltas at every sequence position. A separately frozen
post-outcome diagnostic confirms exact pattern-times-value reconstruction, query-index projection errors of 7.6e-6/4.8e-6, and that
the large 26.40/25.71 maxima occur entirely off query. This diagnoses the implementation fault but does not itself promote the source
banks; an immutable release audit must still distinguish preserved prospective evidence from the post-outcome correction.

That release audit passes without rerunning or rewriting either artifact. It verifies all 576 unique finite scientific records, the
original frozen selection rule and price, all disjoint compression gates, and the independent query-index diagnosis. The released
banks are `L11H3 <- {determiner, period, self}` at 90.34% retention and `L15H5 <- {period, determiner, self}` at 87.03%; their
confirmation increments are positive in both A1 and A2. The evidence class is explicitly prospective scientific arms with a
post-outcome instrument-audit repair. This licenses compilation into the executable paired-causal program, but still requires full
carried and MLP boundary deltas plus the checkpoint/native suffix and does not widen the construction or standalone scope.

Executable v3 compiles that release into typed array equations. In addition to the complete v2 MLP4-to-resid10 API, it validates
six-role causal-prefix partitions, computes the frozen L11H3/L15H5 source sums, scatters selected head vectors into checkpoint
projection space, and composes source-resolved suffix crossings. All five zero-forward release audits and 33 deterministic equation
and rejection cases pass. The program stores no fitted scalar or vector. It remains a paired-causal program: full carried and MLP
deltas at blocks11/15 and native blocks10,12-14,16-17/readout are dependencies, not silently modeled transparent components.

The remaining full MLP11/15 terms now compress as exact bilinear responses. Selection chooses Left-change+Right-change at MLP11 and
Left-change+bilinear-interaction at MLP15. On disjoint confirmation rows those pairs retain 101.15% and 110.07% of the respective
all-three-minus-empty MLP increments, with positive increments in A1 and A2. Exact three-term response reconstruction errors are
0.00115/0.00299, and the all-three scored outputs match the released source-resolved crossings within 4.8e-6. All five frozen gates
pass in 60 forwards and 480 evaluations. The next compilation can therefore replace both native suffix MLP deltas with these typed
two-term equations; the intervening native blocks and carried-state recurrence remain explicit dependencies.

Executable v4 performs that compilation. It inherits the audited v3 interface, adds arbitrary exact bilinear hidden-response subsets,
freezes the selected factor pair by suffix boundary, projects each pair through checkpoint `Down.weight`, and composes it with carried
state and source-resolved attention. All five zero-forward release audits and 25 deterministic equation/rejection cases pass. The
program is now source- and MLP-resolved at both material suffix amplifiers, with zero fitted state. It still depends on checkpoint
weights, paired states, native intervening blocks, and the final readout, so the standalone/free-form/whole-model exclusions remain.

Postexecution review found that the MLP-v1 runner supplied all six source roles where its prior fixed the released three-role bank.
The v1 result and the program-v4 release that compiled it are therefore append-only superseded as invalid; executable v3 remains the
latest valid program. The first design audit itself failed only an erroneous exact-string assertion after correctly diagnosing the
mismatch; a frozen v2 audit repaired that assertion and confirmed the invalid disposition. No v1 threshold or scientific number was
rescued. A sole-correction MLP v2 then ran the previously unopened exact compositions. It selects the same pairs—Left+Right at MLP11
and Left+interaction at MLP15—and on disjoint rows retains 101.16% and 110.05% of the all-three increments, with positive A1/A2
effects and every gate passing. Those corrected results, not v1, license the next executable compilation.

Executable v5 is that clean compilation. It builds directly on valid source-resolved v3 and corrected MLP v2, explicitly excludes
the invalid MLP-v1/program-v4 lineage, and implements the same typed bilinear subset, Down projection, and composed suffix-crossing
interfaces against the correct three-role context. All five release audits and 25 deterministic cases pass. V5 is now the latest
valid executable program; its remaining boundary is the native computation in intervening blocks and the final checkpoint readout.

The first attempt to remove those intervening computations is a valid null. Exact dense delta recurrence through blocks10-17 closes,
but lambda carry plus only the corrected block11 and block15 writes reaches 23.30% recovery versus dense 29.49%, retaining 78.99%
against the frozen 85% bar. Removing block11 damages A1/A2 by 6.69/6.38 points and removing block15 by 2.53/3.00, so both selected
writes are necessary. The six omitted blocks jointly remain material; the efficient next test is a split singleton/leave-one-out
screen over those six new-write boundaries followed by one compact-set confirmation, not six independent factorials.

### `subroutine.induction.equality_score` — site_live

**Read:** whether the current token matches a token at each earlier position. **Operation:** construct an attention score pattern over matching earlier positions independently of the copied payload. **Write:** a score pattern that can be combined with L8H4's value/output payload and read by MLP9. **Endpoint:** signed causal recovery of L8H4's copy-related effect and donor-answer logit movement.

| family | role | status |
|---|---|---|
| `cross_head_score_swap` | interchange | validated |
| `text_match_pattern_edit_payload_fixed` | interchange | proposed |
| `matched_natural_whole_state_swap` | interchange | proposed |
| `payload_swap_match_preserved` | invariance | proposed |
| `match_break_answer_fixed` | necessity | frozen |

**Append-only evidence ledger:**
| event | stage | test | verdict | lifecycle | result artifact |
|---|---|---|---|---|---|
| `equality_score_swap.r459.natural.v1` | complete | composition | **held** | active | `r459_result` |
| `equality_score_swap.r460.code.v1` | complete | ood | **failed** | active | `r460_result` |
| `equality_downstream_gate.r462.null.v1` | complete | composition | **null** | active | `r462_result` |
| `equality_source_correction.r464.v1` | complete | composition | **held** | active | `r464_result` |
| `equality_action_quotient.r498.null.v1` | complete | cross_family_transfer | **null** | active | `r498_result` |
| `equality_mlp9_reader.r500.v1` | complete | composition | **held** | active | `r500_result` |
| `equality_factor_branch_sharing.r531.null.v1` | complete | null_control | **null** | active | `r531_result` |
| `terminal_copy_four_head_removal.collateral_failure.v1` | complete | removal | **failed** | active | `terminal_copy_negative_receipt` |

**Frozen artifacts:** 21. Paths and SHA-256 hashes are in the canonical JSON record.

**Next:** materialize the text-edit and matched-natural answer-changing families plus the payload-preserving invariance family; then measure complete-state query/key/MLP7 ceilings with identical patch semantics before fitting a shared subspace

### `task.bracket.pending_opener` — specified

**Read:** opener, closer, ordering, and recency evidence in the preceding context. **Operation:** maintain which opener type remains pending after completed earlier spans. **Write:** signed evidence for the matching closer token. **Endpoint:** symmetric donor-closer versus base-closer final-logit margin.

| family | role | status |
|---|---|---|
| `direct_three_value_type_substitution` | interchange | frozen |
| `completed_then_reopened_three_value_order` | interchange | frozen |
| `pending_type_preserved_surface_rewrite` | invariance | frozen |
| `pending_type_preserved_distance_extension` | invariance | frozen |
| `pending_type_preserved_nonopener_punctuation` | invariance | frozen |

**Append-only evidence ledger:**
| event | stage | test | verdict | lifecycle | result artifact |
|---|---|---|---|---|---|
| `pending_opener_rank4_das.legacy.v1` | complete | das_interchange | **held** | active | `das_result` |
| `pending_opener_capability.r537.preregistered.v1` | preregistered | capability | **inconclusive** | superseded by `pending_opener_capability.r537.complete.v1` | `—` |
| `pending_opener_common_site_ceiling.r537.preregistered.v1` | preregistered | full_swap_ceiling | **inconclusive** | superseded by `pending_opener_common_site_ceiling.r538.invalid_unverified_checkpoint.v1` | `—` |
| `pending_opener_capability.r537.complete.v1` | complete | capability | **held** | active | `r537_capability_result` |
| `pending_opener_common_site_ceiling.r538.invalid_unverified_checkpoint.v1` | invalid | full_swap_ceiling | **invalid** | superseded by `pending_opener_common_site_ceiling.r538.complete.v2` | `r538_site_invalid_unverified_checkpoint_result` |
| `pending_opener_common_site_ceiling.r538.complete.v2` | complete | full_swap_ceiling | **held** | active | `r538_site_result_v2` |
| `pending_opener_control_ceilings.r539.preregistered.v1` | preregistered | null_control | **inconclusive** | superseded by `pending_opener_control_ceilings.r539.complete.v1` | `—` |
| `pending_opener_control_ceilings.r539.complete.v1` | complete | null_control | **held** | active | `r539_control_result` |
| `pending_opener_cross_family_das.r540.preregistered.v1` | preregistered | cross_family_transfer | **inconclusive** | superseded by `pending_opener_cross_family_das.r540.complete.v1` | `—` |
| `pending_opener_cross_family_das.r540.complete.v1` | complete | cross_family_transfer | **null** | active | `r540_das_result` |
| `pending_opener_split_integrity.r542.invalid_statistical_unit.v1` | complete | seed_stability | **invalid** | active | `r542_split_integrity_result` |
| `pending_opener_rows.r543.v1.invalid_unbalanced_delimiter_pairs` | invalid | null_control | **invalid** | active | `r543_unique_rows_receipt` |
| `pending_opener_four_closer_site_gate.r544.preregistered.v1` | preregistered | full_swap_ceiling | **inconclusive** | superseded by `pending_opener_four_closer_site_gate.r544.native_curly_null.v1` | `—` |
| `pending_opener_four_closer_site_gate.r544.native_curly_null.v1` | complete | capability | **null** | active | `r544_site_gate_result` |
| `pending_opener_three_value_confirmation.r546.preregistered.v1` | preregistered | full_swap_ceiling | **inconclusive** | active | `—` |
| `legacy_bracket_match.r547.invalid_unsealed_rows.v1` | invalid | full_swap_ceiling | **invalid** | active | `r547_legacy_audit` |
| `legacy_bracket_pointer.r547.invalid_dense_decomposition.v1` | invalid | compiled_equivalence | **invalid** | active | `r547_legacy_audit` |
| `legacy_quote_l13h8_parity.r547.invalid_unsealed_rows.v1` | invalid | das_interchange | **invalid** | active | `r547_legacy_audit` |
| `pending_opener_ordered_pair_program.v1` | complete | prospective fixed-vector substitution | **screen** | active | `bracket_l13h8_ordered_pair_displacement_program_ood_validation_v1_result.json` |

**Frozen artifacts:** 69. Paths and SHA-256 hashes are in the canonical JSON record.

The exact opener-term transfer has now been compiled into a fixed program. Six ordered-closer displacement vectors, each 1,152-D and
estimated from 24 SELECT endpoints, were frozen before OOD access. All OOD native cells are 100% capable. On 144 OOD answer-changing
endpoints, the fixed vectors reproduce exact per-prompt opener-term swaps at cosine `0.98986`, relative L2 `0.16973`, norm ratio
`1.08280`, and perfect signs. Both target constructions and all six ordered pairs recur. The selector dispatches exact zero on all 216
answer-preserving endpoints, producing zero logit change. This is a `6,912`-scalar interface program; it does not revive the failed
pair-centered selective-necessity claim and does not infer the pending state from raw text autonomously.

The first simultaneous-program experiment is informative but asymmetric. Bracket-program stress is `2.2515×` the isolated Task14 effect
and Task14 survives with only a `1.863%` interaction, licensing that direction. Task14 stress is only `0.0004896×` the bracket effect,
so the reverse direction and the two-sided parent remain unlicensed rather than being called additive.

**Next:** construct a bracket-side stressor from the Task14 program that is guaranteed live without outcome tuning—for example a frozen
gain derived from artifact norms or a shared-input authority—then rerun only the missing reverse composition direction. Do not rescale
from the opened bracket effects, and do not call mere co-dispatch semantic composition.

### `task.increment.state` — proposed

**Read:** recent numeric state and list relation. **Operation:** apply an increment relation to the numeric state. **Write:** evidence for the next numeric token. **Endpoint:** signed shifted-next-number minus base-next-number logit margin.

| family | role | status |
|---|---|---|
| `coherent_constant_shift` | interchange | proposed |
| `cross_format_operation_swap` | interchange | proposed |
| `incoherent_one_number_edit` | necessity | proposed |
| `operation_preserved_surface_edit` | invariance | proposed |

**Append-only evidence ledger:**
| event | stage | test | verdict | lifecycle | result artifact |
|---|---|---|---|---|---|
| `increment_postattn_rank4_das.legacy.v1` | complete | das_interchange | **held** | active | `postattn_result` |

**Frozen artifacts:** 5. Paths and SHA-256 hashes are in the canonical JSON record.

**Next:** freeze cross-format rows; require number-word transfer and nonincrement numeric controls

### `task.induction.selector_payload` — proposed

**Read:** token equality and source position. **Operation:** select a matching earlier source, then transport its following payload. **Write:** selector-dependent payload contribution to the target-token logits. **Endpoint:** signed donor-answer minus base-answer logit margin at the query continuation.

| family | role | status |
|---|---|---|
| `two_valid_sources_selector_swap` | interchange | proposed |
| `payload_swap_match_preserved` | interchange | proposed |
| `natural_pair_interchange` | interchange | proposed |
| `match_break_payload_preserved` | necessity | proposed |
| `copy_relation_preserved_nuisance_change` | invariance | proposed |

**Append-only evidence ledger:**
| event | stage | test | verdict | lifecycle | result artifact |
|---|---|---|---|---|---|
| `induction_terminal_collateral_failure.legacy.v1` | complete | removal | **failed** | active | `campaign_report` |

**Frozen artifacts:** 5. Paths and SHA-256 hashes are in the canonical JSON record.

**Next:** freeze two-valid-source and payload-swap rows; measure selector and value site ceilings

### `task.successor.pointer` — proposed

**Read:** the final sequence element and coherence/family context. **Operation:** use an identity pointer to retrieve the next element. **Write:** evidence for the successor token. **Endpoint:** signed donor-successor minus base-successor logit margin.

| family | role | status |
|---|---|---|
| `same_family_last_element_swap` | interchange | proposed |
| `coherent_whole_sequence_shift` | interchange | proposed |
| `internal_pointer_imposition` | interchange | proposed |
| `prefix_change_final_pointer_preserved` | invariance | proposed |

**Append-only evidence ledger:**
| event | stage | test | verdict | lifecycle | result artifact |
|---|---|---|---|---|---|
| `successor_cross_family_transfer.legacy.v1` | complete | cross_family_transfer | **failed** | active | `task_report` |
| `successor_layer8_input_ceiling.legacy.v1` | complete | full_swap_ceiling | **null** | active | `task_report` |

**Frozen artifacts:** 5. Paths and SHA-256 hashes are in the canonical JSON record.

**Next:** expand families and test shared-plus-private projectors against failed cross-family transfer


## Summary table

| # | circuit | best (mean) | conc | best (interchange) | conc | agree | members |
|---|---------|-------------|------|--------------------|------|-------|---------|
| 1 | `r.2.0.0` | a8 | 12.28 | a8 | 9.603 | both | 864 / 5,760 |
| 2 | `r.3.0.0` | a16 | 11.87 | a16 | 8.888 | both | 864 / 5,760 |
| 3 | `r.3.0` | a16 | 9.21 | a16 | 6.999 | both | 5,760 / 38,400 |
| 4 | `r.2.0.2` | a8 | 8.99 | a8 | 7.144 | both | 864 / 5,760 |
| 5 | `r.3.0.2` | a16 | 8.56 | a16 | 6.515 | both | 864 / 5,760 |
| 6 | `r.2.0` | a8 | 8.35 | a8 | 6.437 | both | 5,760 / 38,400 |
| 7 | `r.2.0.1` | a8 | 8.35 | a8 | 6.614 | both | 864 / 5,760 |
| 8 | `r.3.0.1` | a16 | 7.57 | a16 | 5.929 | both | 864 / 5,760 |
| 9 | `r.2.1.1` | a8 | 6.16 | a8 | 5.124 | both | 864 / 5,760 |
| 10 | `r.1.2.0` | m16 | 6.12 | m15 | 4.975 | rows-only | 864 / 5,760 |
| 11 | `r.1.0.0` | a16 | 6.06 | a16 | 5.064 | both | 864 / 5,760 |
| 12 | `r.2.2.1` | a8 | 5.93 | a8 | 5.067 | both | 864 / 5,760 |
| 13 | `r.4.1.1` | a16 | 5.47 | a16 | 4.529 | both | 864 / 5,760 |
| 14 | `r.0.0.0` | a3 | 5.07 | a3 | 4.558 | both | 864 / 5,760 |
| 15 | `r.0.0` | a4 | 4.95 | a3 | 4.459 | rows-only | 5,760 / 38,400 |
| 16 | `r.23.2.3` | a8 | 4.93 | a8 | 3.971 | both | 864 / 5,760 |
| 17 | `r.11.1.2` | a8 | 4.90 | a8 | 4.19 | both | 864 / 5,760 |
| 18 | `r.3.1.1` | a17 | 4.80 | a17 | 4.003 | both | 864 / 5,760 |
| 19 | `r.0.0.1` | a3 | 4.60 | a3 | 4.194 | methods-only | 864 / 5,760 |
| 20 | `r.1.2.1` | m16 | 4.51 | m15 | 4.268 | rows-only | 864 / 5,760 |
| 21 | `r.2.2.2` | a6 | 4.37 | a6 | 3.806 | both | 864 / 5,760 |
| 22 | `r.1.2` | m16 | 4.37 | m15 | 3.839 | neither | 5,760 / 38,400 |
| 23 | `r.11.3.1` | a8 | 4.31 | a8 | 3.628 | both | 864 / 5,760 |
| 24 | `r.4.1.0` | a16 | 4.22 | a16 | 3.618 | both | 864 / 5,760 |
| 25 | `r.1.0` | m14 | 4.22 | m14 | 3.308 | both | 5,760 / 38,400 |
| 26 | `r.2.3` | a8 | 4.18 | a8 | 3.479 | both | 5,760 / 38,400 |
| 27 | `r.1.0.2` | m14 | 4.14 | m14 | 3.191 | methods-only | 864 / 5,760 |
| 28 | `r.2.1` | a8 | 4.13 | a8 | 3.401 | both | 5,760 / 38,400 |
| 29 | `r.6.3.0` | a16 | 4.11 | a14 | 3.559 | neither | 864 / 5,760 |
| 30 | `r.1.3.1` | m13 | 4.10 | m14 | 3.485 | neither | 864 / 5,760 |
| 31 | `r.1.1.2` | m16 | 4.08 | m15 | 3.791 | rows-only | 864 / 5,760 |
| 32 | `r.1.1.1` | m16 | 4.08 | m14 | 3.828 | neither | 864 / 5,760 |
| 33 | `r.2.2` | a8 | 4.05 | a8 | 3.371 | both | 5,760 / 38,400 |
| 34 | `r.1.0.3` | m14 | 4.00 | m14 | 3.227 | both | 864 / 5,760 |
| 35 | `r.2.1.0` | a8 | 3.99 | a8 | 3.385 | both | 864 / 5,760 |
| 36 | `r.11.1.1` | a8 | 3.98 | a8 | 3.488 | both | 864 / 5,760 |
| 37 | `r.6.0.1` | a14 | 3.97 | a14 | 3.93 | methods-only | 864 / 5,760 |
| 38 | `r.1.1.0` | m14 | 3.96 | m14 | 3.536 | methods-only | 864 / 5,760 |
| 39 | `r.0.3.0` | a3 | 3.91 | a3 | 3.484 | methods-only | 864 / 5,760 |
| 40 | `r.6.1.0` | m5 | 3.84 | a14 | 3.062 | rows-only | 864 / 5,760 |
| 41 | `r.1.1` | m13 | 3.81 | m14 | 3.357 | rows-only | 5,760 / 38,400 |
| 42 | `r.1.3` | m13 | 3.77 | m13 | 3.298 | both | 5,760 / 38,400 |
| 43 | `r.6.0.3` | m17 | 3.65 | m16 | 3.101 | rows-only | 864 / 5,760 |
| 44 | `r.6.2.2` | m16 | 3.63 | m16 | 3.258 | both | 864 / 5,760 |
| 45 | `r.3.1.0` | a2 | 3.63 | a16 | 3.225 | neither | 864 / 5,760 |
| 46 | `r.1.0.1` | m14 | 3.60 | a16 | 2.969 | neither | 864 / 5,760 |
| 47 | `r.6.2.1` | a9 | 3.59 | a9 | 2.88 | both | 864 / 5,760 |
| 48 | `r.2.2.0` | a8 | 3.55 | a8 | 3.099 | both | 864 / 5,760 |
| 49 | `r.6.1.1` | a16 | 3.53 | a16 | 3.063 | methods-only | 864 / 5,760 |
| 50 | `r.8.1.0` | a3 | 3.50 | a3 | 3.13 | both | 864 / 5,760 |
| 51 | `r.5.0.1` | a16 | 3.40 | a16 | 3.034 | methods-only | 864 / 5,760 |
| 52 | `r.23.2.1` | a8 | 3.38 | a8 | 2.908 | both | 864 / 5,760 |
| 53 | `r.5.3.1` | a15 | 3.33 | a15 | 2.947 | both | 864 / 5,760 |
| 54 | `r.1.3.0` | m13 | 3.23 | a16 | 2.703 | rows-only | 864 / 5,760 |
| 55 | `r.6.0.2` | a9 | 3.19 | a16 | 2.892 | rows-only | 864 / 5,760 |
| 56 | `r.6.2.0` | a16 | 3.17 | a16 | 2.965 | methods-only | 864 / 5,760 |
| 57 | `r.6.3.1` | a16 | 3.13 | a16 | 2.802 | both | 864 / 5,760 |
| 58 | `r.18.2.0` | a7 | 3.10 | a7 | 2.763 | both | 864 / 5,760 |
| 59 | `r.13.2.1` | a3 | 3.04 | a7 | 2.771 | neither | 864 / 5,760 |
| 60 | `r.7.1.1` | a7 | 2.98 | a7 | 2.679 | methods-only | 864 / 5,760 |
| 61 | `r.6.0.0` | a16 | 2.88 | a16 | 2.633 | both | 864 / 5,760 |
| 62 | `r.6.2.3` | a9 | 2.61 | a17 | 2.661 | neither | 864 / 5,760 |

## Per-circuit detail


### 1. `r.2.0.0` — a8, concentration 12.28

864 member positions in a slice of 5,760 (15.0% of the slice).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | a8 12.282 | a14 4.732 | a6 4.145 |
| interchange | a8 9.603 | a6 4.164 | a14 4.06 |

At `a8`: mean|dCE| on members **1.687**, off slice 0.1374, signed dCE on members 0.4605. Second-best component is `a14` at 4.732 — a 2.60x margin.


**DAS (rank 1, held-out):** member dCE 0.2345, concentration 13.045, recovers 0.125 of the full component; overlap with the closed-form direction 0.286.


**Top members** (context → target, dCE when the circuit is ablated):

- `…\n\n#### Ramsgate\n\n#### Sandwich\n\n#### Dover` → `\n`  (dCE -25.41, base CE 29.76)
- `…life\n\n#### Entertainment\n\n#### Shopping\n\n#### Information` → `\n`  (dCE -27.55, base CE 32.04)
- `…life\n\n7PurlB3\n\n7Shopping` → `\n`  (dCE -21.07, base CE 24.34)


### 2. `r.3.0.0` — a16, concentration 11.87

864 member positions in a slice of 5,760 (15.0% of the slice).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | a16 11.866 | a14 6.816 | a15 6.271 |
| interchange | a16 8.888 | a14 5.583 | a15 4.712 |

At `a16`: mean|dCE| on members **0.7896**, off slice 0.0665, signed dCE on members 0.4652. Second-best component is `a14` at 6.816 — a 1.74x margin.


**DAS (rank 1, held-out):** member dCE 0.0824, concentration 10.234, recovers 0.105 of the full component; overlap with the closed-form direction 0.133.


**Story (from the circuit file):** {'blind_name': '', 'program': [['NOT class_other', 'NOT class_subword', 'NOT class_comma']], 'program_bacc': 0.748, 'program_null': 0.597, 'mechanism_level': 'none'}


**Top members** (context → target, dCE when the circuit is ablated):

- `…\n#### Broadstairs\n\n#### Ramsgate\n\n#### Sandwich` → `\n`  (dCE -20.72, base CE 22.84)
- `…\n\n#### Dover\n\n#### Around Dover\n\n#### Rye` → `\n`  (dCE -21.62, base CE 29.5)
- `…\n#### London Highlights\n\n#### History\n\n#### Sights` → `\n`  (dCE -20.86, base CE 24.86)


### 3. `r.3.0` — a16, concentration 9.21

5,760 member positions in a slice of 38,400 (15.0% of the slice).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | a16 9.211 | a14 6.957 | a15 5.347 |
| interchange | a16 6.999 | a14 5.731 | a15 4.115 |

At `a16`: mean|dCE| on members **0.5423**, off slice 0.0589, signed dCE on members 0.3359. Second-best component is `a14` at 6.957 — a 1.32x margin.


**DAS (rank 1, held-out):** member dCE 0.0539, concentration 6.557, recovers 0.094 of the full component; overlap with the closed-form direction 0.336.


**Story (from the circuit file):** {'mechanism_level': 'computational', 'mechanism': {'code': 'heads (16,8),(16,2): z_h(q)=sum over k in top8 |pat_h(q,:)| of pat_h(q,k)*vm_h(k)', 'replication': {'k1': 0.2089, 'k8': 0.1073, 'deletion': 0.5052, 'shuffled_control': 0.358}, 'notes': '79% of named-head function at k=8; a6/a7 bundles not yet coded'}}


### 4. `r.2.0.2` — a8, concentration 8.99

864 member positions in a slice of 5,760 (15.0% of the slice).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | a8 8.987 | a6 4.107 | a14 3.823 |
| interchange | a8 7.144 | a6 3.905 | a14 3.407 |

At `a8`: mean|dCE| on members **1.2345**, off slice 0.1374, signed dCE on members 0.269. Second-best component is `a6` at 4.107 — a 2.19x margin.


**DAS (rank 1, held-out):** member dCE 0.1319, concentration 8.301, recovers 0.094 of the full component; overlap with the closed-form direction 0.223.


**Story (from the circuit file):** {'blind_name': 'input-enrichment for a0 (layer-0 attn) into the a8/a6 double-QK bundle; no surviving behavioral story', 'program': [['NOT class_other', 'NOT prev1_class_other']], 'program_bacc': 0.693, 'program_null': 0.436, 'mechanism_level': 'none', 'mechanism': {'components': ['a8', 'a6'], 'per_component_top_writer': {'a8': {'writer': 'a0', 'ratio': 1.464, 'null_top_ratio': 1.2, 'ENRICHED': True, 'BEATS_NULL': True}, 'a6': {'writer': 'a0', 'ratio': 1.475, 'null_top_ratio': 1.2, 'ENRICHED': True, 'BEATS_NULL': True}}, 'mechanism_line': "a8 and a6 (r.2.0 double-QK bundle) both ENRICHED for writer a0 (layer-0 attn) in their input: ratio 1.464/1.475 vs null 1.20 (BEATS_NULL); a0's absolute share is tiny (~0.0007) vs m0's dominant ~0.42-0.44 (ratio ~0.99, not selective).", 'escalation_status': 'NOT ESCALATED: ENRICHED writer (a0) named but qk_writer_decomp-style writer-pair QK decomposition not run (out of scope for this pass); flagged for ladder follow-up. Note a0 share of input is small in absolute terms (~0.0007) despite ratio enrichment.', 'leaf_mech_file': 'leaf_mech/r.2.0.2.json'}}


**Top members** (context → target, dCE when the circuit is ablated):

- `… Braymer. The New Zealand rabbits were purchased by the Missouri State` → ` Rabbit`  (dCE 0.11, base CE 11.4)
- `…–]Subhazard 48 Punkte49 Punkte50 Punkte` → ` 9`  (dCE -0.51, base CE 5.52)
- `…. But, eventually hiding wasn’t enough and H.` → ` erect`  (dCE -2.83, base CE 5.41)


### 5. `r.3.0.2` — a16, concentration 8.56

864 member positions in a slice of 5,760 (15.0% of the slice).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | a16 8.558 | a14 5.762 | a15 5.396 |
| interchange | a16 6.515 | a14 4.816 | a15 4.179 |

At `a16`: mean|dCE| on members **0.5694**, off slice 0.0665, signed dCE on members 0.3292. Second-best component is `a14` at 5.762 — a 1.49x margin.


**DAS (rank 1, held-out):** member dCE 0.0251, concentration 4.628, recovers 0.041 of the full component; overlap with the closed-form direction 0.094.


**Story (from the circuit file):** {'blind_name': 'attn-input-writer decomp: a14 dominant writer into a15/16/17', 'mechanism_level': 'none', 'program': [['class_ind']], 'program_bacc': 0.613, 'program_null': 0.593, 'mechanism': 'a15/a16/a17 attn (r.3.0 PCA bundles, incl. layer-16); top writer a14 enriched 2.85/2.96/2.84x (null ~1.1-1.25) into all three; a16 also feeds a17 (2.58x)', 'behavior': 'no behavioral claim survives base-rate testing (capitalized/space_word/subword/punct/newline all fail; digit nominally passes but n=3, whole population, not credible)'}


**Top members** (context → target, dCE when the circuit is ablated):

- `… services and hosting globally," said Henderson, CTO and president of` → ` academic`  (dCE -0.0, base CE 10.92)
- `…-collected fishery data, including fishery data held by` → ` RF`  (dCE 0.65, base CE 7.61)
- `…rdinator will have overall responsibility for the technical quality of` → ` Merlin`  (dCE -0.24, base CE 8.32)


### 6. `r.2.0` — a8, concentration 8.35

5,760 member positions in a slice of 38,400 (15.0% of the slice).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | a8 8.347 | a6 3.662 | a14 3.612 |
| interchange | a8 6.437 | a6 3.438 | a14 3.187 |

At `a8`: mean|dCE| on members **0.9828**, off slice 0.1177, signed dCE on members 0.3151. Second-best component is `a6` at 3.662 — a 2.28x margin.


**Top members** (context → target, dCE when the circuit is ablated):

- `…\n\n#### Dover\n\n#### Around Dover\n\n#### Rye` → `\n`  (dCE -27.48, base CE 29.5)
- `…\n\n#### Ramsgate\n\n#### Sandwich\n\n#### Dover` → `\n`  (dCE -27.14, base CE 29.76)
- `…life\n\n7PurlB3\n\n7Shopping` → `\n`  (dCE -19.73, base CE 24.34)


### 7. `r.2.0.1` — a8, concentration 8.35

864 member positions in a slice of 5,760 (15.0% of the slice).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | a8 8.346 | a6 5.437 | a14 3.753 |
| interchange | a8 6.614 | a6 4.741 | a14 3.452 |

At `a8`: mean|dCE| on members **1.1464**, off slice 0.1374, signed dCE on members 0.1527. Second-best component is `a6` at 5.437 — a 1.54x margin.


**Story (from the circuit file):** {'blind_name': 'r.2.0 double-QK bundle (a6/a8); input writer composition NOT enriched (ENRICHED_STABLE=False both); no surviving behavioral story', 'program': [['NOT class_other', 'NOT prev1_class_other']], 'program_bacc': 0.682, 'program_null': 0.561, 'mechanism_level': 'none', 'mechanism_line': "a6 and a8 (r.2.0 double-QK bundle) input writer composition does NOT distinguish members under 5-draw bootstrap: top writer both components is a4 (a6: mean 1.134 [1.063-1.212]; a8: mean 1.132 [1.063-1.211]), ENRICHED_STABLE=False for both (single-draw ENRICHED was already False, so no a0-style collapse to retract). Dominant writer by input mass is m0 (~0.43-0.45 share) but its ratio is flat (~0.986-0.987, not selective) -- consistent with the universal identity code, not this leaf's mechanism.", 'behavior_line': 'no behavioral claim survives base-rate testing (12 class x direction pairs tested, alpha=0.10/12=0.0083, none robust)', 'leaf_mech_file': 'leaf_mech/r.2.0.1.json'}


**Top members** (context → target, dCE when the circuit is ablated):

- `…\n|College Teams||High Schools|\nMarch 4, 2012` → `Tweet`  (dCE -0.49, base CE 12.39)
- `…\n|College Teams||High Schools|\nJanuary 19, 2012` → `1`  (dCE 0.78, base CE 14.49)
- `…\n|College Teams||High Schools|\nMay 24, 2012` → `A`  (dCE -1.67, base CE 10.85)


### 8. `r.3.0.1` — a16, concentration 7.57

864 member positions in a slice of 5,760 (15.0% of the slice).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | a16 7.569 | a14 4.89 | a17 4.807 |
| interchange | a16 5.929 | a14 4.242 | a17 4.193 |

At `a16`: mean|dCE| on members **0.5036**, off slice 0.0665, signed dCE on members 0.2977. Second-best component is `a14` at 4.89 — a 1.55x margin.


**Top members** (context → target, dCE when the circuit is ablated):

- `…\n\n#### Woburn\n\n#### Waddesdon` → `\n`  (dCE -2.51, base CE 16.09)
- `…#### Margate\n\n#### Broadstairs\n\n#### Ramsgate` → `\n`  (dCE -1.42, base CE 19.99)
- `…#### Festivals & Events\n\n#### Sleeping\n\n#### Eating` → `\n`  (dCE -2.28, base CE 22.08)


### 9. `r.2.1.1` — a8, concentration 6.16

864 member positions in a slice of 5,760 (15.0% of the slice).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | a8 6.156 | a14 4.029 | a6 3.837 |
| interchange | a8 5.124 | a14 3.664 | a6 3.58 |

At `a8`: mean|dCE| on members **0.916**, off slice 0.1488, signed dCE on members 0.1477. Second-best component is `a14` at 4.029 — a 1.53x margin.


**Top members** (context → target, dCE when the circuit is ablated):

- `…designed by John Nash). From Pall Mall, walk along Cocks` → `pur`  (dCE -0.48, base CE 2.21)
- `… Frith St, W1; s £216, d/` → `ste`  (dCE -0.9, base CE 15.8)
- `… ; www.shakespearesglobe.com; 21 New` → ` Globe`  (dCE -1.51, base CE 7.83)


### 10. `r.1.2.0` — m16, concentration 6.12

864 member positions in a slice of 5,760 (15.0% of the slice).

> **Confidence: rows-only** — stable across a row split, but the two interventions name different components. Its held-out concentration is 5.6996, so the circuit still localises; it is the single component NAME that is not settled (§2061).

> **Band-localised, not component-localised.** The two methods disagree by one or two layers inside the `m13`–`m16` band, which is the signature of a circuit spread across adjacent MLPs rather than sitting on one.

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | m16 6.119 | m15 4.933 | m13 4.888 |
| interchange | m15 4.975 | m16 4.728 | m13 4.688 |

At `m16`: mean|dCE| on members **1.8821**, off slice 0.3076, signed dCE on members 1.4723. Second-best component is `m15` at 4.933 — a 1.24x margin.


**Story (from the circuit file):** {'blind_name': 'Members are hard-to-predict tokens in venue/address listings; machinery hurts prediction of common words (off, d) but helps rarer proper-noun/genre completions (house, Banqueting, Queen).', 'program': [['class_subword', 'NOT is_newline'], ['mid_word']], 'program_bacc': 0.542, 'program_null': 0.439, 'mechanism_level': 'none', 'redteam_hits': '3/3', 'redteam_verdict': 'HELD'}


**Top members** (context → target, dCE when the circuit is ablated):

- `… Lane, Cowcross St, EC1; s £222,` → ` d`  (dCE -0.82, base CE 8.3)
- `… most popular villages are busy in summer, it's easy to get` → ` off`  (dCE -1.03, base CE 5.37)
- `… Rd Arches)\n\nSeafront club hosting indie,` → ` house`  (dCE 0.84, base CE 4.94)


### 11. `r.1.0.0` — a16, concentration 6.06

864 member positions in a slice of 5,760 (15.0% of the slice).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | a16 6.064 | a14 4.6 | m14 4.427 |
| interchange | a16 5.064 | a14 4.322 | m14 3.213 |

At `a16`: mean|dCE| on members **0.4508**, off slice 0.0743, signed dCE on members 0.2994. Second-best component is `a14` at 4.6 — a 1.32x margin.


**Story (from the circuit file):** {'blind_name': '', 'program': [['NOT prev1_starts_space', 'NOT class_subword', 'NOT prev1_class_ind'], ['circ_r_3_1_0']], 'program_bacc': 0.721, 'program_null': 0.675, 'mechanism_level': 'none'}


**Top members** (context → target, dCE when the circuit is ablated):

- `…eth that sit closest to the river.\n\nSouth Bank\n` → `\n`  (dCE 6.27, base CE 22.8)
- `… lanes leading off it.\n\nTrulloITALIAN\n` → `\n`  (dCE 5.43, base CE 21.6)
- `…\n2130 St Mary Axe\n\n22Heron Tower\n` → `\n`  (dCE 6.31, base CE 20.18)


### 12. `r.2.2.1` — a8, concentration 5.93

864 member positions in a slice of 5,760 (15.0% of the slice).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | a8 5.933 | a6 4.529 | a14 3.686 |
| interchange | a8 5.067 | a6 4.191 | a14 3.409 |

At `a8`: mean|dCE| on members **0.8839**, off slice 0.149, signed dCE on members 0.1215. Second-best component is `a6` at 4.529 — a 1.31x margin.


**Top members** (context → target, dCE when the circuit is ablated):

- `… NW8; tours adult/child £18/12; ` → ` t`  (dCE -0.24, base CE 13.86)
- `…1Sights\n\n1Madame TussaudsB` → `1`  (dCE 0.25, base CE 8.17)
- `…rant preserve.\n\n8Information\n\nTourist OfficeT` → `OUR`  (dCE 1.83, base CE 9.1)


### 13. `r.4.1.1` — a16, concentration 5.47

864 member positions in a slice of 5,760 (15.0% of the slice).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | a16 5.466 | a14 4.202 | a12 3.986 |
| interchange | a16 4.529 | a14 3.704 | a12 3.332 |

At `a16`: mean|dCE| on members **0.3993**, off slice 0.0731, signed dCE on members 0.2903. Second-best component is `a14` at 4.202 — a 1.30x margin.


**Story (from the circuit file):** {'blind_name': 'No strong single-writer mechanism found (a12/m1 top ratios 1.06/1.01 vs thresholds 1.31/1.30); no behavioral claim survives base-rate testing (12 class/direction pairs).', 'program': [['class_ind']], 'program_bacc': 0.559, 'program_null': 0.512, 'mechanism_level': 'none', 'behavior_line': 'no behavioral claim survives base-rate testing', 'story_test_sweep': {'n_tests': 12, 'alpha': 0.0083, 'kinds_x_directions': 'subword,space_word,digit,punct,capitalized,newline x {help,hurt}', 'best_seed_pass_frac': 0.4, 'best_population_p': 0.1135, 'all_ROBUST_V2': False}}


**Top members** (context → target, dCE when the circuit is ablated):

- `…umi in Japan. Later, yields reduced and traditional whaling w` → `aned`  (dCE 2.01, base CE 2.81)
- `…s challenge, however, will be convincing its own investors that the` → ` African`  (dCE 1.36, base CE 6.15)
- `… addition, Nagle said, the district must find two classrooms for` → ` Leh`  (dCE 0.95, base CE 5.76)


### 14. `r.0.0.0` — a3, concentration 5.07

864 member positions in a slice of 5,760 (15.0% of the slice).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | a3 5.072 | m5 4.882 | a4 4.763 |
| interchange | a3 4.558 | a2 4.027 | m5 3.996 |

At `a3`: mean|dCE| on members **1.51**, off slice 0.2977, signed dCE on members 0.1555. Second-best component is `m5` at 4.882 — a 1.04x margin.


**Story (from the circuit file):** {'blind_name': '', 'program': [['circ_r_1_1_2', 'NOT is_punct'], ['NOT class_other', 'NOT class_ind', 'NOT class_digit']], 'program_bacc': 0.744, 'program_null': 0.516, 'mechanism_level': 'none'}


**Top members** (context → target, dCE when the circuit is ablated):

- `…3\n\n34RevengeF4\n\n3Entertainment` → `\n`  (dCE -10.6, base CE 29.94)
- `…\n\n#### Arundel\n\n#### Chichester` → `\n`  (dCE -12.46, base CE 29.36)
- `… Chichester\n\n#### Farnham\n\n#### Hindhead` → `\n`  (dCE -15.01, base CE 27.0)


### 15. `r.0.0` — a4, concentration 4.95

5,760 member positions in a slice of 38,400 (15.0% of the slice).

> **Confidence: rows-only** — stable across a row split, but the two interventions name different components. Its held-out concentration is 5.0709, so the circuit still localises; it is the single component NAME that is not settled (§2061).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | a4 4.953 | m6 4.791 | m5 4.779 |
| interchange | a3 4.459 | a2 4.02 | m5 4.011 |

At `a4`: mean|dCE| on members **1.7276**, off slice 0.3488, signed dCE on members 0.141. Second-best component is `m6` at 4.791 — a 1.03x margin.


**Top members** (context → target, dCE when the circuit is ablated):

- `…\n44Pitt Cue CoB4\n\nPortraitE6` → `\n`  (dCE -29.26, base CE 30.56)
- `…Ye Olde Cheshire CheeseB3\n\n3Entertainment` → `\n`  (dCE -15.69, base CE 23.3)
- `…3\n\n34RevengeF4\n\n3Entertainment` → `\n`  (dCE -15.03, base CE 29.94)


### 16. `r.23.2.3` — a8, concentration 4.93

864 member positions in a slice of 5,760 (15.0% of the slice).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | a8 4.931 | a6 2.579 | m11 2.507 |
| interchange | a8 3.971 | a6 2.447 | m11 2.335 |

At `a8`: mean|dCE| on members **0.7349**, off slice 0.149, signed dCE on members 0.0693. Second-best component is `a6` at 2.579 — a 1.91x margin.


**DAS (rank 1, held-out):** member dCE 0.0222, concentration 3.53, recovers 0.027 of the full component; overlap with the closed-form direction 0.006.


**Story (from the circuit file):** {'blind_name': 'no behavioral claim survives base-rate testing', 'program': [['NOT circ_r_2_1_0', 'prev1_starts_space', 'NOT prev1_class_sentend']], 'program_bacc': 0.531, 'program_null': 0.51, 'mechanism_level': 'none', 'behavioral_tests': {'punct_pred_help_true': {'kind': 'punct', 'pred_help': True, 'n_available': 29, 'per_seed': [{'seed': 1, 'hits': 2, 'n': 5, 'p_value': 0.7493}, {'seed': 2, 'hits': 3, 'n': 5, 'p_value': 0.4137}, {'seed': 3, 'hits': 3, 'n': 5, 'p_value': 0.4137}, {'seed': 4, 'hits': 4, 'n': 5, 'p_value': 0.135}, {'seed': 11, 'hits': 3, 'n': 5, 'p_value': 0.4137}], 'seed_pass_frac': 0.0, 'population': {'n': 29, 'hits': 17, 'base_rate_help': 0.454, 'expected_hits': 13.16, 'p_value': 0.1065, 'beats_base_rate': False}, 'ROBUST': False, 'n_tests': 2, 'alpha': 0.05, 'ROBUST_V2': False, 'gate_note': 'use ROBUST_V2; ROBUST v1 is underpowered'}, 'punct_pred_help_false': {'kind': 'punct', 'pred_help': False, 'n_available': 29, 'per_seed': [{'seed': 1, 'hits': 3, 'n': 5, 'p_value': 0.5863}, {'seed': 2, 'hits': 2, 'n': 5, 'p_value': 0.865}, {'seed': 3, 'hits': 2, 'n': 5, 'p_value': 0.865}, {'seed': 4, 'hits': 1, 'n': 5, 'p_value': 0.9808}, {'seed': 11, 'hits': 2, 'n': 5, 'p_value': 0.865}], 'seed_pass_frac': 0.0, 'population': {'n': 29, 'hits': 12, 'base_rate_help': 0.454, 'expected_hits': 15.84, 'p_value': 0.9472, 'beats_base_rate': False}, 'ROBUST': False, 'n_tests': 2, 'alpha': 0.05, 'ROBUST_V2': False, 'gate_note': 'use ROBUST_V2; ROBUST v1 is underpowered'}, 'n_tests_declared': 2, 'alpha': 0.05, 'note': 'sibling leaves r.13.2.1 and r.18.2.0 carry VERIFIED punctuation claims; tested here as pre-declared hypothesis per assignment; neither direction cleared ROBUST_V2 (p=0.1065 helps-dir vs alpha 0.05, p=0.9472 hurts-dir)'}}


**Top members** (context → target, dCE when the circuit is ablated):

- `…- published: 19 Mar 2013\n-` → ` views`  (dCE 0.12, base CE 3.44)
- `… pitching statistics (DIPS)\n- Defensive Runs Saved (` → `D`  (dCE 0.42, base CE 1.19)
- `…It helps to fight against nuclear, biological and chemical weapons (NR` → `BC`  (dCE 0.19, base CE 11.32)


### 17. `r.11.1.2` — a8, concentration 4.90

864 member positions in a slice of 5,760 (15.0% of the slice).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | a8 4.904 | a14 4.101 | a16 3.239 |
| interchange | a8 4.19 | a14 3.556 | a16 2.798 |

At `a8`: mean|dCE| on members **0.7343**, off slice 0.1497, signed dCE on members 0.0819. Second-best component is `a14` at 4.101 — a 1.20x margin.


**DAS (rank 1, held-out):** member dCE 0.0396, concentration 4.366, recovers 0.043 of the full component; overlap with the closed-form direction 0.07.


**Story (from the circuit file):** {'blind_name': 'punctuation-target CE reduction (weak, single mechanism not found)', 'program': [['prev1_seen_before']], 'program_bacc': 0.587, 'program_null': 0.587, 'mechanism_level': 'none', 'behavior': 'Members at punctuation targets (5.9% of members) get pushed toward lower CE 71% of the time vs 47% base rate (ROBUST_V2, p=0.0007, n=51); at non-punctuation targets the push is near base rate (46%).', 'behavior_test': {'kind': 'punct', 'pred_help': True, 'n_available': 51, 'per_seed': [{'seed': 1, 'hits': 4, 'n': 5, 'p_value': 0.156}, {'seed': 2, 'hits': 5, 'n': 5, 'p_value': 0.0238}, {'seed': 3, 'hits': 4, 'n': 5, 'p_value': 0.156}, {'seed': 4, 'hits': 5, 'n': 5, 'p_value': 0.0238}, {'seed': 11, 'hits': 3, 'n': 5, 'p_value': 0.4502}], 'seed_pass_frac': 0.4, 'population': {'n': 51, 'hits': 36, 'base_rate_help': 0.473, 'expected_hits': 24.14, 'p_value': 0.0007, 'beats_base_rate': True}, 'ROBUST': False, 'n_tests': 12, 'alpha': 0.0083, 'ROBUST_V2': True, 'gate_note': 'use ROBUST_V2; ROBUST v1 is underpowered'}}


**Top members** (context → target, dCE when the circuit is ablated):

- `…influenced by western European harvest festivals, and festivals of the` → ` dead`  (dCE -1.67, base CE 3.74)
- `… essay on “How I met and chose my bridal party` → `.`  (dCE -1.2, base CE 2.79)
- `… to the Missouri Food Bank Association.\nCodi Coats of` → ` Bray`  (dCE 3.97, base CE 2.41)


### 18. `r.3.1.1` — a17, concentration 4.80

864 member positions in a slice of 5,760 (15.0% of the slice).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | a17 4.802 | a16 4.223 | a15 3.5 |
| interchange | a17 4.003 | a16 3.743 | a15 3.059 |

At `a17`: mean|dCE| on members **0.4446**, off slice 0.0926, signed dCE on members 0.3147. Second-best component is `a16` at 4.223 — a 1.14x margin.


**Top members** (context → target, dCE when the circuit is ablated):

- `…#### Festivals & Events\n\n#### Sleeping\n\n#### Eating` → `\n`  (dCE -1.43, base CE 22.08)
- `…#### Brighton & Hove\n\n#### Arundel\n` → `\n`  (dCE 1.98, base CE 16.82)
- `… 2am Thu-Sat, noon-midnight Sun)\n` → `\n`  (dCE -1.42, base CE 12.84)


### 19. `r.0.0.1` — a3, concentration 4.60

864 member positions in a slice of 5,760 (15.0% of the slice).

> **Confidence: methods-only** — both interventions agree, but the argmax moves when the rows are split. On a held-out row split the argmax moves `m5` -> `a3`. Its held-out concentration is 4.4341, so the circuit still localises; it is the single component NAME that is not settled (§2061).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | a3 4.604 | a4 4.516 | m5 4.475 |
| interchange | a3 4.194 | m5 3.687 | a2 3.654 |

At `a3`: mean|dCE| on members **1.3709**, off slice 0.2977, signed dCE on members 0.1213. Second-best component is `a4` at 4.516 — a 1.02x margin.


**Story (from the circuit file):** {'blind_name': '', 'program': [['NOT class_other', 'NOT class_digit', 'NOT prev2_class_bclose']], 'program_bacc': 0.814, 'program_null': 0.55, 'mechanism_level': 'surface'}


**Top members** (context → target, dCE when the circuit is ablated):

- `…\n\nABleak House\n\nAJeake's House` → `\n`  (dCE 3.51, base CE 23.6)
- `…Museum of LondonD2\n\n17PinnacleG3` → `\n`  (dCE 5.15, base CE 18.99)
- `…BarrafinaF6\n\n27BarrafinaD3` → `\n`  (dCE 3.59, base CE 6.97)


### 20. `r.1.2.1` — m16, concentration 4.51

864 member positions in a slice of 5,760 (15.0% of the slice).

> **Confidence: rows-only** — stable across a row split, but the two interventions name different components. Its held-out concentration is 4.576, so the circuit still localises; it is the single component NAME that is not settled (§2061).

> **Band-localised, not component-localised.** The two methods disagree by one or two layers inside the `m13`–`m16` band, which is the signature of a circuit spread across adjacent MLPs rather than sitting on one.

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | m16 4.51 | m15 4.188 | m13 4.077 |
| interchange | m15 4.268 | m14 3.935 | a17 3.931 |

At `m16`: mean|dCE| on members **1.3873**, off slice 0.3076, signed dCE on members 0.9845. Second-best component is `m15` at 4.188 — a 1.08x margin.


**Top members** (context → target, dCE when the circuit is ablated):

- `… B&B\n\n( ` → ` MAP`  (dCE -0.91, base CE 10.2)
- `…s ShedMARKET, RESTAURANT\n\n(` → ` `  (dCE -0.76, base CE 11.81)
- `…;  h10am-5.50pm;  t` → `South`  (dCE -0.2, base CE 13.34)


### 21. `r.2.2.2` — a6, concentration 4.37

864 member positions in a slice of 5,760 (15.0% of the slice).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | a6 4.373 | a8 4.095 | a14 3.064 |
| interchange | a6 3.806 | a8 3.597 | a14 2.923 |

At `a6`: mean|dCE| on members **0.9663**, off slice 0.221, signed dCE on members -0.0159. Second-best component is `a8` at 4.095 — a 1.07x margin.


**Top members** (context → target, dCE when the circuit is ablated):

- `… Museum.\n\n8Getting There & Away\n\nBoat` → `\n`  (dCE -0.32, base CE 19.94)
- `…\n\nTo/From the Airports\n\nGatwick` → `\n`  (dCE 0.29, base CE 22.03)
- `… miles northwest of the centre at Withdean, from where bus` → ` 27`  (dCE 0.51, base CE 9.93)


### 22. `r.1.2` — m16, concentration 4.37

5,760 member positions in a slice of 38,400 (15.0% of the slice).

> **Confidence: neither** — the named component is not stable across rows and the two interventions disagree. On a held-out row split the argmax moves `m16` -> `m13`. Its held-out concentration is 4.2448, so the circuit still localises; it is the single component NAME that is not settled (§2061).

> **Band-localised, not component-localised.** The two methods disagree by one or two layers inside the `m13`–`m16` band, which is the signature of a circuit spread across adjacent MLPs rather than sitting on one.

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | m16 4.368 | m13 4.217 | m15 4.142 |
| interchange | m15 3.839 | m13 3.823 | m14 3.733 |

At `m16`: mean|dCE| on members **1.133**, off slice 0.2594, signed dCE on members 0.822. Second-best component is `m13` at 4.217 — a 1.04x margin.


**Top members** (context → target, dCE when the circuit is ablated):

- `… Embankment or Temple) displays a wealth of 14th-` → ` to`  (dCE 1.18, base CE 6.21)
- `…motelB&B\n\n(  MAP   G` → `OO`  (dCE 0.27, base CE 5.27)
- `… since 1905, riverside Formans boasts prime views over the Olympic` → ` stadium`  (dCE -1.34, base CE 7.28)


### 23. `r.11.3.1` — a8, concentration 4.31

864 member positions in a slice of 5,760 (15.0% of the slice).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | a8 4.313 | a14 3.224 | a3 3.079 |
| interchange | a8 3.628 | a14 2.902 | a16 2.705 |

At `a8`: mean|dCE| on members **0.6339**, off slice 0.147, signed dCE on members 0.046. Second-best component is `a14` at 3.224 — a 1.34x margin.


**DAS (rank 1, held-out):** member dCE 0.0636, concentration 3.754, recovers 0.084 of the full component; overlap with the closed-form direction 0.228.


**Story (from the circuit file):** {'blind_name': 'templated-completion vs list/numeric-break selector', 'text': 'Machinery detects atypical token transitions; pushes toward templated completions -- helps predictable collocations (time units, common verb-noun pairs) but actively hurts unexpected list/numeric continuations.', 'program': [['NOT class_other', 'NOT is_newline']], 'program_bacc': 0.556, 'program_null': 0.551, 'mechanism_level': 'none'}


**Top members** (context → target, dCE when the circuit is ablated):

- `… 4 with Caulker and Muric and it seems tighter and` → ` Does`  (dCE -2.8, base CE 13.96)
- `…30 wake up and take systemic enzyme\n5-5.30` → `pm`  (dCE 1.88, base CE 3.84)
- `… face and affect how much capital they need.\nOr it can` → ` overest`  (dCE 1.22, base CE 6.8)


### 24. `r.4.1.0` — a16, concentration 4.22

864 member positions in a slice of 5,760 (15.0% of the slice).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | a16 4.217 | a14 3.719 | a12 2.969 |
| interchange | a16 3.618 | a14 3.395 | a13 2.8 |

At `a16`: mean|dCE| on members **0.3081**, off slice 0.0731, signed dCE on members 0.2006. Second-best component is `a14` at 3.719 — a 1.13x margin.


**Story (from the circuit file):** {'blind_name': 'Two-signed circuit: hurts predictable words (said, tying, opener) but helps concrete nouns (restaurants); wrong for short function/number words it also suppresses.', 'program': [['circ_r_0_0_1']], 'program_bacc': 0.567, 'program_null': 0.456, 'mechanism_level': 'none', 'redteam_hits': '3/3', 'redteam_verdict': 'HELD'}


**Top members** (context → target, dCE when the circuit is ablated):

- `… the 878th for Boeheim, moving him within one of` → ` tying`  (dCE -4.22, base CE 7.82)
- `… tax oxygen,” said Hodges.\nFranchot` → ` said`  (dCE -2.99, base CE 4.43)
- `… score since taking a punt back against Samford in the 2010 season` → ` opener`  (dCE -1.73, base CE 2.08)


### 25. `r.1.0` — m14, concentration 4.22

5,760 member positions in a slice of 38,400 (15.0% of the slice).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | m14 4.217 | m13 4.145 | m15 3.907 |
| interchange | m14 3.308 | m13 3.222 | m15 3.08 |

At `m14`: mean|dCE| on members **0.5403**, off slice 0.1281, signed dCE on members 0.0613. Second-best component is `m13` at 4.145 — a 1.02x margin.


**Top members** (context → target, dCE when the circuit is ablated):

- `… ska and reggae beats.\n\n#### Islington\n` → `\n`  (dCE 6.28, base CE 21.62)
- `…eth that sit closest to the river.\n\nSouth Bank\n` → `\n`  (dCE 9.3, base CE 22.8)
- `….\n\nJubilee LibraryINTERNET\n\n( ` → ` G`  (dCE 4.79, base CE 11.66)


### 26. `r.2.3` — a8, concentration 4.18

5,760 member positions in a slice of 38,400 (15.0% of the slice).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | a8 4.178 | a6 3.391 | a14 3.046 |
| interchange | a8 3.479 | a6 3.02 | a14 2.871 |

At `a8`: mean|dCE| on members **0.4919**, off slice 0.1177, signed dCE on members 0.0577. Second-best component is `a6` at 3.391 — a 1.23x margin.


**Top members** (context → target, dCE when the circuit is ablated):

- `… to Stay\n\nAHaymarket Hotel\n\nAZetter Hotel` → `\n`  (dCE 3.71, base CE 25.8)
- `…\n\n#### Battle\n\n#### Hastings\n\n#### Eastbourne` → `\n`  (dCE 2.61, base CE 22.77)
- `…\n\n### Best Places to Stay\n\nAHaymarket Hotel` → `\n`  (dCE 8.11, base CE 16.41)


### 27. `r.1.0.2` — m14, concentration 4.14

864 member positions in a slice of 5,760 (15.0% of the slice).

> **Confidence: methods-only** — both interventions agree, but the argmax moves when the rows are split. On a held-out row split the argmax moves `m12` -> `m14`. Its held-out concentration is 3.7273, so the circuit still localises; it is the single component NAME that is not settled (§2061).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | m14 4.138 | m12 3.897 | m13 3.876 |
| interchange | m14 3.191 | m13 3.021 | m15 3.016 |

At `m14`: mean|dCE| on members **0.6422**, off slice 0.1552, signed dCE on members -0.028. Second-best component is `m12` at 3.897 — a 1.06x margin.


**Top members** (context → target, dCE when the circuit is ablated):

- `…** and the Peter Harrison Planetarium (  GOOGLE` → ` MAP`  (dCE 0.94, base CE 3.63)
- `…Post OfficePOST OFFICE\n\n(  GOOGLE` → ` MAP`  (dCE 1.17, base CE 3.08)
- `…OUTIQUE HOTEL\n\n(  GOOGLE` → ` MAP`  (dCE 1.06, base CE 1.97)


### 28. `r.2.1` — a8, concentration 4.13

5,760 member positions in a slice of 38,400 (15.0% of the slice).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | a8 4.126 | a6 3.334 | a14 3.201 |
| interchange | a8 3.401 | a6 2.971 | a14 2.879 |

At `a8`: mean|dCE| on members **0.4859**, off slice 0.1177, signed dCE on members 0.0388. Second-best component is `a6` at 3.334 — a 1.24x margin.


**Top members** (context → target, dCE when the circuit is ablated):

- `… Swain's Lane, N6; East Cemetery adult/child` → ` £`  (dCE 5.19, base CE 5.53)
- `…\n#### Broadstairs\n\n#### Ramsgate\n\n#### Sandwich` → `\n`  (dCE 3.19, base CE 22.84)
- `… House, a Georgian mansion built in 1718 for the wealthy hop` → ` merchant`  (dCE 3.83, base CE 6.8)


### 29. `r.6.3.0` — a16, concentration 4.11

864 member positions in a slice of 5,760 (15.0% of the slice).

> **Confidence: neither** — the named component is not stable across rows and the two interventions disagree. On a held-out row split the argmax moves `a16` -> `a14`. Its held-out concentration is 4.0277, so the circuit still localises; it is the single component NAME that is not settled (§2061).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | a16 4.111 | a14 4.035 | a8 3.207 |
| interchange | a14 3.559 | a16 3.537 | a8 2.843 |

At `a16`: mean|dCE| on members **0.305**, off slice 0.0742, signed dCE on members 0.1579. Second-best component is `a14` at 4.035 — a 1.02x margin.


**Story (from the circuit file):** {'blind_name': 'Hard-to-predict delimiters (newlines, semicolons) in travel-guide listings/headings; circuit usually pushes the wrong token, raising loss on the majority, helping only a minority.', 'program': [['circ_r_2_1_0'], ['NOT starts_space', 'NOT prev_newline', 'NOT prev1_class_comma']], 'program_bacc': 0.677, 'program_null': 0.3, 'mechanism_level': 'none', 'redteam': {'literal_membership_reading': {'hits': 2, 'of': 3, 'note': 'SOP wording as originally read (pre round-1/2 edit): does story structurally describe the example? 53433 (target " p", not a delimiter) = miss; 51111, 54271 (newline at listing/heading boundary) = hit.'}, 'causal_direction_reading': {'hits': 0, 'of': 3, 'note': 'SOP wording as corrected by concurrent round-2 edit (landed mid-task, discovered after my step 5 ran): does story predict machinery helps vs hurts from dCE sign? All 3 random examples have POSITIVE dCE (circuit helps = minority behavior, 39.7% of members) but story gives no context-level rule to distinguish minority-helps from majority-hurts cases, so it cannot correctly call any of the 3 in advance.'}, 'verdict': 'weak (per corrected/current SOP step-5 wording: causal-direction hits=0/3 <=1/3 threshold); flagged for revision -- story lacks a structural rule for which members are in the helping-minority vs hurting-majority split', 'hits': 0, 'of': 3}}


**Top members** (context → target, dCE when the circuit is ablated):

- `…\n#### Broadstairs\n\n#### Ramsgate\n\n#### Sandwich` → `\n`  (dCE 6.79, base CE 22.84)
- `…\n\n34RevengeF4\n\n3Entertainment\n` → `\n`  (dCE -4.55, base CE 24.4)
- `…chen.co.uk; 11 Sicilian Ave, WC1` → `;`  (dCE -4.22, base CE 7.02)


### 30. `r.1.3.1` — m13, concentration 4.10

864 member positions in a slice of 5,760 (15.0% of the slice).

> **Confidence: neither** — the named component is not stable across rows and the two interventions disagree. On a held-out row split the argmax moves `m14` -> `m13`. Its held-out concentration is 4.0055, so the circuit still localises; it is the single component NAME that is not settled (§2061).

> **Band-localised, not component-localised.** The two methods disagree by one or two layers inside the `m13`–`m16` band, which is the signature of a circuit spread across adjacent MLPs rather than sitting on one.

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | m13 4.098 | m14 4.085 | m12 3.942 |
| interchange | m14 3.485 | m13 3.458 | m12 3.3 |

At `m13`: mean|dCE| on members **0.7196**, off slice 0.1756, signed dCE on members 0.1091. Second-best component is `m14` at 4.085 — a 1.00x margin.


**Story (from the circuit file):** {'blind_name': 'Members split near-evenly: pushes CE down for clean word/name completions (cooperate, Koch, shall); pushes CE up for garbled/sub-word fragments (mojibake, angers, ication).', 'program': [['class_subword', 'NOT is_newline']], 'program_bacc': 0.609, 'program_null': 0.47, 'mechanism_level': 'none'}


**Top members** (context → target, dCE when the circuit is ablated):

- `…\nWe have developed an interactive map driven website. Often enough,` → ` cooperate`  (dCE -0.87, base CE 17.81)
- `… first stint with the Marlins from 2003-05, even pulling Billy` → ` Koch`  (dCE -0.77, base CE 10.0)
- `…z, gave away seats from “The Madhouse on Madison` → `�`  (dCE 0.65, base CE 2.55)


### 31. `r.1.1.2` — m16, concentration 4.08

864 member positions in a slice of 5,760 (15.0% of the slice).

> **Confidence: rows-only** — stable across a row split, but the two interventions name different components. Its held-out concentration is 3.9925, so the circuit still localises; it is the single component NAME that is not settled (§2061).

> **Band-localised, not component-localised.** The two methods disagree by one or two layers inside the `m13`–`m16` band, which is the signature of a circuit spread across adjacent MLPs rather than sitting on one.

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | m16 4.083 | m13 3.981 | m14 3.914 |
| interchange | m15 3.791 | m14 3.713 | m13 3.618 |

At `m16`: mean|dCE| on members **1.2798**, off slice 0.3134, signed dCE on members 0.9068. Second-best component is `m13` at 3.981 — a 1.03x margin.


**Story (from the circuit file):** {'blind_name': '', 'program': [['circ_r_1_1_2']], 'program_bacc': 0.773, 'program_null': 0.512, 'mechanism_level': 'surface'}


**Top members** (context → target, dCE when the circuit is ablated):

- `…\n#### London Highlights\n\n#### History\n\n#### Sights` → `\n`  (dCE 1.74, base CE 24.86)
- `…pm Wed-Sat, noon-5pm Sun;  t` → `High`  (dCE -0.12, base CE 15.07)
- `…; Bartholomew Lane, EC2;  h10am` → `-`  (dCE 0.58, base CE 1.14)


### 32. `r.1.1.1` — m16, concentration 4.08

864 member positions in a slice of 5,760 (15.0% of the slice).

> **Confidence: neither** — the named component is not stable across rows and the two interventions disagree. On a held-out row split the argmax moves `m15` -> `m16`. Its held-out concentration is 4.008, so the circuit still localises; it is the single component NAME that is not settled (§2061).

> **Band-localised, not component-localised.** The two methods disagree by one or two layers inside the `m13`–`m16` band, which is the signature of a circuit spread across adjacent MLPs rather than sitting on one.

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | m16 4.082 | m15 4.081 | m14 4.066 |
| interchange | m14 3.828 | m13 3.612 | m15 3.606 |

At `m16`: mean|dCE| on members **1.2793**, off slice 0.3134, signed dCE on members 0.8831. Second-best component is `m15` at 4.081 — a 1.00x margin.


**Story (from the circuit file):** {'blind_name': '', 'program': [['NOT class_other', 'NOT prev1_mid_word', 'NOT dist_nl_le2'], ['prev1_is_punct', 'NOT prev2_upper_initial', 'NOT prev2_class_sentend']], 'program_bacc': 0.76, 'program_null': 0.404, 'mechanism_level': 'surface'}


**Top members** (context → target, dCE when the circuit is ablated):

- `… B&B\n\n(  MAP` → ` `  (dCE -1.29, base CE 4.38)
- `… outside.\n\nGeorge InnPUB\n\n(  MAP` → ` `  (dCE -1.27, base CE 4.18)
- `…\nEast Lee Guest HouseB&B\n\n(  MAP` → ` `  (dCE -1.4, base CE 4.66)


### 33. `r.2.2` — a8, concentration 4.05

5,760 member positions in a slice of 38,400 (15.0% of the slice).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | a8 4.054 | a6 3.285 | a14 3.056 |
| interchange | a8 3.371 | a6 2.932 | a14 2.8 |

At `a8`: mean|dCE| on members **0.4774**, off slice 0.1177, signed dCE on members 0.0606. Second-best component is `a6` at 3.285 — a 1.23x margin.


**Top members** (context → target, dCE when the circuit is ablated):

- `…london.org.uk; 150 London Wall, EC2` → `;`  (dCE -2.61, base CE 6.02)
- `…ito.co.uk; 32 Exmouth Market, EC1` → `;`  (dCE -2.62, base CE 6.32)
- `…ethouse.org.uk; The Strand, WC2` → `;`  (dCE -2.5, base CE 5.52)


### 34. `r.1.0.3` — m14, concentration 4.00

864 member positions in a slice of 5,760 (15.0% of the slice).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | m14 3.997 | m13 3.925 | m12 3.861 |
| interchange | m14 3.227 | m13 3.143 | m12 2.972 |

At `m14`: mean|dCE| on members **0.6202**, off slice 0.1552, signed dCE on members -0.0221. Second-best component is `m13` at 3.925 — a 1.02x margin.


**Top members** (context → target, dCE when the circuit is ablated):

- `… while the heavenly waft from the cheese room beckons.\n` → `\n`  (dCE -0.63, base CE 12.75)
- `…Hotel UnaBOUTIQUE HOTEL\n\n(` → ` `  (dCE -1.49, base CE 10.28)
- `…'s above Heath St, reached via the Holly Bush Steps.\n` → `\n`  (dCE 0.45, base CE 12.65)


### 35. `r.2.1.0` — a8, concentration 3.99

864 member positions in a slice of 5,760 (15.0% of the slice).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | a8 3.992 | a7 3.694 | a6 3.611 |
| interchange | a8 3.385 | a6 3.196 | a14 3.164 |

At `a8`: mean|dCE| on members **0.594**, off slice 0.1488, signed dCE on members -0.0132. Second-best component is `a7` at 3.694 — a 1.08x margin.


**Story (from the circuit file):** {'blind_name': '', 'program': [['NOT class_other', 'NOT class_subword', 'NOT class_name']], 'program_bacc': 0.782, 'program_null': 0.485, 'mechanism_level': 'surface'}


**Top members** (context → target, dCE when the circuit is ablated):

- `…\n\n4Westminster AbbeyF3\n\n1Sights` → `\n`  (dCE -7.7, base CE 31.91)
- `…\n\n33BarbicanE1\n\n7Shopping` → `\n`  (dCE -7.46, base CE 21.63)
- `… Sir Walter Scott and Jane Austen.\n\n### Farnham` → `\n`  (dCE 10.39, base CE 3.55)


### 36. `r.11.1.1` — a8, concentration 3.98

864 member positions in a slice of 5,760 (15.0% of the slice).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | a8 3.979 | a14 3.534 | a3 3.024 |
| interchange | a8 3.488 | a14 3.096 | a16 2.605 |

At `a8`: mean|dCE| on members **0.5958**, off slice 0.1497, signed dCE on members 0.09. Second-best component is `a14` at 3.534 — a 1.13x margin.


**DAS (rank 1, held-out):** member dCE 0.012, concentration 2.587, recovers 0.017 of the full component; overlap with the closed-form direction 0.0.


**Story (from the circuit file):** {'blind_name': 'no surface program; punctuation-target push is the only surviving behavioral claim', 'program': [['NOT prev1_class_other', 'NOT prev2_class_other', 'NOT dist_nl_le2']], 'program_bacc': 0.588, 'program_null': 0.58, 'mechanism_level': 'none', 'mechanism_line': 'No STRONG single-writer mechanism in a4, a3, or a8 (top ratios 1.093/1.093/1.144 vs thresholds 1.364/1.300/1.300; headroom -0.321/-0.269/-0.225).', 'behavior_line': "Machinery is a symmetric 50/50 push overall (dce_pos -0.709, dce_neg 0.886, minority_share 0.5) but flips at punctuation targets: on the 63/864 punct-target members ablation LOWERS CE 69.8% of the time vs a 48% base rate (margin +21.8pp, p=0.0004, Bonferroni alpha=0.0083/12), i.e. this leaf's machinery actively hurts punctuation prediction, while on the other 801/864 (non-punct) members the push runs the other way (mean dCE +0.116, helps prediction).", 'story_test_class_pairs': {'subword_help': {'ROBUST_V2': False, 'n_available': 140, 'population': {'n': 140, 'hits': 64, 'base_rate_help': 0.48, 'expected_hits': 67.25, 'p_value': 0.7365, 'beats_base_rate': False}, 'seed_pass_frac': 0.0}, 'subword_hurt': {'ROBUST_V2': False, 'n_available': 140, 'population': {'n': 140, 'hits': 76, 'base_rate_help': 0.48, 'expected_hits': 72.75, 'p_value': 0.3216, 'beats_base_rate': False}, 'seed_pass_frac': 0.0}, 'space_word_help': {'ROBUST_V2': False, 'n_available': 629, 'population': {'n': 629, 'hits': 289, 'base_rate_help': 0.48, 'expected_hits': 302.12, 'p_value': 0.8616, 'beats_base_rate': False}, 'seed_pass_frac': 0.0}, 'space_word_hurt': {'ROBUST_V2': False, 'n_available': 629, 'population': {'n': 629, 'hits': 340, 'base_rate_help': 0.48, 'expected_hits': 326.88, 'p_value': 0.1569, 'beats_base_rate': False}, 'seed_pass_frac': 0.0}, 'digit_help': {'ROBUST_V2': False, 'n_available': 29, 'population': {'n': 29, 'hits': 16, 'base_rate_help': 0.48, 'expected_hits': 13.93, 'p_value': 0.2795, 'beats_base_rate': False}, 'seed_pass_frac': 0.2}, 'digit_hurt': {'ROBUST_V2': False, 'n_available': 29, 'population': {'n': 29, 'hits': 13, 'base_rate_help': 0.48, 'expected_hits': 15.07, 'p_value': 0.8303, 'beats_base_rate': False}, 'seed_pass_frac': 0.0}, 'punct_help': {'ROBUST_V2': True, 'n_available': 63, 'population': {'n': 63, 'hits': 44, 'base_rate_help': 0.48, 'expected_hits': 30.26, 'p_value': 0.0004, 'beats_base_rate': True}, 'seed_pass_frac': 0.2}, 'punct_hurt': {'ROBUST_V2': False, 'n_available': 63, 'population': {'n': 63, 'hits': 19, 'base_rate_help': 0.48, 'expected_hits': 32.74, 'p_value': 0.9999, 'beats_base_rate': False}, 'seed_pass_frac': 0.0}, 'capitalized_help': {'ROBUST_V2': False, 'n_available': 154, 'population': {'n': 154, 'hits': 72, 'base_rate_help': 0.48, 'expected_hits': 73.97, 'p_value': 0.6544, 'beats_base_rate': False}, 'seed_pass_frac': 0.0}, 'capitalized_hurt': {'ROBUST_V2': False, 'n_available': 154, 'population': {'n': 154, 'hits': 82, 'base_rate_help': 0.48, 'expected_hits': 80.03, 'p_value': 0.4067, 'beats_base_rate': False}, 'seed_pass_frac': 0.0}, 'newline_help': {'ROBUST_V2': False, 'n_available': 2, 'population': {'n': 2, 'hits': 2, 'base_rate_help': 0.48, 'expected_hits': 0.96, 'p_value': 0.2307, 'beats_base_rate': False}, 'seed_pass_frac': 0.0}, 'newline_hurt': {'ROBUST_V2': False, 'n_available': 2, 'population': {'n': 2, 'hits': 0, 'base_rate_help': 0.48, 'expected_hits': 1.04, 'p_value': 1.0, 'beats_base_rate': False}, 'seed_pass_frac': 0.0}}, 'n_tests': 12}


**Top members** (context → target, dCE when the circuit is ablated):

- `…N/C:N/I:N/A:P` → `)`  (dCE -2.21, base CE 6.8)
- `… a single charge from a standard 110-volt outlet.\nThe` → ` Em`  (dCE -0.7, base CE 4.12)
- `…? | Air Canada | RBC | Samsung Galaxy S4 |` → ` Target`  (dCE 1.6, base CE 7.48)


### 37. `r.6.0.1` — a14, concentration 3.97

864 member positions in a slice of 5,760 (15.0% of the slice).

> **Confidence: methods-only** — both interventions agree, but the argmax moves when the rows are split. On a held-out row split the argmax moves `a16` -> `a14`. Its held-out concentration is 3.9932, so the circuit still localises; it is the single component NAME that is not settled (§2061).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | a14 3.973 | a16 3.95 | m17 3.064 |
| interchange | a14 3.93 | a16 3.663 | a13 2.836 |

At `a14`: mean|dCE| on members **0.4359**, off slice 0.1097, signed dCE on members 0.259. Second-best component is `a16` at 3.95 — a 1.01x margin.


**Top members** (context → target, dCE when the circuit is ablated):

- `… INFORMATION\n\n(www.brighton.co.uk)` → `\n`  (dCE 0.99, base CE 12.12)
- `…3\n\n7London St Pancras YHAC2\n` → `\n`  (dCE -0.3, base CE 10.57)
- `…1Sights\n\nStadeNEIGHBOURHOOD` → `\n`  (dCE 0.69, base CE 12.45)


### 38. `r.1.1.0` — m14, concentration 3.96

864 member positions in a slice of 5,760 (15.0% of the slice).

> **Confidence: methods-only** — both interventions agree, but the argmax moves when the rows are split. On a held-out row split the argmax moves `m15` -> `m14`. Its held-out concentration is 3.8114, so the circuit still localises; it is the single component NAME that is not settled (§2061).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | m14 3.963 | m15 3.897 | m13 3.873 |
| interchange | m14 3.536 | m15 3.434 | m13 3.338 |

At `m14`: mean|dCE| on members **0.6206**, off slice 0.1566, signed dCE on members 0.0673. Second-best component is `m15` at 3.897 — a 1.02x margin.


**Story (from the circuit file):** {'blind_name': '', 'program': [['circ_r_1_1_2', 'NOT dist_nl_ge6']], 'program_bacc': 0.735, 'program_null': 0.551, 'mechanism_level': 'none'}


**Top members** (context → target, dCE when the circuit is ablated):

- `… hourly).\n\n### Hastings\n\nPop 90,300\n` → `\n`  (dCE 2.22, base CE 21.86)
- `… also available.\n\nNorth Central London\n\n1Top S` → `ights`  (dCE -0.68, base CE 3.5)
- `….arthousebandb.com; 24 London Rd; r` → ` £`  (dCE -0.07, base CE 11.06)


### 39. `r.0.3.0` — a3, concentration 3.91

864 member positions in a slice of 5,760 (15.0% of the slice).

> **Confidence: methods-only** — both interventions agree, but the argmax moves when the rows are split. On a held-out row split the argmax moves `m5` -> `a3`. Its held-out concentration is 3.7503, so the circuit still localises; it is the single component NAME that is not settled (§2061).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | a3 3.907 | m5 3.748 | a2 3.495 |
| interchange | a3 3.484 | m5 3.197 | a2 3.097 |

At `a3`: mean|dCE| on members **1.1891**, off slice 0.3044, signed dCE on members -0.0045. Second-best component is `m5` at 3.748 — a 1.04x margin.


**Story (from the circuit file):** {'blind_name': 'Members are rare/proper-noun-like continuation tokens (numbers, name-fragments); machinery helps predict these but wrongly pushes against ordinary common-word completions elsewhere.', 'program': [['NOT class_other', 'NOT is_newline']], 'program_bacc': 0.566, 'program_null': 0.504, 'mechanism_level': 'none', 'redteam_hits': '2/3', 'redteam_verdict': 'HELD', 'flag': 'weak: reviewer-two objection (2026-08-20) -- specific claim never exercised by fresh draw; own program_bacc failed; catch-all branch is base-rate. Verdict-rule v2 would say WEAKEN; recorded by driver.'}


**Top members** (context → target, dCE when the circuit is ablated):

- `… the Battle of the Bulge - Charles B. MacDonald.\n` → `002`  (dCE 1.56, base CE 5.0)
- `…\nrestaurants. International restaurant guides are in development.\n` → `Cu`  (dCE 1.65, base CE 5.11)
- `… score since taking a punt back against Samford in the 2010 season` → ` opener`  (dCE -1.58, base CE 2.08)


### 40. `r.6.1.0` — m5, concentration 3.84

864 member positions in a slice of 5,760 (15.0% of the slice).

> **Confidence: rows-only** — stable across a row split, but the two interventions name different components. Its held-out concentration is 3.8558, so the circuit still localises; it is the single component NAME that is not settled (§2061).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | m5 3.836 | a14 3.447 | m6 3.308 |
| interchange | a14 3.062 | m5 2.985 | a16 2.84 |

At `m5`: mean|dCE| on members **1.1044**, off slice 0.2879, signed dCE on members 0.135. Second-best component is `a14` at 3.447 — a 1.11x margin.


**Top members** (context → target, dCE when the circuit is ablated):

- `…orgeinrye.com; 98 High St; d from` → ` £`  (dCE -0.55, base CE 8.32)
- `…vern.com; 35 Chalk Farm Rd, NW` → `1`  (dCE 1.01, base CE 3.73)
- `….myhotels.com; 17 Jubilee St; r` → ` £`  (dCE -0.45, base CE 11.88)


### 41. `r.1.1` — m13, concentration 3.81

5,760 member positions in a slice of 38,400 (15.0% of the slice).

> **Confidence: rows-only** — stable across a row split, but the two interventions name different components. Its held-out concentration is 3.7817, so the circuit still localises; it is the single component NAME that is not settled (§2061).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | m13 3.815 | m14 3.734 | m15 3.691 |
| interchange | m14 3.357 | m13 3.298 | m15 3.261 |

At `m13`: mean|dCE| on members **0.5404**, off slice 0.1416, signed dCE on members 0.1225. Second-best component is `m14` at 3.734 — a 1.02x margin.


**Story (from the circuit file):** {'blind_name': '', 'program': [['class_newline'], ['NOT class_other', 'NOT class_subword', 'NOT class_digit']], 'program_bacc': 0.73, 'program_null': 0.704, 'mechanism_level': 'none'}


**Top members** (context → target, dCE when the circuit is ablated):

- `…\n\nAWallett's Court\n\nAReading Rooms` → `\n`  (dCE 1.98, base CE 28.53)
- `…\n\n### Best Places to Stay\n\nABleak House` → `\n`  (dCE 2.6, base CE 30.09)
- `…ights\n\n1Westminster Hall\n\n2Jewel Tower` → `\n`  (dCE 3.46, base CE 24.17)


### 42. `r.1.3` — m13, concentration 3.77

5,760 member positions in a slice of 38,400 (15.0% of the slice).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | m13 3.775 | m14 3.647 | m15 3.631 |
| interchange | m13 3.298 | m14 3.295 | m15 3.22 |

At `m13`: mean|dCE| on members **0.5347**, off slice 0.1416, signed dCE on members 0.1132. Second-best component is `m14` at 3.647 — a 1.04x margin.


**Top members** (context → target, dCE when the circuit is ablated):

- `… hmuseum 9.30am-5pm;  t` → `F`  (dCE 0.52, base CE 9.44)
- `…-Sat, noon-10.30pm Sun;  t` → `St`  (dCE 0.59, base CE 10.98)
- `… Dover\n\n#### Rye\n\n#### Battle\n\n#### Hastings` → `\n`  (dCE 0.45, base CE 33.14)


### 43. `r.6.0.3` — m17, concentration 3.65

864 member positions in a slice of 5,760 (15.0% of the slice).

> **Confidence: rows-only** — stable across a row split, but the two interventions name different components. Its held-out concentration is 3.5469, so the circuit still localises; it is the single component NAME that is not settled (§2061).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | m17 3.655 | m16 3.182 | a16 2.596 |
| interchange | m16 3.101 | m17 2.968 | a16 2.646 |

At `m17`: mean|dCE| on members **2.5202**, off slice 0.6896, signed dCE on members 2.0933. Second-best component is `m16` at 3.182 — a 1.15x margin.


**Top members** (context → target, dCE when the circuit is ablated):

- `…\n#### Broadstairs\n\n#### Ramsgate\n\n#### Sandwich` → `\n`  (dCE 2.13, base CE 22.84)
- `…6 7890; www.uclh.org; 235 E` → `ust`  (dCE 2.62, base CE 0.48)
- `… 39a Canonbury Sq, N1; adult/child` → ` £`  (dCE 2.72, base CE 4.28)


### 44. `r.6.2.2` — m16, concentration 3.63

864 member positions in a slice of 5,760 (15.0% of the slice).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | m16 3.632 | a17 3.081 | m15 2.82 |
| interchange | m16 3.258 | a17 3.105 | m15 2.953 |

At `m16`: mean|dCE| on members **1.158**, off slice 0.3188, signed dCE on members 0.8188. Second-best component is `a17` at 3.081 — a 1.18x margin.


**Story (from the circuit file):** {'blind_name': '', 'program': [['circ_r_1_1_2', 'NOT class_other', 'NOT is_punct']], 'program_bacc': 0.77, 'program_null': 0.662, 'mechanism_level': 'none'}


**Top members** (context → target, dCE when the circuit is ablated):

- `… has well-appointed rooms to let.\n\n7Shopping` → `\n`  (dCE -2.07, base CE 21.84)
- `…bury River Navigation CompanyB2\n\n4Sleeping\n` → `\n`  (dCE -4.23, base CE 26.74)
- `…\n\n2Marble ArchA5\n\n5Eating` → `\n`  (dCE -0.82, base CE 16.7)


### 45. `r.3.1.0` — a2, concentration 3.63

864 member positions in a slice of 5,760 (15.0% of the slice).

> **Confidence: neither** — the named component is not stable across rows and the two interventions disagree. On a held-out row split the argmax moves `a14` -> `a2`. Its held-out concentration is 2.7737, so the circuit still localises; it is the single component NAME that is not settled (§2061).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | a2 3.629 | m5 3.557 | a17 3.512 |
| interchange | a16 3.225 | a17 3.206 | a3 3.175 |

At `a2`: mean|dCE| on members **1.3427**, off slice 0.37, signed dCE on members 0.1808. Second-best component is `m5` at 3.557 — a 1.02x margin.


**Story (from the circuit file):** {'blind_name': '', 'program': [['NOT class_other', 'NOT class_subword', 'NOT class_digit']], 'program_bacc': 0.862, 'program_null': 0.327, 'mechanism_level': 'surface'}


**Top members** (context → target, dCE when the circuit is ablated):

- `…\n#### Broadstairs\n\n#### Ramsgate\n\n#### Sandwich` → `\n`  (dCE -10.78, base CE 22.84)
- `…'s GlobeD1\n\n4Tate ModernD1\n` → `\n`  (dCE -4.08, base CE 14.76)
- `…\n\nTo/From the Airports\n\nGatwick` → `\n`  (dCE -2.39, base CE 22.03)


### 46. `r.1.0.1` — m14, concentration 3.60

864 member positions in a slice of 5,760 (15.0% of the slice).

> **Confidence: neither** — the named component is not stable across rows and the two interventions disagree. On a held-out row split the argmax moves `m12` -> `m14`. Its held-out concentration is 3.3564, so the circuit still localises; it is the single component NAME that is not settled (§2061).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | m14 3.601 | m13 3.55 | m12 3.449 |
| interchange | a16 2.969 | m14 2.931 | a14 2.741 |

At `m14`: mean|dCE| on members **0.5588**, off slice 0.1552, signed dCE on members -0.0252. Second-best component is `m13` at 3.55 — a 1.01x margin.


**Story (from the circuit file):** {'blind_name': '', 'program': [['prev1_is_punct', 'NOT prev2_class_sentend', 'NOT prev2_class_comma'], ['prev2_class_ind', 'NOT prev2_starts_space', 'NOT class_digit']], 'program_bacc': 0.626, 'program_null': 0.429, 'mechanism_level': 'none'}


**Top members** (context → target, dCE when the circuit is ablated):

- `….uk;  h5am-dusk;  t` → `Reg`  (dCE -1.58, base CE 13.93)
- `… up in the 15th century; the spire dates from the` → ` 19`  (dCE -1.07, base CE 4.28)
- `…4pm Mon-Fri; ` → ` t`  (dCE 2.11, base CE 13.49)


### 47. `r.6.2.1` — a9, concentration 3.59

864 member positions in a slice of 5,760 (15.0% of the slice).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | a9 3.592 | a11 2.695 | m9 2.654 |
| interchange | a9 2.88 | a7 2.449 | a16 2.449 |

At `a9`: mean|dCE| on members **0.8202**, off slice 0.2283, signed dCE on members -0.1922. Second-best component is `a11` at 2.695 — a 1.33x margin.


**Top members** (context → target, dCE when the circuit is ablated):

- `…; www.barbican.org.uk; Silk St` → `,`  (dCE 0.1, base CE 5.6)
- `…rundel and the coast.\n\n1Sights` → `\n`  (dCE -5.28, base CE 15.32)
- `… www.wearebigchill.com; 257-259` → ` Pent`  (dCE 1.01, base CE 11.23)


### 48. `r.2.2.0` — a8, concentration 3.55

864 member positions in a slice of 5,760 (15.0% of the slice).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | a8 3.553 | a14 3.106 | a7 2.992 |
| interchange | a8 3.099 | a14 2.793 | a7 2.677 |

At `a8`: mean|dCE| on members **0.5293**, off slice 0.149, signed dCE on members 0.0098. Second-best component is `a14` at 3.106 — a 1.14x margin.


**Top members** (context → target, dCE when the circuit is ablated):

- `…Eating\n\n16BohoB2\n\n17D` → `ees`  (dCE 1.77, base CE 9.74)
- `…nline; combined tickets with the London Eye, London Dungeon and London` → ` Seal`  (dCE -1.06, base CE 15.58)
- `… Lane, EC4; dm £17-25, d` → ` £`  (dCE -1.23, base CE 5.26)


### 49. `r.6.1.1` — a16, concentration 3.53

864 member positions in a slice of 5,760 (15.0% of the slice).

> **Confidence: methods-only** — both interventions agree, but the argmax moves when the rows are split. On a held-out row split the argmax moves `a8` -> `a16`. Its held-out concentration is 2.9672, so the circuit still localises; it is the single component NAME that is not settled (§2061).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | a16 3.531 | a14 3.387 | m5 3.361 |
| interchange | a16 3.063 | a14 3.056 | a8 3.048 |

At `a16`: mean|dCE| on members **0.2636**, off slice 0.0747, signed dCE on members 0.1349. Second-best component is `a14` at 3.387 — a 1.04x margin.


**Story (from the circuit file):** {'blind_name': "Domain/URL-listing separator token; helps complete website suffixes and prices after 'www.x;' but wrongly raises loss on other post-';' continuations like newlines.", 'program': [['NOT starts_space', 'NOT class_name']], 'program_bacc': 0.606, 'program_null': 0.454, 'mechanism_level': 'none', 'redteam': [{'gi': 36595, 'context': ' Getting hopelessly lost in the crooked medieval streets of Sandwich\n\n', 'target': '2', 'dce': 1.72, 'verdict': 'MISS', 'why': 'narrative town-guide text (not a URL/address-separator context); story does not predict this member at all'}, {'gi': 52954, 'context': '-5pm Mon-Sat, to 4pm Sun)\n', 'target': '\n', 'dce': -1.23, 'verdict': 'HIT', 'why': "listing-style hours text ending in newline, negative dce matches story's wrong-push-on-newlines direction"}, {'gi': 8581, 'context': 'designmuseum.org; 28 Shad Thames, SE1;', 'target': ' adult', 'dce': -0.89, 'verdict': 'HIT', 'why': "exact www-domain + ';' listing separator predicting ' adult', negative dce matches story"}]}


**Top members** (context → target, dCE when the circuit is ablated):

- `…  GOOGLE MAP ) ; www.nhm.` → `ac`  (dCE 3.03, base CE 1.74)
- `…riverbus.co.uk; adult/child 30min trip` → ` £`  (dCE 2.83, base CE 3.16)
- `… www.wearebigchill.com; 257-259` → ` Pent`  (dCE 2.02, base CE 11.23)


### 50. `r.8.1.0` — a3, concentration 3.50

864 member positions in a slice of 5,760 (15.0% of the slice).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | a3 3.499 | a4 3.328 | a5 3.292 |
| interchange | a3 3.13 | a8 2.936 | a5 2.881 |

At `a3`: mean|dCE| on members **1.0746**, off slice 0.3072, signed dCE on members -0.0637. Second-best component is `a4` at 3.328 — a 1.05x margin.


**Story (from the circuit file):** {'blind_name': None, 'program': [['circ_r_0_0_1', 'NOT is_newline']], 'program_bacc': 0.574, 'program_null': 0.498, 'mechanism_level': 'none', 'mechanism': "No writer/courier identified: input-writer decomposition into a4/a9/a7 (this leaf's machinery) is not enriched vs off-slice under the 5-seed bootstrap (min ratios 0.996-1.216, below the 1.3 ENRICHED bar; ENRICHED_STABLE=False for all three components). Surface program (NOT is_newline) also fails doc-disjoint heldout (bacc 0.574 vs null 0.498). No behavioral claim survives base-rate testing across 12 class/direction pairs (Bonferroni alpha 0.0083; best punct/helps population p=0.0455)."}


**Top members** (context → target, dCE when the circuit is ablated):

- `… and her Paramount swan song, "True Confession" --` → ` was`  (dCE -2.82, base CE 9.94)
- `… the Battle of the Bulge - Charles B. MacDonald.\n` → `002`  (dCE 3.81, base CE 5.0)
- `… contests put on by ActiveRain and its members. Everyone can join` → ` the`  (dCE 1.88, base CE 4.22)


### 51. `r.5.0.1` — a16, concentration 3.40

864 member positions in a slice of 5,760 (15.0% of the slice).

> **Confidence: methods-only** — both interventions agree, but the argmax moves when the rows are split. On a held-out row split the argmax moves `a3` -> `a16`. Its held-out concentration is 3.4141, so the circuit still localises; it is the single component NAME that is not settled (§2061).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | a16 3.398 | a3 3.373 | a2 3.001 |
| interchange | a16 3.034 | a14 2.762 | a3 2.727 |

At `a16`: mean|dCE| on members **0.2494**, off slice 0.0734, signed dCE on members 0.1283. Second-best component is `a3` at 3.373 — a 1.01x margin.


**Story (from the circuit file):** {'program': [['circ_r_0_0_1']], 'program_bacc': 0.556, 'program_null': 0.442, 'mechanism_level': 'none', 'behavior_line': 'no behavioral claim survives base-rate testing', 'story_test_by_kind': {'subword': {'n': 5, 'hits': 4, 'base_rate_help': 0.447, 'expected_hits': 2.23, 'p_value': 0.128, 'beats_base_rate': False}, 'newline': {'n': 5, 'hits': 4, 'base_rate_help': 0.447, 'expected_hits': 2.23, 'p_value': 0.128, 'beats_base_rate': False}, 'space_word': {'n': 5, 'hits': 2, 'base_rate_help': 0.447, 'expected_hits': 2.23, 'p_value': 0.7389, 'beats_base_rate': False}, 'punct': {'n': 5, 'hits': 2, 'base_rate_help': 0.447, 'expected_hits': 2.23, 'p_value': 0.7389, 'beats_base_rate': False}, 'digit': {'n': 5, 'hits': 3, 'base_rate_help': 0.447, 'expected_hits': 2.23, 'p_value': 0.4009, 'beats_base_rate': False}, 'capitalized': {'n': 5, 'hits': 2, 'base_rate_help': 0.447, 'expected_hits': 2.23, 'p_value': 0.7389, 'beats_base_rate': False}}}


**Top members** (context → target, dCE when the circuit is ablated):

- `…efoodlikeItalianfood *inhale* Idon't` → `even`  (dCE -1.69, base CE 6.98)
- `… nave, chancel, south porch,\nand south is` → `le`  (dCE 5.41, base CE 1.79)
- `…ant lilies that dominate bouquets and their little bell shapes` → ` could`  (dCE -1.34, base CE 11.47)


### 52. `r.23.2.1` — a8, concentration 3.38

864 member positions in a slice of 5,760 (15.0% of the slice).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | a8 3.385 | a2 3.026 | a3 2.894 |
| interchange | a8 2.908 | a3 2.542 | a6 2.516 |

At `a8`: mean|dCE| on members **0.5044**, off slice 0.149, signed dCE on members 0.0919. Second-best component is `a2` at 3.026 — a 1.12x margin.


**DAS (rank 1, held-out):** member dCE 0.0215, concentration 3.075, recovers 0.036 of the full component; overlap with the closed-form direction 0.029.


**Story (from the circuit file):** {'blind_name': 'generic-vs-proper-noun completion push', 'text': 'High-loss continuations after named/branded contexts; machinery favors generic common-word completions, suppresses specific proper-noun/brand completions (except some place-name suffixes, e.g. Hungaroring).', 'program': [['NOT prev1_seen_before', 'NOT is_punct'], ['circ_r_0_0_1', 'NOT dist_nl_le2', 'NOT prev1_class_other']], 'program_bacc': 0.517, 'program_null': 0.541, 'mechanism_level': 'none', 'red_team': {'hits': 3, 'total': 3, 'detail': [{'gi': 764, 'target': ' miserable', 'dce': 0.94, 'predicted': 'helps (generic common adjective)', 'actual': 'helps', 'hit': True}, {'gi': 154700, 'target': 'ted', 'dce': -0.54, 'predicted': 'hurts (brand/proper-noun continuation of "Target")', 'actual': 'hurts', 'hit': True, 'note': 'ambiguous: completed word "Targeted" is itself generic; scored as hurts because context token is the brand name Target'}, {'gi': 240197, 'target': ' Ars', 'dce': -1.07, 'predicted': 'hurts (specific proper noun, Arsenal)', 'actual': 'hurts', 'hit': True}], 'caveat': 'top example gi=198076 ("Hungar"->"oring", dce=+2.51) contradicts the suppress-proper-noun claim: machinery HELPS complete a place-name compound there. Story is directional/majority, not exceptionless.'}}


**Top members** (context → target, dCE when the circuit is ablated):

- `…so\n|First race||Hungarian Grand Prix||Hungar` → `oring`  (dCE 2.51, base CE 4.13)
- `…� debate highlighted at this year’s PSA Conference by` → ` Matthew`  (dCE -1.78, base CE 11.18)
- `… indigenous black Christians and animists, 100,000 in Darfur` → ` alone`  (dCE 2.45, base CE 5.56)


### 53. `r.5.3.1` — a15, concentration 3.33

864 member positions in a slice of 5,760 (15.0% of the slice).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | a15 3.33 | a16 3.093 | a2 2.88 |
| interchange | a15 2.947 | a16 2.772 | a14 2.474 |

At `a15`: mean|dCE| on members **0.222**, off slice 0.0667, signed dCE on members 0.048. Second-best component is `a16` at 3.093 — a 1.08x margin.


**Story (from the circuit file):** {'blind_name': 'no behavioral claim survives base-rate testing', 'program': [['circ_r_0_0_1', 'dist_nl_ge6']], 'program_bacc': 0.583, 'program_null': 0.488, 'mechanism_level': 'none', 'mechanism': {'note': 'no writer enriches machinery input across bootstrap draws (a2 top a0 mean 1.005 [0.827-1.246]; a4 top a2 mean 1.048 [0.985-1.082]); ENRICHED_STABLE=False both'}}


**Top members** (context → target, dCE when the circuit is ablated):

- `… and its members. Everyone can join the\ngroup and help encourage` → ` each`  (dCE -1.74, base CE 2.4)
- `… who would enjoy the sight of pickle and spider-web or` → `nam`  (dCE -1.97, base CE 2.23)
- `…\nIn a different twist, Bob Fink, an independent music` → `ologist`  (dCE 1.84, base CE 1.37)


### 54. `r.1.3.0` — m13, concentration 3.23

864 member positions in a slice of 5,760 (15.0% of the slice).

> **Confidence: rows-only** — stable across a row split, but the two interventions name different components. Its held-out concentration is 3.2216, so the circuit still localises; it is the single component NAME that is not settled (§2061).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | m13 3.228 | m12 3.151 | m15 3.009 |
| interchange | a16 2.703 | a17 2.682 | a14 2.678 |

At `m13`: mean|dCE| on members **0.5668**, off slice 0.1756, signed dCE on members 0.0048. Second-best component is `m12` at 3.151 — a 1.02x margin.


**Story (from the circuit file):** {'blind_name': '', 'program': [['NOT class_other', 'NOT prev2_upper_initial', 'NOT class_digit']], 'program_bacc': 0.689, 'program_null': 0.49, 'mechanism_level': 'none'}


**Top members** (context → target, dCE when the circuit is ablated):

- `… restaurant. Book well ahead.\n\nJubilee HouseB` → `&`  (dCE 1.65, base CE 8.21)
- `… Severini.\n\nRoyal London Walking Tour\n\n1S` → `ights`  (dCE 0.84, base CE 9.37)
- `… hostel-like vibe. Only one of the three rooms is` → ` en`  (dCE 0.86, base CE 5.43)


### 55. `r.6.0.2` — a9, concentration 3.19

864 member positions in a slice of 5,760 (15.0% of the slice).

> **Confidence: rows-only** — stable across a row split, but the two interventions name different components. Its held-out concentration is 3.254, so the circuit still localises; it is the single component NAME that is not settled (§2061).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | a9 3.185 | a16 2.988 | a14 2.896 |
| interchange | a16 2.892 | a14 2.74 | a9 2.73 |

At `a9`: mean|dCE| on members **0.7357**, off slice 0.231, signed dCE on members -0.1393. Second-best component is `a16` at 2.988 — a 1.07x margin.


**Story (from the circuit file):** {'blind_name': '', 'program': [['NOT prev2_starts_space', 'NOT class_other', 'NOT class_digit'], ['NOT prev1_starts_space', 'NOT prev2_class_upper_pair', 'NOT prev2_class_comma']], 'program_bacc': 0.624, 'program_null': 0.4, 'mechanism_level': 'none'}


**Top members** (context → target, dCE when the circuit is ablated):

- `… John's Wood Rd, NW8; tours adult/child` → ` £`  (dCE 2.34, base CE 2.88)
- `… required for the Monday- and Wednesday-evening candlelit sessions` → ` (£`  (dCE 1.6, base CE 7.64)
- `…\n#### Broadstairs\n\n#### Ramsgate\n\n#### Sandwich` → `\n`  (dCE 2.16, base CE 22.84)


### 56. `r.6.2.0` — a16, concentration 3.17

864 member positions in a slice of 5,760 (15.0% of the slice).

> **Confidence: methods-only** — both interventions agree, but the argmax moves when the rows are split. On a held-out row split the argmax moves `m16` -> `a16`. Its held-out concentration is 2.9067, so the circuit still localises; it is the single component NAME that is not settled (§2061).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | a16 3.167 | m16 2.955 | m17 2.744 |
| interchange | a16 2.965 | m16 2.836 | a14 2.802 |

At `a16`: mean|dCE| on members **0.2375**, off slice 0.075, signed dCE on members 0.111. Second-best component is `m16` at 2.955 — a 1.07x margin.


**Top members** (context → target, dCE when the circuit is ablated):

- `…\n\n33BarbicanE1\n\n7Shopping` → `\n`  (dCE 0.17, base CE 21.63)
- `…Ye Olde Cheshire CheeseB3\n\n3Entertainment` → `\n`  (dCE 0.11, base CE 23.3)
- `…3\n\n34RevengeF4\n\n3Entertainment` → `\n`  (dCE 0.72, base CE 29.94)


### 57. `r.6.3.1` — a16, concentration 3.13

864 member positions in a slice of 5,760 (15.0% of the slice).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | a16 3.133 | a9 3.032 | a14 2.95 |
| interchange | a16 2.802 | a14 2.699 | a9 2.612 |

At `a16`: mean|dCE| on members **0.2324**, off slice 0.0742, signed dCE on members 0.1049. Second-best component is `a9` at 3.032 — a 1.03x margin.


**Top members** (context → target, dCE when the circuit is ablated):

- `… alternative music most nights in this pub's back room (cover usually` → ` £`  (dCE 1.82, base CE 4.65)
- `…\n\n33BarbicanE1\n\n7Shopping` → `\n`  (dCE -1.92, base CE 21.63)
- `…re.co.uk; Belvedere Rd, SE1` → `;`  (dCE -2.36, base CE 6.2)


### 58. `r.18.2.0` — a7, concentration 3.10

864 member positions in a slice of 5,760 (15.0% of the slice).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | a7 3.104 | a9 2.803 | a16 2.802 |
| interchange | a7 2.763 | a16 2.549 | a9 2.488 |

At `a7`: mean|dCE| on members **0.7248**, off slice 0.2335, signed dCE on members -0.0276. Second-best component is `a9` at 2.803 — a 1.11x margin.


**Story (from the circuit file):** {'blind_name': 'no behavioral claim survives base-rate testing except a narrow punctuation-target effect', 'program': [['starts_space', 'prev1_starts_space']], 'program_bacc': 0.555, 'program_null': 0.498, 'mechanism_level': 'none', 'behavior_line': "On punctuation-target positions the leaf's machinery HURTS prediction (ablating it lowers CE 5/5 draws, story_test p=0.041, beats base rate 0.528); but overall member sign-split is near-even (minority_share=0.473) and no other surface class (subword/space_word/digit/capitalized/newline) showed an enriched push in either direction -- the punct effect does not generalize to a whole-leaf story.", 'story_test_punct_help': {'n': 5, 'hits': 5, 'base_rate_help': 0.528, 'expected_hits': 2.64, 'p_value': 0.041, 'beats_base_rate': True}, 'story_test_all_kinds': {'subword': {'gis': [165065, 19833, 48100, 136615, 235398], 'help': {'n': 5, 'hits': 2, 'base_rate_help': 0.528, 'expected_hits': 2.64, 'p_value': 0.8453, 'beats_base_rate': False}, 'hurt': {'n': 5, 'hits': 3, 'base_rate_help': 0.528, 'expected_hits': 2.36, 'p_value': 0.448, 'beats_base_rate': False}}, 'space_word': {'gis': [105887, 32533, 122749, 253437, 216122], 'help': {'n': 5, 'hits': 2, 'base_rate_help': 0.528, 'expected_hits': 2.64, 'p_value': 0.8453, 'beats_base_rate': False}, 'hurt': {'n': 5, 'hits': 3, 'base_rate_help': 0.528, 'expected_hits': 2.36, 'p_value': 0.448, 'beats_base_rate': False}}, 'digit': {'gis': [39325, 208117, 123808, 21741, 255089], 'help': {'n': 5, 'hits': 1, 'base_rate_help': 0.528, 'expected_hits': 2.64, 'p_value': 0.9765, 'beats_base_rate': False}, 'hurt': {'n': 5, 'hits': 4, 'base_rate_help': 0.528, 'expected_hits': 2.36, 'p_value': 0.1547, 'beats_base_rate': False}}, 'punct': {'gis': [182573, 49010, 171307, 181938, 204857], 'help': {'n': 5, 'hits': 5, 'base_rate_help': 0.528, 'expected_hits': 2.64, 'p_value': 0.041, 'beats_base_rate': True}, 'hurt': {'n': 5, 'hits': 0, 'base_rate_help': 0.528, 'expected_hits': 2.36, 'p_value': 1.0, 'beats_base_rate': False}}, 'capitalized': {'gis': [251284, 255203, 232254, 201962, 204239], 'help': {'n': 5, 'hits': 3, 'base_rate_help': 0.528, 'expected_hits': 2.64, 'p_value': 0.552, 'beats_base_rate': False}, 'hurt': {'n': 5, 'hits': 2, 'base_rate_help': 0.528, 'expected_hits': 2.36, 'p_value': 0.7759, 'beats_base_rate': False}}, 'newline': {'gis': [55048, 255810, 17138, 216712, 140365], 'help': {'n': 5, 'hits': 3, 'base_rate_help': 0.528, 'expected_hits': 2.64, 'p_value': 0.552, 'beats_base_rate': False}, 'hurt': {'n': 5, 'hits': 2, 'base_rate_help': 0.528, 'expected_hits': 2.36, 'p_value': 0.7759, 'beats_base_rate': False}}}}


**Top members** (context → target, dCE when the circuit is ablated):

- `… he knows that I want him to be in a position to be` → ` attacking`  (dCE -1.35, base CE 11.81)
- `… it but more lengthening. My main gripe with it back` → ` there`  (dCE -1.58, base CE 7.07)
- `…magical thinking to further her showbiz career -- starting with visual` → `izing`  (dCE -1.62, base CE 4.68)


### 59. `r.13.2.1` — a3, concentration 3.04

864 member positions in a slice of 5,760 (15.0% of the slice).

> **Confidence: neither** — the named component is not stable across rows and the two interventions disagree. On a held-out row split the argmax moves `a3` -> `a7`. Its held-out concentration is 3.0512, so the circuit still localises; it is the single component NAME that is not settled (§2061).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | a3 3.042 | a7 3.009 | a6 2.844 |
| interchange | a7 2.771 | a14 2.631 | a6 2.607 |

At `a3`: mean|dCE| on members **0.9373**, off slice 0.3081, signed dCE on members 0.0746. Second-best component is `a7` at 3.009 — a 1.01x margin.


**Story (from the circuit file):** {'blind_name': 'punctuation-target CE reducer (two-signed, ~50/50 elsewhere)', 'program': [['NOT class_other', 'NOT circ_r_3_1_0']], 'program_bacc': 0.537, 'program_null': 0.484, 'mechanism_level': 'none', 'mechanism_note': 'a7/a6/a3 input writer composition NOT enriched for members: top writer a2 ratio ~1.03-1.05 (bootstrap mean, all 3 components), ENRICHED_STABLE=False everywhere (5-draw bootstrap).', 'behavior_note': 'Machinery pushes CE down (helps) specifically at punctuation targets (39/49=80%, p=0.0, ROBUST_V2 true, vs 51% base rate); elsewhere (space/digit/capitalized/subword/newline) no direction is robust -- push is ~50/50 base rate.', 'behavior_sweep_12pair': {'subword_True': {'n_available': 102, 'population_p': 0.1407, 'population_hits': 58, 'population_n': 102, 'seed_pass_frac': 0.2, 'ROBUST_V2': False}, 'subword_False': {'n_available': 102, 'population_p': 0.8991, 'population_hits': 44, 'population_n': 102, 'seed_pass_frac': 0.0, 'ROBUST_V2': False}, 'space_word_True': {'n_available': 680, 'population_p': 0.9428, 'population_hits': 327, 'population_n': 680, 'seed_pass_frac': 0.2, 'ROBUST_V2': False}, 'space_word_False': {'n_available': 680, 'population_p': 0.0665, 'population_hits': 353, 'population_n': 680, 'seed_pass_frac': 0.0, 'ROBUST_V2': False}, 'digit_True': {'n_available': 24, 'population_p': 0.7624, 'population_hits': 11, 'population_n': 24, 'seed_pass_frac': 0.0, 'ROBUST_V2': False}, 'digit_False': {'n_available': 24, 'population_p': 0.3796, 'population_hits': 13, 'population_n': 24, 'seed_pass_frac': 0.0, 'ROBUST_V2': False}, 'punct_True': {'n_available': 49, 'population_p': 0.0, 'population_hits': 39, 'population_n': 49, 'seed_pass_frac': 0.6, 'ROBUST_V2': True}, 'punct_False': {'n_available': 49, 'population_p': 1.0, 'population_hits': 10, 'population_n': 49, 'seed_pass_frac': 0.0, 'ROBUST_V2': False}, 'capitalized_True': {'n_available': 165, 'population_p': 0.3614, 'population_hits': 87, 'population_n': 165, 'seed_pass_frac': 0.0, 'ROBUST_V2': False}, 'capitalized_False': {'n_available': 165, 'population_p': 0.6951, 'population_hits': 78, 'population_n': 165, 'seed_pass_frac': 0.0, 'ROBUST_V2': False}, 'newline_True': {'n_available': 9, 'population_p': 0.2748, 'population_hits': 6, 'population_n': 9, 'seed_pass_frac': 0.0, 'ROBUST_V2': False}, 'newline_False': {'n_available': 9, 'population_p': 0.8995, 'population_hits': 3, 'population_n': 9, 'seed_pass_frac': 0.0, 'ROBUST_V2': False}}}


**Top members** (context → target, dCE when the circuit is ablated):

- `… skills, ultimately that is what happened, I ran out of fucking` → ` coping`  (dCE -1.59, base CE 4.27)
- `… for the beginner it is a great way to start understanding Archi` → `M`  (dCE -1.8, base CE 4.64)
- `…_________________- woo! so pretty :)\ngivemesomethinganything asked` → ` fuck`  (dCE 1.31, base CE 8.37)


### 60. `r.7.1.1` — a7, concentration 2.98

864 member positions in a slice of 5,760 (15.0% of the slice).

> **Confidence: methods-only** — both interventions agree, but the argmax moves when the rows are split. On a held-out row split the argmax moves `a7` -> `a16`. Its held-out concentration is 2.9765, so the circuit still localises; it is the single component NAME that is not settled (§2061).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | a7 2.977 | a16 2.785 | a11 2.683 |
| interchange | a7 2.679 | a16 2.535 | a14 2.446 |

At `a7`: mean|dCE| on members **0.6911**, off slice 0.2322, signed dCE on members -0.077. Second-best component is `a16` at 2.785 — a 1.07x margin.


**Story (from the circuit file):** {'blind_name': 'no STRONG single-writer mechanism into a7; weak capitalized-target CE reduction', 'program': [['prev1_starts_space', 'NOT is_punct']], 'program_bacc': 0.54, 'program_null': 0.532, 'mechanism_level': 'none', 'mechanism': {'note': 'No STRONG single-writer mechanism into a7 (top writer m1, bootstrap ratio 1.078 [1.048-1.156], threshold_v2 1.433 = max(1.3, null mean 1.17 + 2sd 0.131), headroom -0.385); downstream top consumer a17 ratio 1.191 also below the 1.3 bar (population gates (a)/(b) FAILED; random-subspace consumer 1.086).'}, 'behavior': "At capitalized-initial targets (137/864 members, 15.9%), ablation raises CE (leaf helps prediction) 90/137 times vs 72.3 expected by the leaf's own base rate (p=0.0015, ROBUST_V2, n_tests=12); no other tested class (subword, space_word, digit, punct, newline; both directions, 12 pairs total) clears the corrected bar -- the leaf's near-even two-signed split (minority_share=0.493) is NOT explained by any class outside capitalized targets.", 'behavior_test': {'kind': 'capitalized', 'pred_help': False, 'n_available': 137, 'per_seed': [{'seed': 1, 'hits': 4, 'n': 5, 'p_value': 0.2241}, {'seed': 2, 'hits': 4, 'n': 5, 'p_value': 0.2241}, {'seed': 3, 'hits': 4, 'n': 5, 'p_value': 0.2241}, {'seed': 4, 'hits': 4, 'n': 5, 'p_value': 0.2241}, {'seed': 11, 'hits': 3, 'n': 5, 'p_value': 0.552}], 'seed_pass_frac': 0.0, 'population': {'n': 137, 'hits': 90, 'base_rate_help': 0.472, 'expected_hits': 72.31, 'p_value': 0.0015, 'beats_base_rate': True}, 'ROBUST': False, 'n_tests': 12, 'alpha': 0.0083, 'ROBUST_V2': True, 'gate_note': 'use ROBUST_V2; ROBUST v1 is underpowered'}}


**Top members** (context → target, dCE when the circuit is ablated):

- `… 19, 1924\nTy Cobb can tell you. Personally causing your` → ` team`  (dCE 2.36, base CE 4.88)
- `…\nIn response to Blind faith won’t make climate science` → ` gaps`  (dCE -1.79, base CE 15.02)
- `…\n[–]Subhazard 21 Punkte22 Punkte23` → ` Punk`  (dCE 2.06, base CE 3.94)


### 61. `r.6.0.0` — a16, concentration 2.88

864 member positions in a slice of 5,760 (15.0% of the slice).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | a16 2.884 | a14 2.508 | m16 2.303 |
| interchange | a16 2.633 | a14 2.458 | m16 2.289 |

At `a16`: mean|dCE| on members **0.2143**, off slice 0.0743, signed dCE on members 0.0617. Second-best component is `a14` at 2.508 — a 1.15x margin.


**Story (from the circuit file):** {'blind_name': '', 'program': [['prev1_class_digit']], 'program_bacc': 0.603, 'program_null': 0.502, 'mechanism_level': 'none'}


**Top members** (context → target, dCE when the circuit is ablated):

- `…vern.com; 35 Chalk Farm Rd, NW1` → `;`  (dCE -2.53, base CE 4.48)
- `…  GOOGLE MAP ) ; www.nhm.` → `ac`  (dCE 2.82, base CE 1.74)
- `…gatecemetery.org; Swain's Lane, N` → `6`  (dCE 1.94, base CE 5.35)


### 62. `r.6.2.3` — a9, concentration 2.61

864 member positions in a slice of 5,760 (15.0% of the slice).

> **Confidence: neither** — the named component is not stable across rows and the two interventions disagree. On a held-out row split the argmax moves `a17` -> `a9`. Its held-out concentration is 2.63, so the circuit still localises; it is the single component NAME that is not settled (§2061).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | a9 2.611 | a17 2.61 | a16 2.533 |
| interchange | a17 2.661 | a16 2.485 | a13 2.441 |

At `a9`: mean|dCE| on members **0.5961**, off slice 0.2283, signed dCE on members -0.0643. Second-best component is `a17` at 2.61 — a 1.00x margin.


**Top members** (context → target, dCE when the circuit is ablated):

- `…\n#### Broadstairs\n\n#### Ramsgate\n\n#### Sandwich` → `\n`  (dCE 5.04, base CE 22.84)
- `…\n\n#### Woburn\n\n#### Waddesdon` → `\n`  (dCE 1.23, base CE 16.09)
- `…\n#### Canterbury\n\n#### Leeds Castle\n\n#### Margate` → `\n`  (dCE -0.5, base CE 23.87)
