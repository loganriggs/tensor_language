# Computation-path registry

This index tracks algebraically folded native paths separately from causal circuit
records. A path can cross module boundaries or retain only selected pieces of a
module. Link every path to the relevant [circuit registry](CIRCUIT_REGISTRY.md),
[module dossiers](circuits/MODULE_DOSSIERS.md), primary receipts, and causal tests.

## Evidence labels

- **Proposed:** endpoint and factors are named, but the folding identity is not yet checked.
- **Exact algebra:** the selected native terms replay exactly under stated inputs/background.
- **Approximate path:** error is measured under a stated distribution and norm.
- **Identified:** frozen terms predict held-out cases and survive plausible controls.
- **Adopted:** the executable path passes its registered causal, composition, and simplicity bars.

Low rank, reconstruction quality, or a weight overlap does not advance a path beyond
`Approximate path`. Quantization is outside this registry's simplicity criterion.

## Required path record

Each detailed path dossier should state:

1. semantic decision point, token positions, endpoint reader, and direction of folding;
2. ordered native factors with tensor shapes, biases, RMS/rotary/nonlinear semantics, and gauges;
3. residual-source decomposition and the exact included and omitted self/cross terms;
4. background or input measure, approximation norm, and exact-replay tolerance;
5. literal program price: independent scalars, tensors, nodes, edges, state, and execution cost;
6. held-out predictive and selective causal falsifiers;
7. linked circuit records and module dossiers.

For a bilinear reader with residual inputs $r=\sum_i r_i$ and $s=\sum_j s_j$,
keep the ordered source terms explicit:

$$
B(r,s)=\sum_{i,j} B(r_i,s_j).
$$

This permits a path to retain, for example, an earlier attention-to-MLP cross term
without treating either complete native module as the semantic unit.

## Current path families

| ID | Direction and endpoint | Candidate native terms | Current evidence | Live handoff |
|---|---|---|---|---|
| PATH-SUBJECT-001 | MLP8 input/context forward to L11H3 subject-number write and its downstream readers | Native-axis coordinate multiplied by the head-local response to an MLP6/7 number-source switch | Circuit record `grammatical_subject_number.v34`: exact `[1,z,s,zs]` reaches `.3769` cross-construction relative L2. A mean donor-free source predicts $s$ at `.2880` error; rank-2 activation/displacement PCA reaches only `.2872`, and composed coefficient errors remain `.5159/.5107` | Preserve the transferable response scalar and stop activation-rank sweeps on these rows. On the next circuit hour, define a response-oriented basis from the frozen head operator; require fresh causal substitution after discovery |
| PATH-SET2-001 | Regional UK/US output contrast backward through MLP17 to earlier attention9 and MLP16 | Exact residual propagation followed by MLP17, attention-head, and QK source cross terms | Head9.8's three late-touching QK1 carry blocks replay with `.1119` selected-row and `.1385` fresh-row error; their exact grouped score has bilinear rank two and uses 256 rather than 384 scalar products per cell. Recursive removal is material (`.528–.598` of full-head effect). A fresh 2×2 intervention composes that route with the independently identified head8.2-induced current-value input: the explicit routing×value term is `.591–.619` of the smaller head-space single and the recursive target interaction is `.569–.610` of the smaller behavioral single. A fifth arm attributes that interaction to the explicit head product at `.9993/.9989` cosine and `.165/.097` relative error; suffix curvature mainly changes magnitude. Instruction sources compress the cross vector on discovery (`.191/.150` error) and fresh rows (`.189/.235`), with recursive cross-effect errors `.247/.323`. Cross-specific controls are all below `.097`. The failed work/jobs effect is the routing singleton (`.5260` versus `.5304` additive RMS). DD/DR/RD splitting is distributed: no block localizes collateral or improves selectivity by 30%; block sums replay full target/control within `.206/.081`. A prospective fresh union screen also rejects query-late `DD+DR`, key-late `DD+RD`, and cross-only `DR+RD` as selective reusable units. Its downstream response is led by direct attention9 (`.5422`) and attention17 (`.2495`), while MLP17 is anti-aligned (`−.1378`). The attention17 response localizes to head17.2: norm ratio `1.0557`, cosine `.9990`. The frozen QK2-main + value-main + QK2×value program transfers without reselection to 48 wholly fresh rows: response `.02136` relative L2/`.999976` cosine, causal installation `.02098`, removal `.02090`, perfect family signs, and equal-norm same-head nulls `.898–1.045`. A BF16-only first receipt narrowly failed its exact expansion audit (`2.285e-6` versus `2e-6`); a bound FP64 offline-contraction correction reduced that error to `2.97e-9` while changing every scientific metric by less than `4.1e-6` | Treat the native-QK1/edited-QK2-and-value corner as a fresh-confirmed response module, but count all three factor inputs as live ports. Next split QK2 and value by earlier-source terms or learn donor-free source predictors; do not call this extracted until those ports close. Retain the complete distributed head9.8 QK1 group and close its singleton/union pruning |
| PATH-SET1-001 | MLP9 forward through attention10 into MLP10 and its larger parent | QK/value producer terms and downstream bilinear self/cross terms | Approximate sparse parent screens exist; individual and sign exceptions remain | Use circuit-localized reader/writer evidence to select terms before another folded approximation |
| PATH-UNEMBED-001 | Token or token-family unembedding rows backward through final RMS, MLP17, and earlier contributors | Shared unembedding components, token-specific remainder, MLP17 bilinear terms, selected earlier residual sources | Proposed path family with partial setting3 fits; no adopted full path | Choose one semantic logit contrast and derive its exact backward factors before fitting |
| PATH-EMBED-001 | Embedding forward through selected early decoders to a named intermediate reader | Token/position embedding pieces and selected early attention/MLP self/cross terms | Unopened under the alternating-track directive | Start only from a circuit with a named downstream decision point and selective intervention |

## Primary pointers

- [Joint interaction-path proposal](../polynomial_causal/explanations/for_logan/interaction_path_decomposition_proposal_2026-09-11.md)
- [Setting2 sparse regional math](../polynomial_causal/INTERACTION_SPARSE_REGIONAL_V1_MATH.md)
- [Head17 output-block fit math](../polynomial_causal/HEAD17_OUTPUT_BLOCK_FIT_V1_MATH.md)
- [MLP16 × head17.2 exact-fold null](../polynomial_causal/SETTING2_MLP16_HEAD17_CROSS_TERM_FOLD_V1_RESULT.json)
- [MLP16 × head17.2 preregistration](../polynomial_causal/SETTING2_MLP16_HEAD17_CROSS_TERM_FOLD_V1_PREREGISTRATION.md)
- [Selected-reader six-term census](../polynomial_causal/SETTING2_SELECTED_READER_SIX_TERM_CENSUS_V1_RESULT.json)
- [Six-term census preregistration](../polynomial_causal/SETTING2_SELECTED_READER_SIX_TERM_CENSUS_V1_PREREGISTRATION.md)
- [Earlier-residual/attention census](../polynomial_causal/SETTING2_EARLIER_RESIDUAL_ATTENTION_TERM_CENSUS_V1_RESULT.json)
- [Earlier-residual/attention preregistration](../polynomial_causal/SETTING2_EARLIER_RESIDUAL_ATTENTION_TERM_CENSUS_V1_PREREGISTRATION.md)
- [Task-matched regional term census](circuits/fast_screens/setting2_regional_four_source_term_census_v1_result.json)
- [Task-matched regional preregistration](../polynomial_causal/SETTING2_REGIONAL_FOUR_SOURCE_TERM_CENSUS_V1_PREREGISTRATION.md)
- [Regional upstream-source fold](circuits/fast_screens/setting2_regional_mlp16_upstream_source_fold_v1_result.json)
- [Upstream-source fold preregistration](../polynomial_causal/SETTING2_REGIONAL_MLP16_UPSTREAM_SOURCE_FOLD_V1_PREREGISTRATION.md)
- [Attention9-head × MLP16 fold](circuits/fast_screens/setting2_regional_attn9_head_mlp16_fold_v2_result.json)
- [Attention9-head V2 precision correction](../polynomial_causal/SETTING2_REGIONAL_ATTN9_HEAD_MLP16_FOLD_V2_CORRECTION.md)
- [Head9.8 QK source-interaction fold](circuits/fast_screens/setting2_regional_head9_8_qk_source_fold_v3_result.json)
- [Head9.8 QK source-fold preregistration](../polynomial_causal/SETTING2_REGIONAL_HEAD9_8_QK_SOURCE_FOLD_V1_PREREGISTRATION.md)
- [Head9.8 QK1 carry-source fold](circuits/fast_screens/setting2_regional_head9_8_qk1_carry_source_fold_v1_result.json)
- [Head9.8 QK1 carry-source preregistration](../polynomial_causal/SETTING2_REGIONAL_HEAD9_8_QK1_CARRY_SOURCE_FOLD_V1_PREREGISTRATION.md)
- [Head9.8 QK1 late-group fold](circuits/fast_screens/setting2_regional_head9_8_qk1_late_group_fold_v1_result.json)
- [Head9.8 QK1 late-group preregistration](../polynomial_causal/SETTING2_REGIONAL_HEAD9_8_QK1_LATE_GROUP_FOLD_V1_PREREGISTRATION.md)
- [Fresh late-group routing null](circuits/fast_screens/setting2_regional_head9_8_qk1_late_group_fresh_routing_v3_result.json)
- [Fresh late-group routing preregistration](../polynomial_causal/SETTING2_REGIONAL_HEAD9_8_QK1_LATE_GROUP_FRESH_ROUTING_V1_PREREGISTRATION.md)
- [Fresh QK1 routing × current-value composition](circuits/fast_screens/setting2_regional_qk1_value_composition_fresh_v1_result.json)
- [Fresh composition preregistration](../polynomial_causal/SETTING2_REGIONAL_QK1_VALUE_COMPOSITION_FRESH_V1_PREREGISTRATION.md)
- [Explicit head-cross attribution](circuits/fast_screens/setting2_regional_qk1_value_interaction_attribution_v1_result.json)
- [Head-cross attribution preregistration](../polynomial_causal/SETTING2_REGIONAL_QK1_VALUE_INTERACTION_ATTRIBUTION_V1_PREREGISTRATION.md)
- [Head-cross source census](circuits/fast_screens/setting2_regional_head_cross_source_census_v2_result.json)
- [Source-census preregistration](../polynomial_causal/SETTING2_REGIONAL_HEAD_CROSS_SOURCE_CENSUS_V1_PREREGISTRATION.md)
- [Fresh instruction-cross null](circuits/fast_screens/setting2_regional_instruction_cross_fresh_v1_result.json)
- [Fresh instruction-cross preregistration](../polynomial_causal/SETTING2_REGIONAL_INSTRUCTION_CROSS_FRESH_V1_PREREGISTRATION.md)
- [Cross-specific collateral attribution](circuits/fast_screens/setting2_regional_instruction_cross_collateral_attribution_v1_result.json)
- [Collateral-attribution preregistration](../polynomial_causal/SETTING2_REGIONAL_INSTRUCTION_CROSS_COLLATERAL_ATTRIBUTION_V1_PREREGISTRATION.md)
- [Additive branch collateral split](circuits/fast_screens/setting2_regional_additive_branch_collateral_v1_result.json)
- [Additive branch preregistration](../polynomial_causal/SETTING2_REGIONAL_ADDITIVE_BRANCH_COLLATERAL_V1_PREREGISTRATION.md)
- [QK1 block collateral split](circuits/fast_screens/setting2_regional_qk1_block_collateral_v1_result.json)
- [QK1 block preregistration](../polynomial_causal/SETTING2_REGIONAL_QK1_BLOCK_COLLATERAL_V1_PREREGISTRATION.md)
- [Fresh QK1 structured-union null](circuits/fast_screens/setting2_regional_qk1_structured_union_fresh_v1_result.json)
- [Structured-union preregistration](../polynomial_causal/SETTING2_REGIONAL_QK1_STRUCTURED_UNION_FRESH_V1_PREREGISTRATION.md)
- [QK1 edit downstream-response census](circuits/fast_screens/setting2_regional_qk1_edit_downstream_response_census_v2_result.json)
- [Downstream-response census preregistration](../polynomial_causal/SETTING2_REGIONAL_QK1_EDIT_DOWNSTREAM_RESPONSE_CENSUS_V1_PREREGISTRATION.md)
- [Attention17 nine-head response fold](circuits/fast_screens/setting2_regional_attention17_head_response_fold_v1_result.json)
- [Attention17 head-fold preregistration](../polynomial_causal/SETTING2_REGIONAL_ATTENTION17_HEAD_RESPONSE_FOLD_V1_PREREGISTRATION.md)
- [Head17.2 factor-interaction fold](circuits/fast_screens/setting2_regional_attention17h2_factor_interaction_fold_v1_result.json)
- [Head17.2 factor-fold preregistration](../polynomial_causal/SETTING2_REGIONAL_ATTENTION17H2_FACTOR_INTERACTION_FOLD_V1_PREREGISTRATION.md)
- [Fresh head17.2 frozen-corner confirmation](circuits/fast_screens/setting2_regional_attention17h2_factor_corner_fresh_v2_result.json)
- [Fresh frozen-corner preregistration](../polynomial_causal/SETTING2_REGIONAL_ATTENTION17H2_FACTOR_CORNER_FRESH_V1_PREREGISTRATION.md)
- [Fresh frozen-corner precision correction](../polynomial_causal/SETTING2_REGIONAL_ATTENTION17H2_FACTOR_CORNER_FRESH_V2_CORRECTION.md)
- [Three-block bilinear-rank control](../polynomial_causal/THREE_BLOCK_BILINEAR_RANK_CONTROL_20260915_0549_RESULT.json)
- [05:52 mathematical, organization, and efficiency review](../polynomial_causal/THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-15_0552.md)
- [Subject upstream-context preregistration](../polynomial_causal/SUBJECT_NUMBER_UPSTREAM_CONTEXT_COORDINATE_DISCOVERY_V1_PREREGISTRATION.md)
- [Subject coefficient-law result](circuits/fast_screens/subject_number_coefficient_bilinear_law_v1_result.json)
- [Subject native-axis result](circuits/fast_screens/subject_number_native_weight_axis_v1_result.json)
- [Subject native scalar-readout null](circuits/fast_screens/subject_number_native_scalar_feature_discovery_v1_result.json)
- [Subject upstream context-coordinate null](circuits/fast_screens/subject_number_upstream_context_coordinate_discovery_v1_result.json)
- [Subject native head-response coordinate result](circuits/fast_screens/subject_number_native_head_response_coordinate_discovery_v2_result.json)
- [Subject native head-response preregistration](../polynomial_causal/SUBJECT_NUMBER_NATIVE_HEAD_RESPONSE_COORDINATE_DISCOVERY_V1_PREREGISTRATION.md)
- [Subject donor-free response-proxy null](circuits/fast_screens/subject_number_donor_free_head_response_proxy_v2_result.json)
- [Subject donor-free response-proxy preregistration](../polynomial_causal/SUBJECT_NUMBER_DONOR_FREE_HEAD_RESPONSE_PROXY_V1_PREREGISTRATION.md)
- [Subject rank-2 recipient-state proxy null](circuits/fast_screens/subject_number_rank2_recipient_state_response_proxy_v2_result.json)
- [Subject-number machine dossier](circuits/task_subject_verb_number_agreement.json)
- [MLP dossier index](../polynomial_causal/explanations/MLP_MODULE_DOSSIER_INDEX.md)

Update this index only when a path's evidence label, scope, live handoff, or primary
dossier link changes. Put experiment detail in the path dossier or primary receipt.
