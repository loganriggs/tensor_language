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
| PATH-SET2-001 | Regional UK/US output contrast backward through MLP17 to earlier attention9 and MLP16 | Exact residual propagation followed by MLP17, attention-head, and QK source cross terms | Head9.8's three late-touching QK1 carry blocks replay with `.1119` selected-row and `.1385` fresh-row error; their exact grouped score has bilinear rank two and uses 256 rather than 384 scalar products per cell. Recursive removal is material (`.528–.598` of full-head effect). A fresh 2×2 intervention composes that route with the independently identified head8.2-induced current-value input: the explicit routing×value term is `.591–.619` of the smaller head-space single and the recursive target interaction is `.569–.610` of the smaller behavioral single. A fifth arm attributes that interaction to the explicit head product at `.9993/.9989` cosine and `.165/.097` relative error; suffix curvature mainly changes magnitude. Instruction sources compress the cross vector on discovery (`.191/.150` error) and fresh rows (`.189/.235`), with recursive cross-effect errors `.247/.323`. Cross-specific controls are all below `.097`. The failed work/jobs effect is the routing singleton (`.5260` versus `.5304` additive RMS). DD/DR/RD splitting is distributed: no block localizes collateral or improves selectivity by 30%; block sums replay full target/control within `.206/.081`. A prospective fresh union screen also rejects query-late `DD+DR`, key-late `DD+RD`, and cross-only `DR+RD` as selective reusable units. Its downstream response is led by direct attention9 (`.5422`) and attention17 (`.2495`), while MLP17 is anti-aligned (`−.1378`). The attention17 response localizes to head17.2: norm ratio `1.0557`, cosine `.9990`. The frozen QK2-main + value-main + QK2×value program transfers without reselection to 48 wholly fresh rows: response `.02136` relative L2/`.999976` cosine, causal installation `.02098`, removal `.02090`, perfect family signs, and equal-norm same-head nulls `.898–1.045`. A BF16-only first receipt narrowly failed its exact expansion audit (`2.285e-6` versus `2e-6`); a bound FP64 offline-contraction correction reduced that error to `2.97e-9` while changing every scientific metric by less than `4.1e-6`. An exhaustive preservation-aware search over all 6,885 unit-gain supports through five module-write edges finds a discovery candidate `attn9+mlp9+attn10+attn11+mlp15` (`.2076` response, `.1976/.1985` causal errors, all controls pass), but frozen fresh testing rejects it: response `.25068` narrowly misses `.25` and controls reach `1.1488×` their allowed bound despite causal errors `.2377/.2390`. Thus the entire ≤5 whole-module edge class is closed as non-transferable. Exact native-head splitting then finds an eight-edge discovery program (`attn9h8`, `attn10h2/h5`, `attn11h1/h6/h8`, `mlp9`, `attn15`) that passes every opened-panel gate at `.8311` minimax. On frozen cue-role crossover rows it predicts the target even better (`.1278` response, `.1234/.1230` causal errors) but fails preservation at `1.3163×`, worse than the complete-head expansion's `1.1450×`. Native head identity is therefore also not the transferable coordinate. A multi-reader VJP basis then exposes a rank-eight discovery arm that passes every gate while retaining `20.0%` of source norm, but frozen competing-cues testing rejects preservation at `1.1881×`. The predeclared rank-eight source-PCA comparator passes that first holdout (`.9708×`) while retaining `59.0%`, then fails a second short-context/unseen-endpoint panel at `1.1185×` despite `.1980` response and `.1898/.1900` causal errors. A one-bit maximum-source-energy gate between the response and competing-cues PCA bases also fails on a third zero-overlap panel: target response and causal errors remain `.2464` and `.2408/.2415`, but preservation reaches `1.4590×`; neither the rank-16 union (`1.4414×`) nor the full unprojected support (`1.4008×`) preserves controls. Thus the failure is not merely low-rank truncation, and energy does not expose the needed context slot. Separately, exact per-layer causal-Hessian allocation of the earlier CrossFirst child/remainder interaction closes below `7.7e-7`; a frozen MLP10 + attention17 correction transfers to 48 new prefixes and leaves `.0175--.0944` child-relative aggregate composition error, beating four fixed stage-pair nulls. Its stronger per-row finite-interaction fidelity fails in two families (`.8148/.8544` cosine), so it is a compact aggregate composition correction, not a complete sparse interaction circuit. A second zero-overlap endpoint panel rejects a retrospectively chosen four-stage support (`.325` minimax; no 10% ablation advantage), and even the complete Hessian leaves `.403` child-relative error on `licence/license`. Scale contraction validates the Hessian locally but exposes native-scale higher-order curvature. The exact direct cubic is mechanically sound (symmetry error ≤`5.22e-7`) and materially improves that endpoint (`.3741→.1209` finite-relative error), but remains a preregistered null: one endpoint is `.1303` child-relative, two easy endpoints worsen, and the weakest family improves only `22.4%`. Full-width raw-MLP9 DCT is also a valid null: four factors are stable across seeds, but their reader coverage `.0576` matches a random rank-four median `.0581`; rank-eight coverage `.0711` is below the random median `.0794`. A balanced scalar most-additive split (`lambda=.525`) is stable and cuts local interaction `42.3%` to `.0790`, but four orthogonal writers reach `.0327--.0356`; it is generic rebalancing, not a CrossFirst-specific decomposition. Keeping native context/RMS open produces a genuine transferable rank-eight high-curvature response core (`.997855` response-norm retention on both panels; stable inputs carry `148.34x` random-probe energy), but it still has chance-level CrossFirst reader coverage (`.0719/.0882`). Its exported rank-eight node is a diagnostic null because low-energy pair terms fail under cancellation; the prospective rank-16 correction predicts a second fresh panel at `.00493` relative L2, keeps every ordered pair below `.01047` and every frozen mixture below `.00801`, and replays exactly in isolation using one native `z9` port and `.6751` of native MLP9 tensor parameters | Preserve the fresh-confirmed factor corner and the fresh two-stage composition correction at their stated scopes. Stop widening the failed source support or adding gates around its two fixed bases. Jointly identify a source coordinate and a weight/state-derived context gate across multiple environments—or change the source support—then freeze the whole rule on an untouched panel. Do not keep raising Taylor order on the live deep suffix: it increases evaluator complexity without extraction. Raw or contextual local MLP9 DCT is closed as this behavior's reader source. Preserve the rank-16 contextual DCT package as a generic extracted response primitive; the next promotion criterion is a downstream reader plus fresh selective install/remove, not more local response compression. Do not behavior-test the scalar most-additive split; a real next split must partition consumer-relevant vector/tensor modes across multiple readers |
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
- [Head17.2 port-source fold null](circuits/fast_screens/setting2_regional_attention17h2_port_source_fold_v2_result.json)
- [Port-source preregistration](../polynomial_causal/SETTING2_REGIONAL_ATTENTION17H2_PORT_SOURCE_FOLD_V1_PREREGISTRATION.md)
- [Port-source BF16 correction](../polynomial_causal/SETTING2_REGIONAL_ATTENTION17H2_PORT_SOURCE_FOLD_V2_CORRECTION.md)
- [Preservation-aware width-five discovery](circuits/fast_screens/setting2_regional_attention17h2_port_preservation_search_v3_result.json)
- [Frozen five-edge fresh null](circuits/fast_screens/setting2_regional_attention17h2_port_preservation_fresh_v1_result.json)
- [Fresh five-edge preregistration](../polynomial_causal/SETTING2_REGIONAL_ATTENTION17H2_PORT_PRESERVATION_FRESH_V1_PREREGISTRATION.md)
- [Native-head discovery candidate](circuits/fast_screens/setting2_regional_attention17h2_head_source_beam_v1_result.json)
- [Frozen native-head crossover null](circuits/fast_screens/setting2_regional_attention17h2_head_source_fresh_v1_result.json)
- [Native-head crossover preregistration](../polynomial_causal/SETTING2_REGIONAL_ATTENTION17H2_HEAD_SOURCE_FRESH_V1_PREREGISTRATION.md)
- [Consumer-response rank frontier](circuits/fast_screens/setting2_regional_attention17h2_consumer_response_basis_v1_result.json)
- [Exported rank-eight projector](circuits/fast_screens/setting2_regional_attention17h2_response_basis_export_v1_result.json)
- [Frozen response-basis null](circuits/fast_screens/setting2_regional_attention17h2_response_basis_fresh_v1_result.json)
- [Second-panel source-PCA null](circuits/fast_screens/setting2_regional_attention17h2_source_pca_fresh_v1_result.json)
- [Context-gated source-basis null](circuits/fast_screens/setting2_regional_attention17h2_context_gated_source_basis_v1_result.json)
- [Context-gated preregistration](../polynomial_causal/SETTING2_REGIONAL_ATTENTION17H2_CONTEXT_GATED_SOURCE_BASIS_V1_PREREGISTRATION.md)
- [Context-gated execution note](../polynomial_causal/SETTING2_REGIONAL_ATTENTION17H2_CONTEXT_GATED_SOURCE_BASIS_V1_EXECUTION_NOTE.md)
- [CrossFirst per-layer causal-Hessian allocation](../polynomial_causal/CROSSFIRST_PER_LAYER_CAUSAL_HESSIAN_V1_RESULT.json)
- [Fresh MLP10 + attention17 composition correction](../polynomial_causal/CROSSFIRST_HESSIAN_TOP2_FRESH_V1_RESULT.json)
- [Fresh composition scale/sign audit](../polynomial_causal/CROSSFIRST_HESSIAN_TOP2_FRESH_V1_AUDIT.json)
- [Fresh four-stage Hessian null](../polynomial_causal/CROSSFIRST_HESSIAN_FOUR_STAGE_FRESH_V1_RESULT.json)
- [CrossFirst Hessian scale curve](../polynomial_causal/CROSSFIRST_HESSIAN_SCALE_CURVE_V1_RESULT.json)
- [CrossFirst direct cubic null](../polynomial_causal/CROSSFIRST_DIRECT_CUBIC_V1_RESULT.json)
- [Direct cubic implementation correction](../polynomial_causal/CROSSFIRST_DIRECT_CUBIC_V1_CORRECTION.md)
- [MLP9 DCT unsupervised-reader null](../polynomial_causal/MLP9_DCT_UNSUPERVISED_READER_V1_RESULT.json)
- [MLP9 DCT implementation correction](../polynomial_causal/MLP9_DCT_UNSUPERVISED_READER_V1_CORRECTION.md)
- [MLP9 DCT rank-stability audit](../polynomial_causal/MLP9_DCT_UNSUPERVISED_READER_RANK_STABILITY_V1_AUDIT.json)
- [CrossFirst most-additive split discovery null](../polynomial_causal/CROSSFIRST_MOST_ADDITIVE_SPLIT_DISCOVERY_V1_RESULT.json)
- [Most-additive split preregistration](../polynomial_causal/CROSSFIRST_MOST_ADDITIVE_SPLIT_DISCOVERY_V1_PREREGISTRATION.md)
- [MLP9 contextual DCT range result](../polynomial_causal/MLP9_CONTEXTUAL_DCT_RANGE_V1_RESULT.json)
- [MLP9 contextual DCT range preregistration](../polynomial_causal/MLP9_CONTEXTUAL_DCT_RANGE_V1_PREREGISTRATION.md)
- [MLP9 contextual DCT rank-eight node diagnostic null](../polynomial_causal/MLP9_CONTEXTUAL_DCT_NODE_FRESH_V1_RESULT.json)
- [MLP9 contextual DCT rank-16 extracted compositional node](../polynomial_causal/MLP9_CONTEXTUAL_DCT_NODE_RANK16_FRESH_V2_RESULT.json)
- [MLP9 contextual DCT rank-16 preregistration](../polynomial_causal/MLP9_CONTEXTUAL_DCT_NODE_RANK16_FRESH_V2_PREREGISTRATION.md)
- [MLP17 causal-Hessian identity certificate](../polynomial_causal/MLP17_CAUSAL_HESSIAN_IDENTITY_V1_RESULT.json)
- [Causal-Hessian identity preregistration](../polynomial_causal/MLP17_CAUSAL_HESSIAN_IDENTITY_V1_PREREGISTRATION.md)
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
