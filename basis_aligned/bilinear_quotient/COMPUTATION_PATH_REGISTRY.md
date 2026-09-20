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
| PATH-REGIONAL-CITY-001 | Residual6 and token IDs forward through attention7 and MLP7 into the head8.2 city-removal write | All nine attention7 heads; MLP7 `L/R/Down/bias`; five folded head8.2 readers; full ordered residual6/initial7/attention7 self and cross terms; native QK1×QK2×V with RMS and rotary semantics | Corrected exact-RMS replay reaches `1.97e-6` maximum relative write error on 40 opened fixtures and `1.61e-6` on 40 fresh outcome-blind FineWeb sequences (20 distinct documents, 240 probes), with zero off-support write. The packed eight-head approximation separately predicts selective removal on fresh Pile rows, but FineWeb direction transfer fails and key/value composition fails (`interaction/smaller=2.34`). Linked packages: `city_attention7_drop3_v1`, `city_residual6_single_input_v1`, `city_mlp8_value_mediator_v1`; module navigation: `MLP7_REGIONAL_PATH_ADDENDUM_2026-09-17.md` and the MLP8 index addendum. This is an exact extracted path at a declared boundary, not a five-property circuit. | On the next CIRCUIT hour, install the frozen corrected upstream write before the existing coupled value operator on preregistered structurally varied, distinct context cells; score full-suffix prediction, equal-norm same-site nulls, at least three unrelated readers, and explicit joint-versus-single composition. Preserve the complete coupling; do not re-open arbitrary source partitions. |
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
- [MLP9 contextual DCT downstream-reader discovery null](../polynomial_causal/MLP9_CONTEXTUAL_DCT_DOWNSTREAM_READER_DISCOVERY_V1_RESULT.json)
- [MLP9 contextual DCT finite-scale discovery](../polynomial_causal/MLP9_CONTEXTUAL_DCT_FINITE_SCALE_DISCOVERY_V1_RESULT.json)
- [MLP9 contextual DCT fresh causal install/remove result](../polynomial_causal/MLP9_CONTEXTUAL_DCT_CAUSAL_FRESH_V1_RESULT.json)
- [MLP9 contextual DCT structure-matched specificity audit](../polynomial_causal/MLP9_CONTEXTUAL_DCT_CAUSAL_FRESH_V1_SPECIFICITY_AUDIT_RESULT.json)
- [Contextual DCT to natural equality-response bridge null](../polynomial_causal/MLP9_CONTEXTUAL_DCT_EQUALITY_RESPONSE_DISCOVERY_V1_RESULT.json)
- [Equality consumer-response basis transfer null](../polynomial_causal/MLP9_EQUALITY_CONSUMER_RESPONSE_BASIS_DISCOVERY_V1_RESULT.json)
- [Equality projected-MLP9-write causal ceiling null](../polynomial_causal/MLP9_EQUALITY_PROJECTED_WRITE_CAUSAL_DISCOVERY_V1_RESULT.json)
- [Equality post-MLP9 two-edge oracle boundary factor](../polynomial_causal/EQUALITY_POST_MLP9_STATE_FACTORIAL_DISCOVERY_V1_RESULT.json)
- [Equality boundary-factor execution note](../polynomial_causal/EQUALITY_POST_MLP9_STATE_FACTORIAL_DISCOVERY_V1_EXECUTION_NOTE.md)
- [Equality pre-MLP9 three-edge sparse factorial](../polynomial_causal/EQUALITY_PRE_MLP9_THREE_EDGE_FACTORIAL_DISCOVERY_V1_RESULT.json)
- [Frozen A8 code-OOD calibration null](../polynomial_causal/EQUALITY_A8_EDGE_CODE_OOD_CONFIRMATION_V1_RESULT.json)
- [Projected-payload extracted-node null](../polynomial_causal/EQUALITY_L8H4_EXTRACTED_NODE_CODE_OOD_V1_RESULT.json)
- [Exact-order extracted-node result](../polynomial_causal/EQUALITY_L8H4_EXACT_ORDER_NODE_CODE_OOD_V2_RESULT.json)
- [Exact reversible equality-edge result](../polynomial_causal/EQUALITY_L8H4_REVERSIBLE_EDGE_CODE_OOD_V3_RESULT.json)
- [Reusable L5H5 score-port result](../polynomial_causal/EQUALITY_REUSABLE_SCORE_PORT_CODE_OOD_V1_RESULT.json)
- [L5H5-to-L8H4 score-adapter export](../polynomial_causal/EQUALITY_L5H5_SCORE_ADAPTER_EXPORT_V1_RESULT.json)
- [L5H5 bilinear score-node causal-mask red-team null](../polynomial_causal/EQUALITY_L5H5_BILINEAR_SCORE_NODE_CODE_OOD_V1_RESULT.json)
- [Exact causally masked L5H5 bilinear score node](../polynomial_causal/EQUALITY_L5H5_BILINEAR_SCORE_NODE_CODE_OOD_V2_RESULT.json)
- [Sparse L2+L3+L4 residual-source graph for L5H5](../polynomial_causal/EQUALITY_L5H5_RESIDUAL_SOURCE_GRAPH_V1_RESULT.json)
- [Residual correction-port red-team](../polynomial_causal/EQUALITY_L5H5_RESIDUAL_CORRECTION_REDTEAM_V1_RESULT.json)
- [Exported L2+L3+L4 to L5H5 score graph](../polynomial_causal/EQUALITY_L5H5_L234_SCORE_GRAPH_EXPORT_V1_RESULT.json)
- [Invalid flattened module-write refinement](../polynomial_causal/EQUALITY_L5H5_L234_MODULE_WRITE_GRAPH_V1_RESULT.json)
- [Canonical module-write sparsity null](../polynomial_causal/EQUALITY_L5H5_L234_MODULE_WRITE_GRAPH_V2_RESULT.json)
- [Exported five-write L5H5 score graph](../polynomial_causal/EQUALITY_L5H5_FIVE_WRITE_SCORE_GRAPH_EXPORT_V1_RESULT.json)
- [Native M4 product sparsity null](../polynomial_causal/EQUALITY_L5H5_M4_PRODUCT_CHANNEL_GRAPH_V1_RESULT.json)
- [Invalid float32 M4 contracted-mode receipt](../polynomial_causal/EQUALITY_L5H5_M4_CONTRACTED_MODE_GRAPH_V1_RESULT.json)
- [Float64 M4 contracted-mode compactness/composition null](../polynomial_causal/EQUALITY_L5H5_M4_CONTRACTED_MODE_GRAPH_V2_RESULT.json)
- [M4 most-additive mode-split null](../polynomial_causal/EQUALITY_L5H5_M4_MOST_ADDITIVE_MODE_SPLIT_V1_RESULT.json)
- [Exported M4 rank-256 mode boundary](../polynomial_causal/EQUALITY_L5H5_M4_RANK256_MODE_GRAPH_EXPORT_V1_RESULT.json)
- [Explicit M4 child/remainder interaction graph](../polynomial_causal/EQUALITY_L5H5_M4_EXPLICIT_INTERACTION_GRAPH_V1_RESULT.json)
- [Exported explicit M4 interaction graph](../polynomial_causal/EQUALITY_L5H5_M4_EXPLICIT_INTERACTION_GRAPH_EXPORT_V1_RESULT.json)
- [M4 shared-projection kernel null](../polynomial_causal/EQUALITY_L5H5_M4_SHARED_PROJECTION_KERNEL_V1_RESULT.json)
- [M4 shared-projection arithmetic-gauge null](../polynomial_causal/EQUALITY_L5H5_M4_SHARED_PROJECTION_GAUGE_V1_RESULT.json)
- [M4 one-map score-space correction](../polynomial_causal/EQUALITY_L5H5_M4_SHARED_PROJECTION_CORRECTION_V1_RESULT.json)
- [M4 corrected shared-kernel behavioral null](../polynomial_causal/EQUALITY_L5H5_M4_SHARED_KERNEL_BEHAVIOR_V1_RESULT.json)
- [M4 behavior-selected native-projection null](../polynomial_causal/EQUALITY_L5H5_M4_BEHAVIORAL_PROJECTION_SELECTION_V1_RESULT.json)
- [M4 exact four-port Möbius low-order null](../polynomial_causal/EQUALITY_L5H5_M4_PORT_MOBIUS_ORDER_V1_RESULT.json)
- [Invalid M4 three-node real-arithmetic factor graph](../polynomial_causal/EQUALITY_L5H5_M4_TWO_FACTOR_CORRECTION_GRAPH_V1_RESULT.json)
- [M4 precision-corrected four-node factor graph](../polynomial_causal/EQUALITY_L5H5_M4_TWO_FACTOR_CORRECTION_GRAPH_V2_RESULT.json)
- [Exported M4 precision-factor graph](../polynomial_causal/EQUALITY_L5H5_M4_PRECISION_FACTOR_GRAPH_EXPORT_V1_RESULT.json)
- [M4 no-oracle raw-port factor graph](../polynomial_causal/EQUALITY_L5H5_M4_RAW_PORT_FACTOR_GRAPH_V1_RESULT.json)
- [Exported M4 raw-port factor graph](../polynomial_causal/EQUALITY_L5H5_M4_RAW_PORT_FACTOR_GRAPH_EXPORT_V1_RESULT.json)
- [M4 residual-port factor graph](../polynomial_causal/EQUALITY_L5H5_M4_RESIDUAL_PORT_FACTOR_GRAPH_V1_RESULT.json)
- [Exported M4 residual-port factor graph](../polynomial_causal/EQUALITY_L5H5_M4_RESIDUAL_PORT_FACTOR_GRAPH_EXPORT_V1_RESULT.json)
- [A8 sparse-edge red-team note](../polynomial_causal/EQUALITY_A8_SPARSE_EDGE_EXECUTION_NOTE.md)
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

### 17 September: regional typed face, masks (1,4,5)

Cross-link: regional inherited-city edge / PATH-SET2-001 / head8.2 and head9.8-O.
[Package](../polynomial_causal/extracted_circuits/odd_attention8h2_typed_face_v1/README.md)
closes routing/current maps and the eight-token inherited-value table, leaving two
native-state inputs. Isolated write replay passes; strict all-readout recursive
replay fails and is preserved. [Fresh paired midpoint](../polynomial_causal/ODD_ATTENTION8H2_TYPED_FACE_REMOVAL_V1_RESULT.json)
passes on four unused contexts: target12.09xmedian of16equal-norm same-head nulls,
18/21capable cells attenuated by mean2.583%, controls .0449–.0971of target RMS.
This is a limited selective component, not a complete five-property circuit:
independent composition fails (interaction/minimum-single1.737), matched-effect
simplicity comparison and random-split specificity remain missing. [Current Logan report](../polynomial_causal/explanations/for_logan/research_update_2026-09-17_2110_regional_native_face.md).

### 17 September, 21:40: regional source transfer and composition specificity

[Fresh source transfer](../polynomial_causal/TYPED_FACE_KEY_SOURCE_FRESH_V1_RESULT.json) rejects carry-only (cue errors .411–.763, gate .35); carry plus MLP7 passes (.104–.183, gate .20) on eight unused contexts with shifted city position. Four new city token IDs occur; endpoints unchanged. Both frozen head-key norms pass the .10 behavioral gate. No source-selectivity claim or port-count reduction. [Physical-write composition](../polynomial_causal/TYPED_FACE_WRITE_COMPOSITION_V1_RESULT.json) is nearly additive but fails specificity: worst interaction is 1.91 times random-split median versus gate .50. This resolves the previously missing random-split comparison with a failure. The earlier two-input additive failure remains. [Current Logan report](../polynomial_causal/explanations/for_logan/research_update_2026-09-17_2140_regional_source_transfer.md) tracks all four properties and simplicity. Next donor MLP7 generator has [CPU algebra evidence](../polynomial_causal/TYPED_FACE_MLP7_DONOR_V1_CPU_RESULT.json), but native replay and full weight pricing remain required.

**21:46 donor-boundary update:** [MLP7 donor package](../polynomial_causal/extracted_circuits/odd_attention8h2_mlp7_donor_v1/README.md) passes native and isolated CPU replay on 16 opened sequences. Donor input now precedes MLP7; two native-state arrays remain. Storage rises to 16,836,739 floats. Exact extraction boundary progress, no compression or new fresh/selectivity/composition certification.

**21:52 source handoff:** [MLP7 input-source intervention](../polynomial_causal/MLP7_INPUT_SOURCE_V1_RESULT.json) fails both residual6-only and residual6+attention7 omission hypotheses on eight opened contexts. Initial7 stays explicit; posthoc source5 is unconfirmed. [Local exact response](../polynomial_causal/MLP7_EXACT_RESPONSE_V1_RESULT.json) preserves normalization, cross/quadratic terms and residual skip. Next circuit work tests structural transfer of the whole retained path.

**22:00 structural transfer:** [Full retained face](../polynomial_causal/TYPED_FACE_STRUCTURE_V1_RESULT.json) passes all six registered gates across base/no-colon/no-quote/neither/short-frame. Four modified conditions fresh at freeze; four context/city cells per condition. Prediction17–20%, attenuation2.1–3.9%, beats16/16directions each, four collateral ratios<=.219. [Current report](../polynomial_causal/explanations/for_logan/research_update_2026-09-17_2200_regional_structure.md). Extraction still two native ports; composition specificity fails; source omissions remain rejected ([source failure](../polynomial_causal/MLP7_INPUT_SOURCE_V1_RESULT.json), [local exact response](../polynomial_causal/MLP7_EXACT_RESPONSE_V1_RESULT.json)). No broad OOD upgrade from posthoc predictor-null audit.

**22:07 prospective qualification:** [Fixed-baseline transfer](../polynomial_causal/TYPED_FACE_PROSPECTIVE_V1_RESULT.json) passes prediction improvement on all five fresh construction families; constants and text-OLS remain frozen. Minimum attenuation fails for line breaks (.0191105<.02). All paired directions remain positive, matched-null target and collateral gates pass. This qualifies the preceding all-pass punctuation screen, without retracting it. [Magnitude audit](../polynomial_causal/PROSPECTIVE_ATTENUATION_V1_AUDIT.json) distinguishes absolute damage and native gap; no downstream causal attribution yet.

**22:15 fixed-edit census:** [Response census](../polynomial_causal/TYPED_FACE_RESPONSE_CENSUS_V1_RESULT.json) passes reconstruction and family-stability gates but rejects direct-only and fixed early-MLP sufficiency. Attention17 has13–20%aligned response; this is not new head17.2 causal evidence and prior source-support failures remain. [Exact head9 value-difference cancellation](../polynomial_causal/ODD_VALUE_DELTA_RAW_V1_CPU_RESULT.json) removes inherited-first input algebraically and combines reentry states; native verification is pending. [Current report](../polynomial_causal/explanations/for_logan/research_update_2026-09-17_2215_regional_response.md).

**22:18 reduced head9 boundary:** [Export](../polynomial_causal/extracted_circuits/odd_value_delta_raw_v1/README.md) passes native and isolated CPU replay on40opened fixtures,901,121stored scalars. No first-value or separate initial-state input; raw9 and upstream delta remain. Wider head8/head9 boundary would require three native-state arrays. No fresh scientific gate, composition or whole-model simplicity promotion.

**22:25 combined boundary verified:** [Complete conditional write](../polynomial_causal/extracted_circuits/typed_face_composed_raw_v1/README.md) passes native and isolated40fixture replay. Three native-state inputs,1,788,419floats; no external first-values, initial-state or delta8 input. Native suffix remains; implementation composition is not causal composition certification. Earlier removal and random-split failures remain.

**22:33 destination partition null:** [Framing/clause split](../polynomial_causal/DESTINATION_PARTITION_V1_RESULT.json) has two live, nearly additive pieces but fails specificity1.0969xrandommedian,beats1/9. Stop arbitrary repartition search. Next [native8 scope registration](../polynomial_causal/TYPED_FACE_NATIVE8_SCOPE_V1_PREREGISTRATION.md) tests the same write at actual attention8 output against the conditional odd-value operator; implementation begun, no result yet.

### 17 September 23:41 — Regional MLP8 normalization screen

Both registered full-context normalization approximations pass opened causal gates (200 forwards, 3.45s). Frozen norm effect error .00031–.00072; no fresh/null upgrade. [Receipt](../polynomial_causal/REDUCED_MLP8_NORMALIZATION_V1_RESULT.json). Next one-head normalization candidate has 2.1–6.4% CPU local error, not a causal result: [diagnostic](../polynomial_causal/SINGLE_HEAD_NORMALIZATION_V1_CPU_RESULT.json), [registered screen](../polynomial_causal/SINGLE_HEAD_NORMALIZATION_V1_PREREGISTRATION.md). Exact package and composition failures preserved.

### 17 September23:48 — Single-head normalization fresh confirmation

[Fresh result](../polynomial_causal/SINGLE_HEAD_FRESH_V1_RESULT.json) passes all6gates:1.1–15%prediction error,16–22%attenuation,16/16nulls,collateral<=.163. Same endpoints and2native inputs; composition unchanged. [Pruned package CPU check](../polynomial_causal/SINGLE_HEAD_PRUNED_V1_CPU_RESULT.json):16,899,587floats,70tokens,40fixture replay pass; native/layout certification pending. This is a normalization approximation, not exact source elimination.

**23:54 certification and next source:** Pruned single-head package passes [native](../polynomial_causal/SINGLE_HEAD_PRUNED_V1_RESULT.json) and [isolated](../polynomial_causal/SINGLE_HEAD_PRUNED_V1_STANDALONE_RESULT.json) replay. Next [donor-free inherited-city removal protocol](../polynomial_causal/CITY_INHERITED_REMOVAL_V1_PREREGISTRATION.md) changes the counterfactual; one native input and7.2–13%local CPU response error, causal effect untested. No transferred swap/selectivity claims.

### 18 September00:07 — Regional inherited-only sign failure and full-city factorial

[New-endpoint removal](../polynomial_causal/CITY_REMOVAL_ENDPOINT_FRESH_V1_RESULT.json) predicts effects andpassescontrols/nulls butfails direction75%<90%plain-note. Exactnative removal sharesreversals; [cell diagnostic](../polynomial_causal/CITY_REMOVAL_ENDPOINT_V1_SIGN_DIAGNOSTIC.json). [Native current/inherited factorial](../polynomial_causal/CITY_VALUE_BRANCH_FACTORIAL_V1_RESULT.json) restores120/120positivefullcityremovals,8.4–14%attenuation; interaction.369>.35copyfamily fails. Keep fullcitycoupled; do notpromote independentpieces. [Nextscreen](../polynomial_causal/CITY_FULL_VALUE_REMOVAL_V1_PREREGISTRATION.md), [latestreport](../polynomial_causal/explanations/for_logan/research_update_2026-09-18_0003_removal_sign_failure.md). Paired extractioncertified; removalone-inputpackage/freshfullcity/compositionremainopen.

### 18 September00:18 — One-input full-city removal extraction

[Fresh full-city result](../polynomial_causal/CITY_FULL_FRESH_V1_RESULT.json) passes all6gates:2.2–13%predictionerror,10–13%attenuation,120/120positive,16/16nulls,collateral<=.167. [Native export](../polynomial_causal/CITY_FULL_EXTRACTED_V1_RESULT.json) and [isolated export](../polynomial_causal/CITY_FULL_EXTRACTED_V1_STANDALONE_RESULT.json) pass;16,877,827floats,53tokens,1native residual7 array. Inherited-onlyfreshsignfailure andbranchinteraction.369>.35remain. Naturalcorpusconfirmation next; no corpus/token-only/composition completeness claim.

### 2026-09-18 — Regional full-city removal: filtered Pile transfer
`CITY_FULL_PILE_V3_RESULT.json` passes all seven registered gates on 20 source documents, with unchanged candidate/null outputs relative to V2. V2 analytic-reference precision failure remains in `CITY_FULL_PILE_V2_RESULT.json`; V3 is an opened FP32 reference repair. Effect error .02364 aggregate, .06596 untouched natural arms, .02114 city substitutions. CPU document diagnostic finds six shared native/candidate reversals and two additional approximation sign errors. One native residual7 array and external suffix remain; inherited/current independent composition still fails. Canonical report: `basis_aligned/polynomial_causal/explanations/for_logan/research_update_2026-09-18_0033_pile_transfer.md`. No complete-circuit or matched-effect simplicity promotion.

### 2026-09-18 — Full-strength regional removal and block7 city-source fold
`CITY_FULL_STRENGTH_V1` passes all8gates (2.35%effecterror,30.46%attenuation,16nulls); opened matched-suffix `CITY_FULL_QUADRATIC_V1` supports retaining the quadratic term for accuracy (full2.35%,secant7.30%,linear13.60%; all broad screens pass). `CITY_SOURCE7_V1` captures four upstream city-state sources; 80ordered K1*K2*V terms replay nativewrite within1.03e-6relative. `CITY_SOURCE7_CONTRAST_V1_CPU_RESULT.json`: MLP7-present paired aligned.3068/norm.3228; attention7-present aligned.1432/norm.1644, overlapping groups. Native query/normalization dependencies remain. Source attribution nominates a source-specific edit, not module-removal equivalence or port closure. for_logan/0033 remains canonical; extraction/composition limitations unchanged.

### 2026-09-18 00:56 UTC — Regional source7 evidence and native MLP7 reader handoff

[Source-group edit](../polynomial_causal/CITY_SOURCE7_EDIT_V1_RESULT.json) passes its opened selective screen; [random-split specificity](../polynomial_causal/CITY_SOURCE7_SPLIT_NULL_V1_RESULT.json) **fails** (interaction .345711 vs random median .244758, beats 0/16). Keep the coupled response. Primary-owned [MLP7 three-reader fold](../polynomial_causal/CITY_MLP7_READERS_V1_PREREGISTRATION.md) has [synthetic CPU identity](../polynomial_causal/CITY_MLP7_READERS_V1_CPU_RESULT.json), 12,386,688 vs 16,368,768 floats; native replay pending at review cutoff, no port-closure or fresh causal claim. Prior-art context: [MLP7/8/9 dossier](../polynomial_causal/explanations/MLP8_MLP9_MLP12_CURRENT_UNDERSTANDING.md), [attention-middle dossier](modules/attn-middle-pooling.md), [circuit registry](CIRCUIT_REGISTRY.md), [module records](circuits/MODULE_DOSSIERS.md). [Hourly assessment](../polynomial_causal/HOURLY_STRATEGIC_REVIEW_2026-09-18_0056.md) records boundary, full source terms, limitations and track handoffs. This supersedes pending random-split language only; historical failures remain.

### 2026-09-18 — MLP7 reader input expansion (CPU; native checks pending)
`CITY_MLP7_INPUT_TERMS_V1_RESULT.json`: nine ordered residual6/initial7/attention7 products reproduce the384cityreaders; maximum nativewriteerror1.06e-6. Self-only omission causes47–48%pairedreadererror and10.90%aggregatecitywriteerror. AllfourCPU predictionspass. Residual6 source recoveredby subtraction, queries/keyRMS/mixed8RMS native; no causal omission or portclosure claim. New city_mlp7_integrated_v2 shared reader-value helper matches frozenV1 exactly; V1 andbindings preserved. Native reader andinstalled checks remain queued behind liveatlas. Canonical report for_logan/research_update_2026-09-18_0058_mlp7_readers.md.

### 2026-09-18 — Normalization information limit of three-reader MLP7 fold
`CITY_MLP7_NORM_INFORMATION_V1_MATH.md` andCPUresult: a hidden direction in ker(WD) has ||Dv||²=7.8413, so complete output norm does not universally factor through the384readings alone. Scope arbitraryhiddenh, reachability by native z/text unproved. Do not treat12.39Mreader storage as full-executor savings: retainingD/b for an exact localnorm costs17.70Mtotal versus16.37Munfolded. DirectGram21.23M alone. This is a priced simple alternative, not universal algorithmic lowerbound. Native queries/RMS/context stay explicit; GPUcertificates stillqueued.

### 2026-09-18 — Earlier residual6 city-reader export, CPU certified
`extracted_circuits/city_reader_residual6_v1` generates all9attention7heads and foldedMLP7cityreaders from residual6prefix+tokens, retaining nativehead8queries andmixed8cityRMS. `CITY_EARLIER_BOUNDARY_CPU_V1_RESULT.json` all3pass,80sequence-equivalent fullforwards21.027s; targeteffecterror5.37e-6,controls<=7.33e-5 versusGPUreference. `CITY_READER_RESIDUAL6_V1_ISOLATED_RESULT.json` 40fixturestandalonepasses.21,851,526FP32values;23,169nativeinputscalars. Earlierdependency, higherweight/stateprice; not totalcompression, token-only, newfresh or independentcomposition. NativeMLP8/suffix external; olderGPUcertificates remainqueued. Latestfor_logan index points to newboundaryreport.

### 2026-09-18 — CITY_RESIDUAL6_SINGLE_INPUT_V1: one-array approximate extraction
Full attention7 at every position plus five folded MLP7 readers now generate complete head8.2 city removal from residual6 + tokens alone. No supplied query or RMS; mixed8 RMS is approximated with other-source norm, all numerator terms retained. Opened CPU full-suffix screen passes a–e: effect error.012430, controls<=.079730,102/111positive,16/16nulls beaten (38.42×median). Isolated40fixture generator replay passes.23,326,342FP32weights,36,864native floats atT32: larger than previous interface. Extraction and opened selectivity advance; fresh/OOD pending, independent composition remains failed0/16random splits, matched-effect simplicity untested. Native blocks0–6 and MLP8/suffix remain external.
Report: ../polynomial_causal/explanations/for_logan/research_update_2026-09-18_0143_single_residual6.md. Program: ../polynomial_causal/extracted_circuits/city_residual6_single_input_v1/manifest.json. Installed receipt: ../polynomial_causal/CITY_RESIDUAL6_SINGLE_INPUT_CPU_V1_RESULT.json. Fresh continuation: CITY_RESIDUAL6_SINGLE_INPUT_FRESH_V1 rows/tables frozen outcome-blind; no fresh behavioral result yet. Older queued GPU checks retain their original boundary/scope.

### 2026-09-18 — Fresh residual6 conditional prediction/selectivity passes
CITY_RESIDUAL6_SINGLE_INPUT_FRESH_V1: all5registered gates pass on20new Pile documents/40sequences/240fixed probes; native capture+screen800CPU sequence-equivalents in50.068s. Effecterror.019047, untouched.011895/substituted.019853; controlratios<=.181133;114/120positive, mean.273865;16/16same-site nulls36.04xmedian. One document(context10) reverses all6contrasts in both native and generated interventions; retained. Fresh isolated40fixture execution exact, using the386-token weight-derived attention7 table; all other programweights/executable unchanged. Declared native residual6 input and MLP8/suffix remain external. Fresh same-corpus evidence, not domain-general OOD or token-only execution. Independent composition failed0/16random splits; matched-effect simplicity absent. Fresh price23,303,302FP32values and36,864nativeinputfloats.
Report: ../polynomial_causal/explanations/for_logan/research_update_2026-09-18_0152_fresh_residual6.md; receipt: ../polynomial_causal/CITY_RESIDUAL6_SINGLE_INPUT_FRESH_V1_RESULT.json. Older CITY_MLP7_READERS_V1, CITY_MLP7_INTEGRATED_V1 and CITY_MLP7_GENERATED_NORM_EDIT_V1 GPU gates now all pass at their original boundaries; no scope borrowing. Next registered CITY_ATTENTION7_OMISSION_V1 local preflight passes; full/no/leave-one-head writes ready, no causal subset yet selected.

### 2026-09-18T01:55:50.775452+00:00 — Scheduled CIRCUIT review: completed certificates and fresh residual6 evidence

[Native MLP7 readers](../polynomial_causal/CITY_MLP7_READERS_V1_RESULT.json), [installed reader replay](../polynomial_causal/CITY_MLP7_INTEGRATED_V1_RESULT.json), and [generated-normalizer edit](../polynomial_causal/CITY_MLP7_GENERATED_NORM_EDIT_V1_RESULT.json) now pass their opened gates; earlier queued wording is historical. [Fresh one-input edit](../polynomial_causal/CITY_RESIDUAL6_SINGLE_INPUT_FRESH_V1_RESULT.json) passes on20distinct documents (40sequences/240fixed probes): effecterror.019047,114/120positive,collateral<=.181133,16nulls beaten. [Overlap audit](../polynomial_causal/CITY_RESIDUAL6_SINGLE_INPUT_FRESH_V1_ROW_AUDIT.json) and [isolated replay](../polynomial_causal/CITY_RESIDUAL6_SINGLE_INPUT_FRESH_V1_ISOLATED_RESULT.json) pass. One residual6 array and native suffix remain; mixed8 RMS approximate; fresh tablevariant23,303,302FP32values. [Source composition specificity](../polynomial_causal/CITY_SOURCE7_SPLIT_NULL_V1_RESULT.json) remains failed0/16; no five-property completion or matched-effect simplicity claim. Links: [circuit registry](CIRCUIT_REGISTRY.md), [graph inventory](../polynomial_causal/CIRCUIT_GRAPH_REGISTRY_V1.md), [module records](circuits/MODULE_DOSSIERS.md), [MLP7 addendum](../polynomial_causal/explanations/MLP7_REGIONAL_PATH_ADDENDUM_2026-09-17.md), [attention-middle](modules/attn-middle-pooling.md), [review](../polynomial_causal/HOURLY_STRATEGIC_REVIEW_2026-09-18_0155.md). Primary implementation and omission screen remain owned; circuit handoff is fresh frozen source-group removal.

### 2026-09-18 — Source-group failure and folding handoff
Fresh MLP7-present generator predicts effects but failsdirection80%vs90%; self/mixed split also fails to recover consistent selfdirection65%vs90%. See CIRCUIT_REGISTRY.md freshfailure entry and ../polynomial_causal/explanations/for_logan/research_update_2026-09-18_0208_source_direction_failure.md. Keep fullcoupledreader path. CITY_ATTENTION7_OMISSION_V1 selected drophead3 onopeneddata (error.013346, potential884736float saving); noattention7 fails.40348>.05. Selectedsubset is notpacked/freshcertified; preservefornextfoldinghour. CurrentCIRCUITstep is registeredfinite-strength source diagnostic.

### 2026-09-18 — Complete city field interface for interchange
city_residual6_fields_v1.py factors the frozen full9head/fivereader generator into Q1/K1/Q2/K2/V. Self-removal exactlyreplays existinggenerator; nativefieldsreplaycapturedremoval<=1e-4. CITY_FULL_INTERCHANGE_V1 predicts donorcitywrite-recipientcitywrite withrecipientqueries, two logicalnativecontexts/sharedweights; freshrows/tableprepared, behaviorpending. Sourcepartitions remainfailed; see CIRCUIT_REGISTRY.md and ../polynomial_causal/explanations/for_logan/research_update_2026-09-18_0220_conditional_source_roles.md. Eight-head omission retainedfornextfoldinghour, notpromoted.

### 2026-09-18 — Two-prefix coupled city interchange export
extracted_circuits/city_interchange_prefix_v1 needs recipientresidual6 throughlastdestination and donorresidual6 throughcity, plusIDs/city/mask. Full9headattention7+fiveMLP7readerweights shared acrosscontexts. Native-statecount50,688 at31+13tokens; static23,300,998FP32values. Freshinterchange + openedprefixreplay + isolatedexecutionpass. No suppliedquery/RMS; approximate mixed8RMS; nativeprefix/suffixexternal. See CIRCUIT_REGISTRY.md and ../polynomial_causal/explanations/for_logan/research_update_2026-09-18_0235_city_interchange.md. Compositionstillunresolved; next5arm key/valueexchange testretainsheadcross.

### 2026-09-18 — CITY_INTERCHANGE_COMPOSITION_V1: coupled operator retained
Opened complete-city key/value interchange screen passes replay but fails both composition gates: joint interaction/smaller=2.3448235>.35; headcross omission/joint=.4186297>.10; suffix interaction/smaller=1.0746265>.35. Both singles live. All20 document joint-interaction ratios fail. Fresh full-swap prediction/selectivity and two-prefix extraction remain correctly scoped; no independent composition or matched-effect simplicity promotion. Primary receipt: `basis_aligned/polynomial_causal/CITY_INTERCHANGE_COMPOSITION_V1_RESULT.json`; explanation: `basis_aligned/polynomial_causal/explanations/for_logan/research_update_2026-09-18_0245_city_composition.md`. Subsequent opened signed accounting finds partial cancellation (cosine−.67844); no new partition selected.

### 2026-09-18 — Packed attention7 drop3 fresh city-removal confirmation
CITY_ATTENTION7_DROP3_FRESH_V1 all5gatesPASS on20freshPile documents/40sequences/240probes: error.0188112<=.05;114/114capable positive;controls<=.123827;16/16nulls,30.891xmedian. Physically packed eight-head prefix, fiveMLP7readers and head8 factors:22,418,566FP32 values on386tokens, saving884,736 versus equal-vocabulary nine-head generator. Isolated40fixture replay exact. Native residual6 plus native suffix remain external; composition and matched-effect simplicity unresolved. Package `basis_aligned/polynomial_causal/extracted_circuits/city_attention7_drop3_v1`; primary result `CITY_ATTENTION7_DROP3_FRESH_V1_RESULT.json`; Logan update `research_update_2026-09-18_0252_packed_city_removal.md`. FineWeb transfer registered and row construction started; no outcome assumed.

### 2026-09-18 — FineWeb direction failure and receiving-value fold
CITY_ATTENTION7_DROP3_FINEWEB_V1 passesprediction(.018626),controls/nulls/isolation butFAILSdirection88/102<.90;nativeandgenerated14reversalsagree. CITY_DROP3_FINEWEB_INTERCHANGE_V1 opened swap likewiseFAILSdirection84/102;error.016716passes. Retain both failures, no cross-corpus selective promotion. CITY_FINEWEB_RESPONSE_V1:early/head9sufficiencyFAIL;head9.8 prominent but laterattentionnecessary. CITY_FINEWEB_HEAD9_FOLD_V1 exact7terms: routing-onlyFAIL(.8122/1.1343),deltaV aligned.7588/1.1075;triple small, no independent composition. Next registered MLP8→value-reader six-piece fold implemented with actual-weight algebra control, nativecapturepending. Full report/primary links: `basis_aligned/polynomial_causal/explanations/for_logan/research_update_2026-09-18_0304_fineweb_reversals.md`.

### 2026-09-18 — MLP8→head9.8 value reader: native fold and opened intervention
CITY_FINEWEB_MLP8_VALUE_FOLD_V1 native replayPASS(max2.64e-5), omitMLP8FAIL(2.007/.418),quadratic-negligibilityFAIL(.367/.349). CITY_FINEWEB_VALUE_MEDIATION_V1 all4basicgatesPASS: subtractMLP8 changes reversed target.8815×swap;quadratic.2586/.2656;direct.9649/.5683. This changes onlyhead9.8value, not nativeMLP8. Opened positive94/102 versus84/102, but reversedgroup control/target<=.707 warrants no uniform selectivity claim. Six-term helper11,354,114FP32values; native z8/edit/mixed9norm contexts explicit. Same-boundary16nulls registered; fresh/composition/simplicity unresolved. Report `basis_aligned/polynomial_causal/explanations/for_logan/research_update_2026-09-18_0317_value_mediation.md`; primary receipts have matching stems.

### 2026-09-18 — Fresh-confirmed reduced-interface MLP8 value mediator
CITY_FINEWEB_VALUE_NULL_V1 all3PASS,22.235xnullmedian. CITY_VALUE_MEDIATION_FRESH_V1 all5PASS:prediction7.14e-6;97/100positive;controls<=.139;16nulls21.547xmedian. Parent nativecityswap91/100positive,mean.5299→corrected.4326: not uniformly stronger. No original packed-prefix or wholeMLP8ablation claim. CITY_MLP8_VALUE_EXTRACTED_V1 eliminates unusedmixed9vectors exactly:11,206,658FP32values;nativez8+editedRMS36896scalarsT32,delta36864separate.40local/isolated and80installed-equivalent replaypass. Package `basis_aligned/polynomial_causal/extracted_circuits/city_mlp8_value_mediator_v1`; Logan `research_update_2026-09-18_0326_extracted_mediator.md`. Composition of direct/MLP8value paths registered, no independence or matched-effect simplicity promotion.

### 2026-09-20 — Subject-number conditional suffix, blocks11–17

Four-reader width8 response dictionary; full attention12–17 and sparse18/36 MLP pair cores. Fresh regular target/control transfer and CPU prepared-context extraction pass; native contexts/initial response remain open. Source-write per-cell preservation fails even with dense cores and source-expanded calibration; no five-property promotion. Full dynamic finite readers now pass native closure for calibration. Raw conditional artifact is not a new verified package in the generated graph inventory. [Dossier](../polynomial_causal/JOINT_READER_SPARSE_RESPONSE_2026-09-20.md), [reuse audit](../polynomial_causal/SOURCE_REUSE_CPU_AUDIT_2026-09-20.json), [current update](../polynomial_causal/explanations/for_logan/research_update_2026-09-20_0726_sparse_subject_response.md), [math review](../polynomial_causal/THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-20_0726.md).

- 2026-09-20 update: native factored-attention v679 matches expanded attention (8.88e-16), but costs1.39–2.13x storage on short prompts; retain expanded here. [Baseline](../polynomial_causal/FACTORIZED_ATTENTION_BASELINE_2026-09-20.md). Source oracle v680 rejects final-basis limitation: A/B modal errors<=0.326% with native final states; reduced propagation remains unresolved. Primary receipt `subject_attention_freeze_v680_result.json`.

- Subject v682–685: postattention reset and fused-block controls fail to rescue early-source preservation. Exact native suffix/readout after initial width8 reconstruction still gives A modal6.69%; fixed initial decoder is insufficient. [Diagnostic and kernel audit](../polynomial_causal/SOURCE_REUSE_AND_FULL_READERS_2026-09-20.md). No all-width8 lower bound.

- Subject v686–688 [hybrid output predictor](../polynomial_causal/SUBJECT_HYBRID_RESPONSE_2026-09-20.md): sparse nonlinear number plus baseline-gradient modal branches passes fresh48-row A/B prediction/capability/collateral gates. IndependentCPU replay2.08e-15 on24sourcecases. Native gradient/context/fullsource inputs remain external; no residual-state replacement or fullmodel adoption. Prior eight-state source-reuse failures remain valid.

- Subject v689–691 [selected modal derivative path](../polynomial_causal/SUBJECT_MODAL_DERIVATIVE_PATH_2026-09-20.md): retain attention12 and all six token-local MLP derivatives. Prospective48-row longer-combination prediction/capability/collateral gates pass with uncorrected sparse number branch. Earlier simple path cuts fail; later attention derivatives omitted only in modal predictor. Number branch/native context generators still charged; no primal ablation/adoption claim.

- Subject v692–695 [signed and matched controls](../polynomial_causal/SUBJECT_SIGNED_AND_MATCHED_CONTROLS_2026-09-20.md): negative-B/difference number prediction fails despite dense pairs and paired-block native composition. Post11 matched-random native specificity19.6/20.2x passes; prediction bars use source-effect denominator, random-relative errors remain55–60%. Semantic attractor-control rows prepared, native execution pending.

- Subject v696–699 [Semantic noun controls and composition](../polynomial_causal/SUBJECT_SEMANTIC_COMPOSITION_2026-09-20.md): opposite-number magnitude specificity fails; paired congruence sign flip passes. Opposite-number post11 composition passes budget gates; congruent composition fails number/modal fidelity despite successful interaction predictions. Serialization-invalid v698 preserved, repaired v698r1 independently audited. Native dependencies and missing HT comparison remain.

- [Semantic source ports and finite pair graph](../polynomial_causal/SEMANTIC_SOURCE_PORT_CENSUS_2026-09-20.md): direct recurrence closes numerically but fails A sufficiency; early/middle writes pass signed-role screen. B all-pair effect reconstruction1.90% opened/1.99% new constructions; frozen two-pair prediction fails18.68%, and native capability failures prevent full transfer promotion. Full native computation charged; no compressed execution claim.

- [Semantic pair origin across block11](../polynomial_causal/SEMANTIC_PAIR_BOUNDARY_2026-09-20.md): replay passes; both boundary-only and suffix-only dominance fail. Output reconstruction needs both contributions (13.32%/27.98% errors separately versus1.98% together). Next folded observable must retain transported local response and downstream curvature; no native capability repair or standalone extraction.

- [Conditional three-source quadratic observable](../polynomial_causal/SEMANTIC_SOURCE_QUADRATIC_JET_2026-09-20.md): validated native derivatives predict unit/negative/mixed effects below10%, doubled edits fail37.16%. Signed rank-one baseline also passes first three arms but directions vary across contexts.192-context CPU coefficient extraction passes; native coefficient generation remains required. v1 precision-invalid and v2 missing-bias-invalid preserved.

- [Shared quadratic and coverage correction](../polynomial_causal/SHARED_SOURCE_QUADRATIC_DICTIONARY_2026-09-20.md): original amplitude designrank2/6; six independent native settings validate fullquadratic4.72% but reject per-context rank1 15.74%. Sharedrank2 dictionaries also fail. Preserve restricted-arm successes; no shared semantic circuit identification.

- [Four-output source observable and selectivity](../polynomial_causal/SEMANTIC_SOURCE_MULTIOBSERVABLE_2026-09-20.md): native replay/prediction pass (number9.10%,modal2.82%), native selectivity fails129/288cells. Local three-source leakage bound and clipped candidate expose a restricted-interface limitation; no global impossibility or circuit adoption.

- [Five-source null edit](../polynomial_causal/FIVE_SOURCE_MODAL_NULL_2026-09-20.md): native collateral below4.39%, but target-strength/selection conjunction passes only1/32cells. [Exact normalized MLP source core](../polynomial_causal/NORMALIZED_MLP_SOURCE_CORE_2026-09-20.md): weight-contracted numerator/shared RMS denominator and analytic local Hessian pass planted CPU replay/gauge/bias controls; captured native-context validation pending.

- [Native normalized MLP source-core validation](../polynomial_causal/NORMALIZED_MLP_SOURCE_CORE_2026-09-20.md): exact local output/derivative andCPUexport replay pass; localMLP11curvature alone fails number53.11%/modal8.16%. Full-path source-Jacobian/adjoint capture implemented; planted chain-rule proof passes, native all-node closure pending.

- [Native source-Hessian decomposition](../polynomial_causal/NATIVE_SOURCE_CURVATURE_DECOMPOSITION_2026-09-20.md): fifteen local terms close1.25e-15 relative. Broad curvature omissions fail; full fifteen-direction finite test passes4.70%number/1.39%modal. Early-layer-only A/B cross-curvature candidate passes6.82/2.00% on openeddata; material interaction errors canreach41.69%. Fresh validation and native-generator reduction remain pending.


20 September: [Exact source attention fold and spectral baseline](../polynomial_causal/ATTENTION_FOLD_AND_SPECTRAL_BASELINE_2026-09-20.md). Native attention fold replays exactly but fails the one-shot speed gate (1.024x). Per-context signed rank-two Hessians pass opened finite-edit tests with68 versus80 stored values; native generators remain required. Rank-one fails. No native HT or complete-circuit claim.

20 September: [Shared five-source features](../polynomial_causal/SHARED_FIVE_SOURCE_FEATURES_2026-09-20.md) fail the reuse screen: common rank2 plane17.46% heldout number error; per-output planes do not rescue it; three calibrated sparse residual pairs improve to11.86% but still fail. Full-basis recovery is exact. Opened data only.

20 September: [Dictionary optimization redteam](../polynomial_causal/SHARED_DICTIONARY_OPTIMIZATION_REDTEAM_2026-09-20.md). Joint analytic coefficient fits still fail, but native-outcome minimax oracle fits number effects in all32cells (worst7.49%, gap<3.3e-8). Therefore fixed-dictionary capacity is not ruled out. Oracle uses labels and does not establish prediction or modal preservation.

20 September: [Derivative-only minimax](../polynomial_causal/DERIVATIVE_ONLY_MINIMAX_2026-09-20.md) passes opened construction-heldout number9.68%/modal0.89%, but calibrationnumber12.19% fails the overall gate. Fullquadratic8.94% error combines with6.04% compression error. Native generators and group-dependent fitting remain limitations.

20 September: [Per-input minimax extraction check](../polynomial_causal/SINGLETON_MINIMAX_2026-09-20.md) fails heldout number37.22%; retaining group budgets reduces it to10.104% but still fails. True singleton coefficients pass batch permutation/peer-removal checks exactly. Grouped partial success is not standalone extraction.

20 September: [Native two-MLP quartic/HT baseline](../polynomial_causal/NATIVE_TWO_MLP_QUARTIC_HT_2026-09-20.md). Exact homogeneous numerator fold6.59e-15; rank8 symmetric tree9.35% error but656values loses to exact canonical280. Rank2 costs116 but fails36%coefficient error. This is a named direct polynomial branch, not the normalized full model or causal extraction.

20 September: [Matched-strength modal-null control](../polynomial_causal/MATCHED_STRENGTH_NULL_2026-09-20.md). All-cell strength matching fails22/32pass; all22 matched cells show>2x collateral reduction, but original target-strength retention failure remains. Conditional reduced-strength selectivity evidence only.

20 September: [Shared selective directions](../polynomial_causal/SHARED_SELECTIVE_DIRECTIONS_2026-09-20.md) have zero feasible uniform linear retention across calibration; small conflict witnesses and planted controls pass. Singular-subject fixedall5 exception has native1.55–1.68x target strength but fails modal10.422%; baseline andcandidate bothpass7/8primarycells. No general selective circuit.

20 September: [Prospective source-response transfer](../polynomial_causal/SOURCE_OOD_TRANSFER_2026-09-20.md) passes full19-arm prediction on48newtexts: number6.409%/modal2.129%, all16nativecapabilitycells100%. Null strengthretention stillfails15/16cells. Posthoc fixed-rule spectral2passes7.527/3.568%; tangentfails61.47/13.41%. Fresh native derivative generators remain required.

20 September: [Budgeted modal selectivity](../polynomial_causal/BUDGETED_MODAL_SELECTIVITY_2026-09-20.md) improves joint native passes from2/48exactnull to35/48 at5%gradientbudget; all13remaining failures arestrength, no nativecollateral failures. This redteams unnecessary exactnull constraints but stillfails all-cell adoption andrequires native gradients.
