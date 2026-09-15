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
| PATH-SUBJECT-001 | MLP8 input/context forward to L11H3 subject-number write and its downstream readers | Native-axis coordinate, E/A/U/W background coordinate, and their bilinear interaction | Proposed and preregistered; prior native scalar-only readout is a recorded null | Run the frozen upstream-context discovery on its next `CIRCUIT` hour, then use any held interaction to define a narrower folded path |
| PATH-SET2-001 | Selected outputs backward through MLP17 to head17.2 and its retained three-contraction producer | Both QK factors, current/inherited values, shared raw-state corners, MLP17 mixed terms | Exact synthetic identities and regional approximate screens exist; native causal identification remains incomplete | On a `WEIGHT_FOLDING` hour, expand the selected MLP17 reader by named residual sources and test only the terms that reach the chosen output block |
| PATH-SET1-001 | MLP9 forward through attention10 into MLP10 and its larger parent | QK/value producer terms and downstream bilinear self/cross terms | Approximate sparse parent screens exist; individual and sign exceptions remain | Use circuit-localized reader/writer evidence to select terms before another folded approximation |
| PATH-UNEMBED-001 | Token or token-family unembedding rows backward through final RMS, MLP17, and earlier contributors | Shared unembedding components, token-specific remainder, MLP17 bilinear terms, selected earlier residual sources | Proposed path family with partial setting3 fits; no adopted full path | Choose one semantic logit contrast and derive its exact backward factors before fitting |
| PATH-EMBED-001 | Embedding forward through selected early decoders to a named intermediate reader | Token/position embedding pieces and selected early attention/MLP self/cross terms | Unopened under the alternating-track directive | Start only from a circuit with a named downstream decision point and selective intervention |

## Primary pointers

- [Joint interaction-path proposal](../polynomial_causal/explanations/for_logan/interaction_path_decomposition_proposal_2026-09-11.md)
- [Setting2 sparse regional math](../polynomial_causal/INTERACTION_SPARSE_REGIONAL_V1_MATH.md)
- [Head17 output-block fit math](../polynomial_causal/HEAD17_OUTPUT_BLOCK_FIT_V1_MATH.md)
- [Subject upstream-context preregistration](../polynomial_causal/SUBJECT_NUMBER_UPSTREAM_CONTEXT_COORDINATE_DISCOVERY_V1_PREREGISTRATION.md)
- [MLP dossier index](../polynomial_causal/explanations/MLP_MODULE_DOSSIER_INDEX.md)

Update this index only when a path's evidence label, scope, live handoff, or primary
dossier link changes. Put experiment detail in the path dossier or primary receipt.
