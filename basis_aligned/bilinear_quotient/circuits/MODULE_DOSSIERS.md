# Module dossiers

These records collect stable facts about native components independently of any one behavior
circuit. They complement the task-defined records in `DOSSIER.md`. A native module boundary is an
index for retrieving evidence, not an assumption that the module is one semantic unit.

## `module.attention.5`

Aliases: attention block 5, `attn5`, L5 attention, induction gate, copy gate, content gatherer,
pooler. Related but narrower object: head 5.7, also called the sink or constant-write head.

### Established before 2026-09-04

- Sections 877 and 882: removing attention block 5 removes nearly all measured induction; the
  front circuit was summarized as attention 0 supplying the key and attention 5 performing copy.
- Sections 998, 1006, and 1007: the content-gathering effect is concentrated in layers 3–5 and is
  dominated at head level by layer-5 head 7, with a jointly important distributed remainder.
- Sections 1039, 1043, 1044, and 1047: attention 5 operates on the value residual and belongs to
  the broad/local-residual routing part of the attention map; simple token and embedding-bag
  stand-ins do not reproduce it.
- Legacy registry entry `Sink (5.7)`: head 5.7 can be replaced by one constant vector without the
  measured loss cost of deletion.

### 2026-09-04 extension

The whole block, not only head 5.7, has a nearly one-dimensional output geometry on the measured
corpora: 98.1% of write energy lies in one direction; independently fitted natural-text directions
have absolute cosine 1.000; the code direction has cosine 0.997; and a fixed-vector replacement
recovers about 95% of the whole-block value. This is a held-out geometric refinement of an existing
module account, not a new functional localization.

### Still unknown

- which input-dependent computation produces the nearly fixed whole-block write;
- how the induction/copy and content-routing functions split below the native head basis;
- whether one intervention-defined subspace can selectively change either function while preserving
  the other; and
- which downstream computations treat parts of this write as the same variable.

Any future attention-5 experiment must cite this dossier and state which unknown it resolves.

## Saved correlative interface: 26 heads across layers 3–16

10 September update: [weight pullback and input-reader overlap](../../polynomial_causal/explanations/2026-09-10/attention_ov_input_reader_overlap.md). The fixed block projector and complement use `O_P,h=(Oq)q_h^T` and `O_R,h=O_h-O_P,h`, respectively. Their within-head QK1/QK2 routing is identical by construction. At relative rank tolerance1e-6, 19 per-head remainder writers retain all128 coordinates and can read the saved scalar’s full value-input function. Seven single-head blocks retain127 coordinates with partial input overlap. Full-block cross-head cancellations prevent interpreting these per-head overlaps as globally duplicated computation.

The earlier donor double dissociation supports a swap interface, while selective mean removal and endpoint-addition tests failed. Neither the broad complement nor an OV row-space overlap is a named semantic circuit. Check specialist-heads.md, attn-middle-pooling.md, channels.md and the linked native/CPU receipts before claiming new per-head functionality.

### 15:17 UTC: QK1/QK2/value dependencies

[The complete factor lattice](../../polynomial_causal/explanations/2026-09-10/attention_qk1_qk2_value_dependencies.md) extends the earlier combined-routing test to both branches. Full swaps and prior P route/value cells replay. Opposite global stored-score-half dependence failed: QK1-minus-QK2 conditional recovery losses .00637/.00118/.01895 on P-A1/P-A2/R-C, below.20 and not opposite. Conditional value losses .81049/.78021/.89233 pass. Current values contain earlier routing computation, so this does not establish routing irrelevance.

R-C QK1+value has9.51% full-vocabulary error versus19.47% for QK2+value, a descriptive candidate only; no independent promotion or threshold rescue. Higher-order endpoint interaction is .3956/.3343/.3638. Stored QK labels are independently exchangeable per head. The broad complement remains unexplained and original selective-removal failure unchanged. Canonical subroutine.correlative.score_half_task_split is rejected with valid instrument/value-dependence evidence retained.
## Gerund readout scalar, 10 September 2026

The [MLP17 dossier](../../polynomial_causal/explanations/MLP17_CURRENT_UNDERSTANDING.md) now includes `subroutine.readout.gerund_shared_scalar`: a direction from eight unembedding bare/-ing contrasts, with sixteen distinct test verbs. Final-state scalar cue transfer .586/.621 and cross-verb transfer .584/.620 are partial and fail the registered .80 sufficiency bar. Zero removal target CE+.565/+.686 versus unrelated C meanabsCE .0232 passes the narrow registered removal test. MLP-only scalar contributes .071/.100 recovery; most scalar change arrives from earlier state. No independently extracted producer or broad OOD/composition evidence. See [the computation and controls](../../polynomial_causal/explanations/2026-09-10/shared_gerund_component_from_unembedding.md); do not repeat it as discovery of generic quadratic folding or promote it to a sufficient grammar circuit.

The [distributed output-port follow-up](../../polynomial_causal/explanations/2026-09-10/gerund_scalar_writes_and_live_feedback.md) at16:06 held joint-swap sufficiency (.936/1.017) and narrow zero-removal selectivity (C absCE.0433), while rejecting scalar-only edit prediction (full-vocabulary errors .654/.670). Both claims are separately recorded under `subroutine.readout.gerund_scalar_write_network`. All36 output e-coordinates prescribed to donor guarantee the final scalar, but native complementary state changes materially. Weighted native MLP contributions84–87% are an attribution identity, not an independent producer or individual-module circuit. All native weights remain.


### Fresh grammatical transfer and agreement failure, 10 September 16:31 UTC

The fixed all36-port interface transfers to new16 verbs and might/were or would/was cues: recovery .95175/.86341, paired intervals [.92557,.97813]/[.83145,.89820]. All96 native pairs capable; old R replay and instrument pass. Broader preservation/removal fail on closer he/they runs/run agreement G: swap recovery .13342 and meanabsCE .10793, zero meanabsCE .54788 (CI .51396–.58346), versus .10 limits. G is one agreement contrast over16 contexts, not16 agreement readouts; it remains a failed control. Original C-only selectivity and failed scalar prediction are preserved. No independent producer, single-module promotion, pretraining-OOD or saving. The updated gerund_scalar_writes_and_live_feedback.md explains both individual-token and shared-structure-plus-remainder folds. Canonical distributed-port revision records positive fresh transfer and failed broader selectivity separately.


### Grammatical scalar consumers, 10 September 16:42 UTC

Fixed all18 post-RMS MLP e reads are held at natural base values before the original all36 output e edits. Exact weight-folded cross/square correction, no refit. Target swap recovery falls .95175->.66036 and .86341->.57286: losses .29139/.29055 with paired intervals [.25625,.32386]/[.26203,.31871]. Base-only G zero CE falls .38682->.11766 (69.6% reduction), not directly comparable with the previous two-endpoint .54788. Native scalar prediction errors .64876/.64467 become .15287/.08449 on the altered blocked computation; joint <=.10 closure still fails. A/B/C/E held D failed. All64 native pairs capable, exact replay, folded relative bridge <=1.60e-6,28forwards448seq in1.7175seconds. Folded context readers k_v=2Q(v)e for e and runs/run have median cosine .2053 across layers; descriptive immediate-output maps, no site selection or semantic separation proof. Original v184/v185 algebra reused; new causal consumer evidence under subroutine.readout.gerund_mlp_scalar_consumers. No independent context producer, OOD or selective repair. See gerund_scalar_writes_and_live_feedback.md and GERUND_MLP_CONSUMER_V1_RESULT.json.


### Selected-reader response state, mathematical review16:49

The exact MLP17 finite-difference contraction compiles two context rows K and squared coefficients a for e and runs/run. With q=Ku, response=delta*q+delta^2*a and q_next=q+2delta*a. Trained FP64 selected-response error1.01e-13 and sequential-composition error1.59e-13; saved2306-coefficient program reload also passes without native checkpoint. This is local conditional response extraction, not absolute outputs or a text-to-answer producer. All16 synthetic equal-e/equal-K/equal-norm reflection pairs still have different full-output responses; any common prediction has minimum relative error .1087–.2416. No natural-text reachability/OOD claim. Canonical gerund_selected_response is specified for selected local operation, rejected for full-output closure; zero native forwards. Review THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-10_1649.md proves the kernel criterion and state update, compares exact reduction methods, and keeps initial contextual producers/downstream readers as the missing work. LATEST16:54 and existing gerund explanation carry this scoped result.
