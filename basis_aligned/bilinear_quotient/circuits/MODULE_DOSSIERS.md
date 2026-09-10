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
