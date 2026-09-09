# The small model reuses earlier answers for longer-hop queries

The existing `attn-mlp-attn-rms-seed0` model's roughly 25% longer-hop accuracy is
largely explained by repeating answers already present in its context. This is a
causally tested source mechanism, with an executable full-output decomposition.
The contextual key and value computations still use the original learned weights.

Each document defines a function through24 initial bindings, followed by48 queries
`[Q, entity, hop-count, answer]`. A repeated query has the same entity and hop count
as an earlier query. The answer to that earlier query is already an input token.

| Population, hops2–3 | Repeated-query accuracy | Novel-query accuracy |
|---|---:|---:|
| Fresh random24-cycles |100.0% (615 queries)|6.28% (2438 queries)|
| Short-cycle functions, OOD |94.85% (680 queries)|5.41% (2494 queries)|
| Unique-query documents, OOD |No repeats|6.63% (3094 queries)|

This supports the registered repetition hypothesis. It does not establish that all
successful longer-hop predictions use history or that another deeper checkpoint
uses the same mechanism. The higher-hop predictions remaining after history removal
are retained in the measurements.

## What was removed

At the final attention layer, query positions were prevented from reading earlier
answer positions with matching `(entity, hop)` keys. All heads were included, and
all original upstream computation remained live. On the615 IID repeated higher-hop
queries, accuracy dropped from100% to6.50%, and mean correct-answer probability
dropped by0.9345. Removing initial matching binding values changed correct-answer
probability by only0.000006 on these same queries.

The control removed the same number of earlier answers with a matching hop count
but a different entity, wherever enough such answers existed. On606 eligible IID
repeated higher-hop queries, the matching-history loss was0.93425 versus0.000043
for the control. Both directions of that contrast were registered before outcomes.
On669 corresponding short-cycle OOD queries the losses were0.87510 and0.000268.

Conversely, initial binding removal mattered for novel hop1 queries: correct-answer
probability fell by0.6721 on IID and0.5867 on short-cycle OOD. The model therefore
uses different context sources for fresh one-step lookup and repeated longer-hop
answers, rather than all these scores measuring the same generic ablation damage.

The controls are matched on count and hop label, not exact age. Matching answers
were about67 tokens old on average versus19 tokens for controls. A fixed-position
key/payload counterfactual is the next test, because it can change key identity
while keeping source position fixed. No claim of an internal discrete equality
gate is established by the source-mask experiment alone.

## The extracted full-output computation

The complete29-way logit vector is partitioned as

`logits = residual + binding + matching_history + other_sources`.

The program starts from token IDs. Its first attention and MLP compute every input
to the final contextual Q/K/V maps; there are no saved teacher activations. Final
W_O is folded into the vocabulary readout and physically removed from the copied
model. Each source contribution is an explicit contraction of the live two score
factors, live value, and folded full-vocabulary reader. The residual and all other
sources remain in the computation and its cost.

On all384 documents, all239 input positions, all29 logits, and five removal arms,
the largest original-versus-extracted error was5.12e-13 in FP64, against1e-9 absolute
and relative tolerance. The native history+binding joint removal obeyed the predicted
full-vector identity to6.25e-13. This composition law follows from the final affine
readout after disjoint edge cuts; it does not establish that arbitrary upstream
replacements compose or that probabilities are additive.

The exported program contains387968 parameters/folded coefficients versus400640
native, plus145920 fixed buffer scalars. The12672-constant saving is ordinary exact
matrix folding and is accounted as compiler infrastructure, not semantic discovery.
Most weights remain opaque. The portable artifact includes those weights and exact
cached positional buffers, so loading it does not require the original checkpoint.
A fresh8-document CPU export/reload check, seed5919, matches all five original
removal arms to2.27e-13.

Artifacts:

- [`CONTEXTUAL_HISTORY_V1_PROGRAM.pt`](../CONTEXTUAL_HISTORY_V1_PROGRAM.pt): self-contained executable weights/buffers.
- [`contextual_history_reference.py`](../contextual_history_reference.py): explicit readout and source parser.
- [`export_contextual_history_v1.py`](../../bilinear_quotient/ops/export_contextual_history_v1.py): loader and fresh replay.
- [`CONTEXTUAL_HISTORY_V1_RESULT.json`](../CONTEXTUAL_HISTORY_V1_RESULT.json): native/compiled distributions, removals, group counts, all four registered predicates true.
- [`CONTEXTUAL_HISTORY_V1_EXPORT_REPLAY.json`](../CONTEXTUAL_HISTORY_V1_EXPORT_REPLAY.json): independent export/reload receipt.

GPU execution took3.61 seconds after entering the experiment; CPU export and fresh
replay took2.22 seconds. These exclude research and authoring time.

## Relation to the four desired properties

OOD prediction, extraction, removal and joint composition are verified for this
explicit source decomposition with its declared native background. Reuse is
supported across hop2 and hop3 queries, but a shorter implementation of the shared
lookup remains unproven. The next target is its inputs and operation: separate the
entity key, hop key and returned payload with fixed-position counterfactuals, then
test whether the two score factors implement those fields separately or jointly.

This is progress on an existing small contextual model. The full bilin18 model and
the overall goal remain unfinished; a compact learned routing program has not yet
replaced the contextual native weights.
