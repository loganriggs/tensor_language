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

## Fixed-position field counterfactuals — completed 14:32 UTC

`CONTEXTUAL_HISTORY_FIELDS_V1_RESULT.json` preserves a valid failure of the stronger
registered key/payload and retargeting criteria (A true, B/C/D false). The source,
current query and nonmatching control positions are fixed across all arms. Changing
either stored key field largely removes the original answer; payload-only changes
raise the foil probability to .788/.808 for IID hops2/3 and .751/.701 on short-cycle
OOD. Jointly retargeting stored/current keys and payload gives .839/.784 IID and
.798/.732 OOD. Several groups miss the required .80, so this is suggestive composite
key/value behavior, not a passing manipulable symbolic dictionary. The mostly-zero
foil token follows the fixed first-eligible-token protocol and limits generality.

The additional one-native-factor-per-field screen also fails: zero IID heads meet
its conjunction. First factors in heads0/3 are relatively insensitive to entity
changes (RMS effect .058–.082), while hop changes cause 1.04–1.08; their second factors
are mixed. This licenses a narrower falsifiable candidate, not semantic labeling
from sensitivity alone: `HOP_STATE_COMPILER_V1_PREREGISTRATION.md` substitutes one
causal role/hop state codebook into these first-factor projections. It must preserve
all29-way distributions and single/joint native-corresponding removals on fresh text.

## Shared hop-state replacement — rejected

The fixed30-state causal role/most-recent-hop compiler passed its numerical control
and reduced charged constants387968→375424, but failed full-output and removal tests.
Query KL was2.465 IID,2.514 short-cycle OOD and1.485 short-history OOD; joint-removal
relative errors .936/.961/.927. These savings do not establish interpretability.
`HOP_STATE_COMPILER_V1_RESULT.json` preserves A/D true, B/C false and the frozen books.

A CPU diagnostic on the same already opened panels replaces Q1 alone, K1 alone, or
both, without fitting anything. Neither side passes: query-only KL .822–1.573 and
key-only1.497–2.624. The Q/K output interaction has49–56% of the joint-error norm.
This rejects both a one-sided rescue and independent-error accounting. Receipt:
`HOP_STATE_FACTORIAL_V1_RESULT.json`; this is diagnostic reuse, not fresh confirmation.
The next object is the actual upstream producer dependency, not another semantic
state vocabulary inferred from a sensitivity screen.

## Which upstream producer supplies the final retrieval factors?

`UPSTREAM_PRODUCER_FACTOR_V1_RESULT.json` is a valid15-path/3-joint atlas on128 fresh
documents, with all native projection/full-forward reassembly within1.25e-14. The
MLP contribution dominates: removing its K1/K2/V inputs loses .859–.957 repeated
higher-hop gold probability and .567–.671 novel-hop1 probability. In comparison,
removing direct embedding/first-attention inputs usually changes these probabilities
by less than .03. Thus the MLP supplies both lookup uses, not a selective single
reader path. Zero pairs pass the registered target/control criteria; A/D true,B/C false.

These are normalized reader-edge cuts: the original final RMS gain is still computed
from the complete live residual. They are not upstream MLP removals. All4 heads and
positions were included without selection. Joint all-reader effects cannot be added
from singleton effects: relative errors .226–.228 for embedding,1.218–1.221 for
attention and2.627–2.638 for MLP. The complete coupled product must remain.

The successor HOP_MLP_TOKEN_CONTEXT_V1 tests the MLP's known token-token/mixed/
context-context terms on fresh lookup documents, with actual recomputation of the
final normalization. It reuses the algebra documented in the older
MLP0_TOKEN_CONTEXT_TENSOR_FACTORIAL_FINDINGS.md; no rediscovery of that identity is
claimed. The new issue is whether mixed token/context dependence is necessary and
sufficient for both source-defined lookup uses. All original coefficients count.

## Closure of the weak-checkpoint simplification branch

HOP_MLP_TOKEN_CONTEXT_V1 completed validly (A/D true,B/C false). The mixed term
alone has query KL2.731/2.733 on IID/OOD and near-chance lookup probability. Mixed
plus context-context retains .886/.834 repeated higher-hop gold probability, but
only .259/.236 novel-hop1 probability; full terms give .999/.938 and .737/.631.
All8 candidate/native term interventions replay within3.09e-12 with downstream
normalization recomputed. The three terms cooperate; the shared mixed-term-only
hypothesis is closed without threshold or branch tuning. All400640 parameters remain.

This closes the current weak-model simplification branch, not the overall goal.
The existing equally sized four-attention checkpoint passes fresh cold and unique
multi-hop capability gates. Reconstruction now moves to that actual composition
setting; see [cold_query_composition_reconstruction.md](cold_query_composition_reconstruction.md).
The earlier source-history result and all failed stronger claims remain preserved.
