# Reconstructing composition without earlier answers

The existing `attn4-rms-seed0` checkpoint is a better target for studying composed
lookup than the weak checkpoint used in the bounded reconstruction pilot. A fresh
CPU capability audit verifies that it can answer multi-hop queries before seeing
any previous answers. Both checkpoints have400640 parameters; this comparison
does not attribute the difference to extra capacity or establish training reliability.

| Fresh population | Four-attention hop2 / hop3 accuracy | Pilot hop2 / hop3 accuracy |
|---|---:|---:|
| IID cycles,16 documents |100% /99.48%|31.58% /25.00%|
| Unique query keys,16 documents |100% /96.20%|7.50% /6.52%|
| Short-cycle functions,16 documents |94.58% /92.00%|27.59% /24.00%|
| First query only,64 worlds,16 per hop |100% /100%|12.50% /6.25%|

The last population contains only24 function bindings and one query. No previous
answer is present. Thus the stronger checkpoint's capability cannot be explained
solely by the repeated-answer mechanism identified in the pilot. Both prospective
capability gates pass, including short-cycle OOD. These finite samples support
capability, not an already identified internal algorithm.

Receipt: [`HOP_COMPOSITION_CAPABILITY_V1_RESULT.json`](../HOP_COMPOSITION_CAPABILITY_V1_RESULT.json).
Executable: [`audit_hop_composition_capability_v1.py`](../../bilinear_quotient/ops/audit_hop_composition_capability_v1.py).
Checkpoint SHA256: `c9388cc8b1ba7e95c6a043cbf4d01592987447867a50781d3babe31ce3f8358b`.
CPU execution took3.94 seconds, excluding authoring. No model was trained or refitted.

## What is and is not known about the computation

The older results_hop.md retracts a clean layer-by-layer story inferred from one
attention trace. Its later linear probes show decodability of intermediate function
powers, but that also does not establish causal pointer advance. We will use exact
full-vocabulary contributions and native-corresponding interventions to distinguish
possible implementations. Attention scores can be signed, and source values may
already carry information computed from other bindings.

The first prospective test partitions the final query's sources into the requested
answer binding, earlier chain bindings, query tokens, and everything else. It asks
whether retaining the answer binding plus query sources suffices for the native
distribution, and whether removing it matters more than removing earlier bindings.
Repeated chain positions in short cycles are deduplicated. Four query forks share
one binding world;512 forks count as128 independent worlds.

The exact final W_O/vocabulary fold and token-to-logit executor are reused from the
history work. All three preceding attention layers and final Q/K/V remain live and
charged. Generic matrix folding is not semantic discovery. This experiment can
identify the source interface of composition; explaining and simplifying its learned
readers, writers and shared updates remains the central unfinished task.

Protocol: [`COLD_COMPOSITION_SOURCE_V1_PREREGISTRATION.md`](../COLD_COMPOSITION_SOURCE_V1_PREREGISTRATION.md).
Reference: [`cold_composition_source_reference.py`](../cold_composition_source_reference.py).
Ten independent parser/extraction controls pass. Managed runner integration is
underway; no trained result for this source test has been opened yet.

## Final-source test and the binding-order clue

COLD_COMPOSITION_SOURCE_V1 completed validly (A/D true,B/C false). On the larger
64-world cold panels, hop3 accuracy is .906IID/.813 short-cycle OOD, below the
initial16-example capability estimate. All full logits replay within1.42e-13.
Keeping only the requested answer binding and query sources gives hop3 query
KL1.771/1.828 and accuracy .625/.578. Removing earlier chain bindings lowers gold
probability .381/.309, so the final layer does not uniformly read the final fact.

The following CPU diagnostic reuses those opened panels; it is not fresh evidence.
When the final fact occurs later than all earlier chain sources, earlier-source
removal is nearly inert (.00020/-.00175 IID/OOD). When an earlier source occurs
latest, the loss is .697/.551. The six order groups nominate a more precise rule:
read the later of the second and third bindings, independently of the first.
For order2,3,1 the target binding still matters and the earlier sources do not;
for3,2,1 the earlier source matters. The original mask groups the first two bindings,
so distinguishing them requires a new test. All perworld logits are retained in
COLD_COMPOSITION_ORDER_V1_RESULT.json rather than only aggregate statistics.

The prospective CAUSAL_SUFFIX_JOIN_V1 test counterbalances all6 chain-binding orders
for each of32 IID and32 OOD worlds with two12-cycles. It defines J as the later
suffix binding and tests its removal against the other chain bindings, plus strict
full-distribution sufficiency.32 fixed-point controls test the model's observed
failure on degenerate self-loops rather than omitting them. A causal join of two
bindings on their shared middle entity is a hypothesis; native join writers and
readers have not yet been identified or extracted. All background remains charged.
