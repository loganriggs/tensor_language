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
Ten independent parser/extraction controls pass. The source test and its
counterbalanced follow-up have now completed; their results are below.

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


## An explicit operation to test

Represent one binding u→v by the matrix E=|v><u|, which maps the one-hot key u to
its value v. Let C contain the sum of earlier binding matrices. At the arrival of
one new binding, compute

\[
J = EC + CE,\qquad C\leftarrow C+E.
\]

J contains the two-edge paths completed by this new fact: one orientation follows
an earlier edge then the new edge; the other follows the new edge then an earlier
edge. Each completed path is stored at the later of its two binding sources.
There are no learned coefficients in this candidate operation.

After every binding has arrived, writing F for the full function matrix,

\[
\sum_i J_i = F^2-\sum_i E_i^2.
\]

This is an elementary expansion, not an empirical rank claim. For a function with
no self-loops every same-edge square is zero, so the sum is exactly F². An isolated
self-loop is excluded because composing it with itself would reuse the same stored
edge. The proposed hop3 computation first obtains F e and reads the two-edge table
at that key: (sum J_i) F e. This yields F³ e on the non-loop permutation domain and
predicts the observed missing result on fixed-point queries. Full teacher logits
and native intermediate implementation remain unverified by this algebra.

The source-order nomination and failure-mode nomination therefore have one concrete
candidate explanation. They are not merely separate descriptions of attention plots.
The counterbalanced native experiment below tests its source prediction; the
underlying native writer computation remains to be established.

The integer reference checks all27 three-node functions, all6 source orders, and
all8 subsets of retained original facts (1296 cases), plus single/joint source-write
removals. All checks pass. Deleting an original fact recomputes subsequent J writes;
simply deleting that fact's own J write gives a different result in the explicit
counterexample. This distinction is required for a reusable stateful explanation.

Reference: [causal_edge_join_reference.py](../causal_edge_join_reference.py).
Receipt: [CAUSAL_EDGE_JOIN_REFERENCE_CONTROLS.json](../CAUSAL_EDGE_JOIN_REFERENCE_CONTROLS.json).
These are candidate algebra and intervention controls, with zero trained-model calls.
They do not establish extraction, compact native replacement or the overall goal.


## Counterbalanced source test — nomination supported, strict fidelity failed

CAUSAL_SUFFIX_JOIN_V1 passes A/B/D and fails C. All12 population/order groups pass
the prospective causal source criteria. Removing the later suffix binding J loses
.856–.978 mean correct-answer probability; removing the other chain bindings has
absolute mean effect at most .019. This resolves the first audit's ambiguity about
B1 versus B2: the relevant location follows the later of B2/B3 across all6 orders,
including when B1 is last. The test covers32 IID24-cycle and32 two12-cycle worlds,
with384 correlated order variants and no per-row correctness filter.

Retaining J plus query sources gives100% task accuracy on those384 variants, but
its native-to-candidate KL is .0043–.526, missing the strict full-distribution bar.
It removes native errors as well as other source contributions. Better task answers
are not faithful prediction of the original model; native background remains part
of the exact decomposition and its price. Native/extracted all-arm logits agree
within1e-13. The32 fixed-point controls all fail natively, mean correct-answer
probability .00082, as the distinct-edge join hypothesis predicted. This supports
a failure-mode prediction; it is not full-distribution prediction on those controls.

The next SUFFIX_JOIN_WRITER_V1 test cuts the actual S→J pair-to-pair attention edges
in each prefix layer and jointly, with matched nonjoining edges and queryhop0/1/2
controls. It also restores only native final K1/K2/V atJ from the same binding
world's hop0 execution, reused for every queryhop. These are explicit causal
reference activations, not an extracted runtime cache. A successful rescue must
recover both effect and full distribution, not only task accuracy. Ten synthetic
masked-attention/rescue controls pass; managed integration is complete.


## Upstream causal writer localized to layer2

SUFFIX_JOIN_WRITER_V1 passes A/B/D and fails C. Cutting the four pair-to-pair S→J
edges across all heads of layer2 (zero-based) loses .891–.971 hop3 gold probability
across IID/OOD and both binding orientations. Lower-hop and adjacent nonjoining
source controls remain small. The same selectivity gate passes for the joint
three-prefix-layer cut. Layers0/1 individually do not pass the shared writer bar.
This connects the nominated join source to a specific upstream attention operation.
It does not yet identify the writing heads or prove middle-entity equality matching.

Native final source ports from the same world's hop0 execution can be reused
exactly across the other query hops. Restoring K1/K2/V atJ after all3 prefix layers
were cut recovers .9991–1.0025 of the behavioral loss, but three of four groups miss
the strict full-distribution gate. Mean KL is .00036–.00206; p99 is .0053–.0504.
The native errors/background cannot be discarded just because rescue accuracy is high.
These ports are labeled causal reference activations, not an extracted runtime cache.

The new SUFFIX_JOIN_HEAD_PORT_V1 protocol isolates only layer2. It tests each of its
four heads' edge contributions and all8 combinations of final K1/K2/V restoration.
A proposed directional distinction is falsifiable: later B2 needs a new value from
B3, whereas later B3 needs a new key from B2. Native port names alone do not prove
that distinction, and failure will retain a coupled representation. With only L2
edges cut, restoring all source ports should exactly restore the final query logits:
other destinations within L2 never consume its changed outputs. This exact statement
does not cover logits at the changed J positions themselves.

Nine synthetic head-mask/port/closure controls pass. The next protocol and reference
are implemented; managed runner integration is outstanding. All400640 parameters
remain native and charged. Full reconstruction and compact extraction remain open.
