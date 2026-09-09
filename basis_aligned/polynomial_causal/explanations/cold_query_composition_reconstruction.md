# Reconstructing composition without earlier answers

Current result (18:03 UTC): the [forward endpoint field](forward_endpoint_field_circuit.md)
passes registered joined-field interchange, but a raw/computed two-field state
abstraction fails selective and joint semantics. Holding normalization gains
fixed does not repair it. The active source-port factorial tests content versus
source-selection effects without changing the failed field definitions. Every
native weight/background remains charged; structural reduction is unfinished.

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

Nine synthetic head-mask/port/closure controls pass. The head/port experiment has
now completed; the results are below. All400640 parameters
remain native and charged. Full reconstruction and compact extraction remain open.


## Directional writing heads and source-port use — all predictions pass

SUFFIX_JOIN_HEAD_PORT_V1 passes A/B/C/D on32 fresh worlds,768 correlated order/hop
variants. In both IID and two12-cycle OOD, the forward B2-later writer is L2H1
(hop3 loss .971/.946); the primary backward B3-later writer is L2H2 (.645/.536).
H3's backward contribution (.201/.107) is preserved and has not been discarded
from a claimed two-head replacement. Matched-edge and lower-hop controls pass.

Forward value-only port restoration recovers1.007/1.012 of the removed effect;
restoring only the two key ports recovers approximately zero. Backward key-pair
restoration recovers1.064/1.028, while value-only restores approximately zero.
Restoring K1 or K2 alone is insufficient for the full backward effect. Fractions
slightly above1 reflect changed native background/probabilities, not superadditive
proof or exact fidelity. All3 ports together reproduce the final query logits with
observed max error0, using ports reused fromhop0 on the same binding world.

This identifies directional writers and consumers consistent with the proposed
EC+CE join. The original sources and complete native background remain live and
priced. Next, SUFFIX_JOIN_MIDDLE_MATCH_V1 independently relabels the shared middle
entity in the key/value columns of a valid permutation. Both changes together
restore the join; either alone breaks it. It tests both the native joint routing
score and the signed full-query effect of cutting the frozen H1/H2 source edges.
The field correspondence is being tested causally, not assumed from attention names.


## Shared middle-entity matching — all predictions pass

SUFFIX_JOIN_MIDDLE_MATCH_V1 passes A/B/C/D on32 fresh worlds and768 correlated
order/intervention variants. Swapping the middle label only in binding keys or
only in binding values preserves a valid permutation but breaks the old pair's
join. Joint native score RMS falls to2.7–6.9% of the smaller matched-case RMS;
centered full-query ablation-effect RMS falls to0.9–4.3%. Swapping both columns
conjugates the function, restores matching, and restores a live causal route.
Matched-route gold-probability losses are .900–.954 forward and .671–.707 backward.
All native per-case capability gates pass, with no correctness filtering.

This supplies causal evidence for the equality on the shared middle entity, beyond
source location and port semantics. It does not claim each native dot-product factor
has one semantic field: the measured object is the complete product-attention score
and its full-vocabulary effect. A CPU inspection of saved score cells finds98.7–99.7%
of matched score energy in binding-value→binding-value cells. The other cells remain
in the recorded evidence; small scores are not exact zeros.

The next SHARED_JOIN_KERNEL_V1 candidate moves from localization to replacement.
It proposes causal local key/value records, one equality rule reused with swapped
fields by the two heads, the original live RMS gains, and small role/lag/entity
coefficients fitted only on native routing scores. The full native prefix and value/
readout computations remain charged. It would physically remove32768 selected Q/K
weights and add6576 coefficients, for a new26192-constant reduction beyond generic
readout folding. Every input position and full29-way distribution must pass fresh
prediction and single/joint-removal tests, including fixed-point failures.

The record grammar extends the confirmed binding-value mechanism to other cold
input roles and is explicitly a new hypothesis. Same-binding joins are excluded
before fit, matching the distinct-edge operation. No unseen coefficient, teacher
activation cache or function-answer oracle is allowed at runtime. The parser and
fixed8-pass weighted fit core pass CPU controls; physical executor/managed integration
is outstanding. No learned weights have been removed or fidelity established yet.

## Shared-kernel compilation: first instrument failure

`SHARED_JOIN_KERNEL_V1_RESULT.json` physically removes the L2H1/H2 Q/K rows
and counts361776 arbitrary constants, compared with387968 after the generic
final-readout fold (400640 original). Twenty-two CPU parser/fitting/executor
controls pass. This does not establish trained fidelity: the random256-example
calibration observes555 role/lag cells, but the held IID/OOD/fixed-point panels
require14/16/9 unseen cells. Evaluation correctly refuses to invent coefficients.
No held model outputs were evaluated. B/C are therefore untested, not measured
scientific failures; the reported exact-error0 is an empty maximum, not closure.
The matched calibration score residual.7838 is diagnostic only.

V2 makes one declared coverage repair, preserving the same semantic rule and8
ALS passes: append96 entity×hop forks of one independent calibration world to
original256 inputs. A CPU audit establishes590/590 structurally possible role/lag
cells before training. New held seeds16910/11/12 are unopened until coefficients
freeze. The repeated calibration prefix is counted, not called96 independent
worlds. No feature, head, role, lag or fitting-iteration sweep is allowed. A valid
failure closes this fixed local-record kernel. The causal middle-match evidence
remains distinct from the untested claim that entire native heads implement it.

## Shared-kernel v2: valid rejection and failure diagnosis

`SHARED_JOIN_KERNEL_V2_COVERAGE_RESULT.json` passes instrument and physical
accounting gates (A/D), and fails distribution and intervention gates (B/C).
The managed run takes1.83s. All590 possible cells are observed, values are finite,
and exact native/folded replay closes at1.14e-13. With both replaced heads removed,
native and candidate agree below8.6e-14: the retained computation is correct.

| Held population | Full mean KL, all positions | Full mean KL, query | Native / candidate query accuracy |
|---|---:|---:|---:|
| IID64 |36.274|1767.451|1.000 / .625|
| Two12cycles64 |82.944|4135.141|.969 / .578|
| Fixed-point32 |80.368|4023.867|.000 / .063|

All single/joint removal-effect gates fail. The prospective reduction from387968
to361776 arbitrary constants is physically real but behaviorally unusable. It is
not a successful extraction or simpler explanation. The fixed local-record kernel
is closed; no role/head/lag/feature or fitting-iteration rescue follows.

`SHARED_JOIN_KERNEL_FAILURE_AUDIT_V1.json` is a CPU diagnosis on those already
opened cases, explicitly not new held-out evidence. Retaining the exact native
scores wherever the rule permits a match, while zeroing its complement, still
fails fidelity: all-position meanKL1.050/1.167/1.090, query meanKL.130/.475/5.814.
IID query accuracy remains.984, illustrating why task accuracy cannot substitute
for full native distribution. CPU/native saved-GPU replay error is below4.5e-13.
Thus the rule's support erasure is itself insufficient, independently of fitting.

The huge fitted outliers also have a concrete coefficient explanation. In the
worst IID score case the fitted rule gives-215876 versus native-.0362, despite
ordinary RMS gains2.62 and3.01. Its theta=-2027 was calibrated only on entities1/3
with nearly clamped gamma; a new entity14 has gamma1.709. The largest OOD and
fixed-point cases use theta2764, calibrated on only entity14 (gamma≈1e-6), then
reuse it with order-one gamma for other entities. This is a failure of the proposed
separable positive entity factor, not evidence of tiny-norm singularities or an
executor fault. No regularization or additional fit is selected after the fact.

A separate algebraic sensitivity check confirms that scaling native L2 residuals
by2 leaves RMS-normalized scores unchanged to2.5e-15, while the proposed fixed
rule scales by1/16. This synthetic rescaling is not a legal token counterfactual;
it exposes a dependence that the semantic ansatz did not explain. The observed
held-token failures above are the actual rejection evidence.

What survives is narrower and useful: causal middle-entity matching, directional
writing by L2H1/H2, and native final-source key/value consumers reused across query
hops. Entire heads are not identified with that operation. The next scientific
object is a causal join contribution with its necessary contextual inputs, not a
refitted whole-head equality lookup. The failure audit has begun that separation
by distinguishing exact-native matched support from learned score amplitudes.

## Two simultaneous join writes: causal reuse and contextual limits

`JOIN_CONTRIBUTION_CONTEXT_V1_RESULT.json` tests32 fresh worlds (16 IID24-cycle,
16 two12-cycle), two arrangements of two disjoint query chains, and eight query
forks perarrangement. One selected writer is H1/forward and the other H2/backward.
Donor/recipient serializations preserve all six chain facts and their positions,
as well as the full function, while shuffling the other18 facts. This tests an
explicit additive write, not a whole-head semantic replacement.

Instrument and selectivity gates A/B pass. Joint restoration of the native writes
closes all logits at4.97e-14. Both writes are exactly unchanged across the eight
query forks. Removing each loses.423–.987 gold probability on its own hop3 query;
other-query and lowerhop controls all meet their registered absolute.10 bar.
Thus the two computations can be selectively removed together in one context,
with an explicit additive L2 interface and live downstream recomputation.

Cross-context distribution and quantitative effect gates C/D fail. Joint transfer
has all-position meanKL.0069–.0173 and query meanKL.00335–.01136 across the four
population/arrangement groups. Joint query-effect relative errors.167–.263 exceed
.01. Forward transfer alone passes the registered query-distribution gates pooled
over all eight query forks, but neither direction passes the stricter causal-vector
comparison. Identical local facts/positions do not determine a context-independent
native write at the requested precision. All400640 parameters and donor-prefix
execution remain required; this is not an independently extracted smaller program.

The immediate `JOIN_CONTEXT_FACTOR_AUDIT_V1.json` successor uses those opened cases
for exact score/value factor diagnosis, not fresh identification. Since each write
is A V O, its donor-minus-recipient difference splits exactly into score change,
value change and their interaction. Mixed factors are run through the full native
downstream reader. Donor scores with recipient values pass all pooled query KL
gates, but query-effect errors remain3–11%. Donor values reproduce the backward
transfer failures. Value-change write RMS is.946–.968 of total write-change RMS;
these correlated terms are not additive variance fractions. Native write replay
from the CPU factor implementation closes at3.77e-15.

Next, `JOIN_VALUE_PRODUCERS_V1_PREREGISTRATION.md` tests an exact producer split:
pre-L2 x2=E/4+Y0/4+Y1/2. The three corresponding V terms share the live L2 RMS gain
and original selected routing. The fixed semantic hypothesis is forward current
value from E and backward origin key from Y0+Y1. All8 subsets are an exact causal
factorial, not a best-subset search; a fixed single/joint candidate must pass the
original distribution and effect bars on fresh18909/18910 inputs. The primitive
`join_value_producer_reference.py` is implemented and passes five CPU controls,
including nonzero term closure and exact joint restoration. Fresh generator and
managed runner integration remain. No native dependency or coefficient is hidden
or counted as eliminated by this diagnostic representation.

## Exact payload producers: semantic field evidence, insufficient precision

`JOIN_VALUE_PRODUCERS_V1_RESULT.json` passes A/B and fails C/D on fresh18909/18910
worlds. The exact residual decomposition closes at8.88e-16, selected write sums
at2.22e-15 and joint all-term logits at4.26e-14. Runtime6.14s via managedGPU.
Forward E-only recovery is.969–.999; backward (Y0+Y1) recovery.952–1.003. Their
complements recover at most.103. This supports the specified current-value versus
contextually supplied origin-key distinction at the task-effect level.

The predeclared complete factorial further nominates Y1, the second attention
output, as the backward field producer: Y1alone recovery.951–.996; Y0alone≈0.
This nomination needs a fresh producer-source test. No best subset replaces the
registered candidate. That candidate still fails full native prediction: joint
all-position meanKL.0135–.0333, query meanKL.00332–.00973, joint query-effect
relative error.197–.232. Task recovery near100% remains weaker than the goal.

The next `JOIN_ORIGIN_WRITER_V1_PREREGISTRATION.md` tests a single explicit path:
L1 previous-key→current-value attention, feeding the L2 backward join V and final
key consumers. Fresh19909/19910 worlds; all four L1 heads retained, no head/source
selection. The path and exact complement are separately restored, both with the
independent forward route present and removed. Its reference passes five CPU
controls including all-source Y1 identity, single/conditional-joint replay and
future-token invariance. All native coefficients and normalization dependencies
remain charged. A semantic path result will not erase the quantitative failures.

## Origin-copy path and a derived composition boundary

`JOIN_ORIGIN_WRITER_V1_RESULT.json` passes A/B and fails C/D on fresh19909/19910
worlds. The explicit preceding-key→value-position L1 path feeding L2H2 restores
.883–.988 of the backward task effect; its exact complement restores at most.107.
Native single/conditional-joint restoration closes at4.26e-14. Quantitative query
effect errors.311–.403 still prevent a sufficient replacement claim. Source identity
is now explicit across two attention layers, but semantic task effect is not the
same as exact native execution.

The identical raw query-effect vectors with the independent forward route present
and removed motivate a different grouping criterion. The1648 mathematical review
derives an exact restriction for this checkpoint's one-layer suffix: disjoint
post-L2 source-state edits give at most pairwise interactions in Boolean edit bits.
At unchanged query positions the effects are additive. Pair interactions are
restricted to changed query positions reading another changed source. Multiple
producer edits entering the same RMS-normalized source state can interact at
higher order. Log probabilities may be nonadditive even when logits are additive.

This is an application of anchored function decomposition/Boolean inversion,
with a model-specific dependency-support proof, not a learned parameter reduction.
The full native prefix/reader remains charged. It does not apply unchanged with
another suffix layer, source softmax or post-attention normalization. The CPU
reference verifies integer and FP64 disjoint/overlap controls. The fresh trained
`SOURCE_SUPPORT_INTERACTION_V1_PREREGISTRATION.md` audit now tests withheld triple
prediction, pair-support localization, overlapping producer edits and nonlinear
log-probability controls. The goal remains a smaller independently executable
program; this step establishes which circuit groupings permit exact composition.

## Trained source-support law verified; exact edit executor underway

`SOURCE_SUPPORT_INTERACTION_V1_RESULT.json` passes all four gates in3.27s. On fresh
IID/three8-cycle inputs, baseline/single/pair responses predict the withheld triple
removal at maximum all-logit error1.85e-13/1.56e-13. Forbidden pair-support leakage
is at most2.14e-13; allowed pair effects are live (max10.84/6.08). Three overlapping
producer edits have third-order centered RMS.00583/.01235. At unchanged finalquery
positions, log-probability pair interactions reach7.44/6.45 even though raw-logit
pair interactions vanish. Native cached-suffix/full-forward replay is4.97e-14.

This establishes a useful composition/grouping rule under the precise one-layer
suffix assumptions. It does not discover smaller model weights or make the
context-independent semantic kernel valid. The next step turns the shared query/
source reader structure into a direct executable interface, rather than retaining
seven response vectors perinput as an unexplained cache.

`exact_source_edit_reference.py` computes its native-prefix and final Q/K/V cache
from supplied tokens, then recomputes normalized projections only at changed
positions. It updates source messages to all queries and recomputes the complete
source sum for changed query positions. Overlapping component edits are combined
before normalization, so their higher-order effects are preserved. Absolute RoPE
uses original positions. Final O is physically removed using the known generic
O/head fold; expected constants387968. That saving is generic, not a new semantic
reduction, and all native prefix/reader weights remain charged.

`EXACT_SOURCE_EDIT_V1_CPU_CONTROLS.json` passes15 controls, with nonzero edit replay
at1.11e-16 and explicit wrong-position negatives. Initial bitwise comparisons of
projections with different GEMM row counts were replaced before trained execution
by the registered numerical tolerance; full-logit controls were already passing.
`EXACT_SOURCE_EDIT_V1_PREREGISTRATION.md` specifies fresh21909/21910 disjoint,
overlapping and half-dose tests, independent CPU export/replay, and preparation/
edit/cache pricing against both full native and cached native suffix baselines.
Managed trained runner and export integration remain. This is reusable intervention
infrastructure on the path toward the goal, not an adopted compact circuit model.

## Exact edit executor and export verified

`EXACT_SOURCE_EDIT_V1_RESULT.json` passes all four gates. All17 disjoint,
overlapping and half-dose edit arms on384fresh query variants (32independent
worlds) close within1.14e-13 across all logits. Preparation closes at9.95e-14.
FreshCPU export replay136example-arms closes at2.06e-13, with original checkpoint
reads blocked. The exported program recomputes every runtime state and native
write from supplied tokens and its own weights; test fixtures are separate.
Usage and limits are in `EXACT_SOURCE_EDIT_V1_README.md`.

The performance result is a useful negative: sparse edits take.972/.990ms for2/6
changed positions, versus.511/.503ms for the native cached suffix. Full native
forward is1.841ms; program preparation1.882ms. Cache1510008bytes, fixed buffers
2334720bytes,387968 independent constants. The12672 constant saving is entirely
generic final-readout folding; new semantic parameter saving remainszero. All
native prefix/readers stay explicitly charged. Do not adopt this as a speed win.

The next test uses that precise execution interface for semantic origin-field
interchange. `JOIN_ORIGIN_FIELD_SWAP_V1_PREREGISTRATION.md` fixes two backward
joins and exchanges only their copied L1 value reads, preserving recipient gates,
gains and complements. If they denote origins, joint exchange should swap the two
hop3 answers, with lowerhop specificity and consistent single transfers. This is
not a claim that edited intermediate joins must remain the square of the original
function. Fresh22909/22910 inputs, no fitting or donor/head selection.
`join_origin_field_swap_reference.py` passes seven CPU controls, including an
independent native pattern/value-hook correspondence. Managed integration remains.

## Origin interchange is selective but fails reliable semantic retargeting

`JOIN_ORIGIN_FIELD_SWAP_V1_RESULT.json` completes in5.11s: instrument and
specificity/composition pass, semantic retargeting and isolated reuse fail.
Native hop3 accuracy is.9375 in every query/population group. Joint swaps reach
only.50–.6875 desired-answer accuracy against the frozen.80 bar; isolated singles
reach.50–.6875. Desired-probability gains are.426–.670. Lowerhop mean goldP changes
are at most2.95e-5, and raw query composition closes within1.28e-13.

The exported program's independent native intervention correspondence closes at
1.42e-13 over all logits, relative RMS8.28e-16; independently hooked paths close
at2.11e-15, identity exactly0. Thus execution is valid, but the edited contextual
copy is not established as a reliably interchangeable origin field. This is not
an extraction or semantic parameter-reduction success;387968 native constants
remain necessary. Do not relax the bar to generic damage or partial retargeting.

`JOIN_ORIGIN_FIELD_SWAP_FAILURE_AUDIT_V1.json` uses only saved outputs, retaining
all cases. Of64 hop3 query cases,38 retarget; the26 failures include10 original
answers,5 other selected-chain entities and11 unrelated tokens. Joint/single
correctness differs on only2 cases: one rescue and one loss. Some desired ranks
reach21–28. This is broader than residual original-answer competition and is not
resolved by removing the other selected route. This opened-case diagnosis does
not establish a fresh intervention claim or justify a donor/head/gain sweep.

The distinct next test is `JOIN_ENDPOINT_FIELD_SWAP_V1_PREREGISTRATION.md`.
Earlier producer evidence localized forward endpoint transport mainly to the
original embedding term (.969–.999 task-effect recovery). Exchange just its
source-value embedding contribution between two forward joins, under recipient
scores/gains and with all other producers retained. Singles should retarget their
own query while preserving the other, and their joint use tests reuse. This is a
semantic interchange test, not an attempt to improve the failed backward donor.
Fresh23909/23910 worlds and all bars are fixed before trained outcomes.
`join_endpoint_field_swap_reference.py` passes7 CPU controls, including an
independent native L2 pattern/value-input hook oracle. Managed integration remains.
