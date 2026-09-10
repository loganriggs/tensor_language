# About/for query-phase portability screen

Frozen before execution, 10 September 2026. Original handoff/pilot controls.
Prior work: Claude v473 exposes a long-tail transfer failure for a fitted
about/for direction while its complete head set remains effective. Longer text
also changes contextual content; the existing result does not identify RoPE as
the mechanism. This is a different circuit and interface from the closed
first-attention write and is/was query/source searches.

Reuse v473's native fit without changing its recipe: five A–E frames, 80 A1
fit pairs and 80 C fit pairs, relaxed greedy pool40/target.97/min_gain.001/
max_units30, rank1 per selected layer, 120 steps, lr.05, seed0, complement
weight1, control weight30. Freeze units and axes before evaluating. All 16
held-out rows per eight frames remain; no capability filtering is permitted
to remove any row. Rows were already opened by v473. No fresh OOD claim.

CPU token audit: A has query-minus-cue index gap4; B–E gap3; fu gap4;
fw gap9; at/to control gap4. All A1 pairs have equal lengths and precisely
one changed token. The prose descriptions' five/ten tokens include the cue;
the actual coordinate changes from gap9 to gap4, a shift of minus5.

At every native attention invocation, retain native outputs and upstream
states. Separately compute the final-query read with its normalized content
unchanged but rotary position p'=min(p,cue_index+4). Both query factors move;
keys, values, firstV, causal mask and all source positions remain native.
Use R(p') R(p)^-1, with actual rounded BF16 tables and their non-unit
determinants. This is a local virtual-phase intervention, not rewritten text.
Read-only captures cannot propagate phase changes into later native content.

For each selected head form an alternative donor endpoint

    native_base_head + virtual_donor_head - virtual_base_head.

Use existing BLOCK-LIVE interchange with the frozen native fitted axes: each
layer moves its live head values toward this endpoint along its own axes.
Thus the receiving background and downstream recomputation stay native;
the alternate endpoint is conditional on both native prefix runs. No raw-text
independent extraction or learned semantic matching rule is claimed.

Arms: native baseline; original complete-set interchange; original fitted-axis
interchange; virtual-difference fitted-axis interchange; virtual-difference
complete-set interchange (diagnostic). Also capture donor native outputs and
full centered vocabulary logits. Short-gap rows are exact phase identities.

Predictions, all scored as written:

A. Instrument and baseline: tiny controls pass; native cache/identity bridges
meet BOTH maxabs1e-3 and relativeFrobenius1e-5; no dropped rows; finite effects;
24 selected units; all eight original complete and fitted recoveries replay
v473 within .015 absolute; methods/hooks restored; measured work stays below
2600 forwards, 300000 sequence evaluations, 120 backward steps, 600 seconds.
Exact native predictions are primary; an instrument miss is invalid, not null.

B. Phase portability: on fw, transported fitted-axis recovery divided by
original complete-set recovery is >=.80 and improves over original fitted-axis
fraction by >=.10. Opposing null: direct query geometry does not repair transfer.

C. Causal fidelity: on all seven about/for target frames, transported fitted-axis
effects reproduce original complete-set effects with <=.10 relative error in
BOTH centered full-vocabulary and answer-margin frames (denominator floor1e-8).
This stronger gate prevents a margin-only improvement from being an explanation.

D. Specificity/identity: at/to transported absolute recovery <=.30; for all
seven short-gap frames (five fit shapes, fu, at/to), transported and original
axis-interchange full logits meet the same numerical bridge as A.

No phase shift, head/site set, rank, fitting objective, dose, or template may be
adjusted after results. A B-pass/C-fail is a partial margin transfer, not circuit
identification. A B-fail closes this direct query-phase portability proposal;
do not immediately search key phases, individual heads or another shift.

All 545902902 native weights, fitted axes, row-pair cue locator and prefix/suffix
computations remain charged. No parameter or structural savings are claimed.
The test targets an explicit position-conditioned consumer operation and
held-frame causal portability. Independent extraction, selective removal on
matched grammatical controls, new-text OOD and cross-circuit reuse remain open.

The canonical C-control endpoint confound reported by Claude is preserved:
this protocol replays the existing fit for attribution, not endorsement of the
old C control as sufficient evidence of semantic selectivity.
