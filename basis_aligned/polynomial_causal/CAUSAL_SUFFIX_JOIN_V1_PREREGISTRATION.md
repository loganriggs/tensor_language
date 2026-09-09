# Causal suffix-join source hypothesis

The cold final-answer-binding test failed. Its subsequent opened-panel order audit
passes both registered visibility implications: earlier-chain loss .00020/-.00175
when the target fact is latest, versus .697/.551 when an earlier source is latest.
The six order groups suggest a more precise new hypothesis: final hop3 retrieval
reads the later of bindings B2=(f(e),f2(e)) and B3=(f2(e),f3(e)), independently of B1.
For example order2,3,1 uses B3 although B1 is latest; order3,2,1 uses an earlier-chain
source. The missing B1/B2 distinction in the first audit requires a fresh test.

Interpretation to test, not assume: a causal join on the middle entity stores the
two-step relation f(e)->f3(e) at the later suffix binding. If B2 is later, it can
read the earlier B3 to obtain the answer; if B3 is later, it can read the earlier
B2 to obtain the query key. This would explain two serialization orientations of
one composed relation. Source evidence alone does not identify the native join heads.

Fresh32 IID24-cycle worlds seed11909 and32 OOD worlds consisting of two12-cycles
seed11910. One entity/world, hop3 only. Place its three distinct chain bindings in
slots3/11/19, enumerate all6 orders while keeping the other bindings fixed. These
are64 independent worlds and384 correlated order variants. The long-cycle OOD
keeps the proposed two-edge join nondegenerate; this restriction is explicit.
Also32 fresh permutation worlds seed11911 with the queried entity forced to be a
fixed point by swapping permutation images. This control addresses the observed
failure on all4 fixed-point examples in the previous OOD set; it is not omitted
to make a capability gate pass. Fixed-point worlds are32 additional independent cases.

For each cold51-token input define J as both positions of the later B2/B3 binding;
I is the other chain-binding positions, excluding J; Q is the3 query positions.
For a fixed point B2/B3 are the same edge: J is empty (the hypothesis is a join of
two distinct stored edges). I contains that binding pair. Sets are disjoint.

Arms: full, removeJ, removeI, removeJ+I, keepJ+Q. Reuse exact final readout fold,
live3-layer prefix, Q/K/V and residual; charge all opaque weights and parser.
The J rule is frozen here, not fit to new attention scores or outcomes.

A: parser/order/overlap controls; every native/extracted full29-way logit and
centered removal vector agrees at1e-9, including exact disjoint joint additivity.
B: in each of12 population/order groups, native accuracy>=.8, removeJ gold-P
loss>=.50, and absolute removeI gold-P loss<=.10. No per-row capability filtering.
C: keepJ+Q query distribution KL mean<=1e-3,p99<=1e-2 in each of those12 groups.
D: fixed-point native hop3 gold probability<=.15, as predicted by a non-reflexive
join account of its observed failure. Report the full32 control outcomes even if
this fails. The candidate's fixed-point distribution discrepancy is reported too;
correct task labels alone cannot establish full native-distribution prediction.

B can nominate a causal join location even if strict full-distribution C fails;
do not call that a complete extracted join or overall success. If B fails, close
the simple later-suffix-source rule, no order/head/threshold sweep. If B passes,
test which upstream writers construct its middle-entity matching and key/value
payload, including both orientations and their joint manipulation, before semantic
promotion. GPU only via managed enqueue, batch4 FP64,1800s,<256MiB per tensor.
