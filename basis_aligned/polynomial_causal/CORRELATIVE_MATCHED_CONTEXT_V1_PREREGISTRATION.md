# Matched inner-cue test of the fixed correlative interface

Prior art correction: v493/v495/v497 already tested the completed inner
`either ... or` construction. v495 normalized transfer was .882 on a shorter
neutral insertion and .686 on the discharged construction, with full-head
ceilings .932/.883. v493 explicitly left length/wording confounded. This test
replaces only `either` with `only`, holding token count, outer cue position,
final token and all other words fixed. It addresses a length-only explanation
for this substitution; lexical and syntactic interpretations remain confounded.

Rows use the original discharged A1 and A2 templates, 16 authored reporter
groups, seed9111290. The shared BehaviourSpec validates construction. Additional
checks enforce exactly one token substitution across contexts and one outer-cue
substitution within each context, in the same positions. No outcome filtering.
The old discharged template is reused deliberately; no global novelty or
training-OOD claim. Four panels have16rows each, both cue directions balanced.

Freeze the saved26heads/14per-block rank-one directions and76,032coefficient
folded port core. No fitting, head selection, rank, gain or mix adjustment.
For each of two frames: capture native base/donor for both contexts (4forwards),
then each context gets an independent native base, a full26head donor swap,
a folded14scalar donor swap, and a same-answer context swap from the other
context's native base (8forwards). Total24forwards384sequences, zero updates.
Reuse the compiled executor and the original native-head intervention helper.

Five frozen predicates:

A. Instrument: finite outputs, exact counts, native-base full-vocabulary bridge
maxabs<=1e-3 and relativeL2<=1e-5, all native/folded scalar checks satisfy
absdifference<=1e-4+1e-5*abs(reference).

B. Capability and reachability: all64rows have positive native base AND donor
answer margins, all donor-recovery denominators>1e-6, and full-head mean raw
recovery>=.8 in each of four panels.

C. Neutral transfer: B and normalized scalar recovery>=.8 in BOTH only panels.
Raw recovery is mean((base_margin-patched_margin)/(base_margin+donor_margin)).
Normalized scalar recovery is its raw recovery divided by full-head raw
recovery, exactly the historical v495 comparison. Both quantities are saved.

D. Matched inner-cue deficit: C and only-minus-either normalized scalar recovery
>=.15 in EACH frame. Opposing outcome: either remains comparably transferable;
a C miss means the proposed neutral benchmark itself failed. Preserve all.

E. Same-answer context invariance: B and either-to-only AND only-to-either
context-swap mean absolute task-margin movement <=.23 times the recipient
context/frame's midpoint median native outer-cue separation, in all four panels.
These are answer-preserving contexts for the SAME target computation, not
unrelated-removal controls. A pass alone cannot establish semantic identity.

Save all four native scalar corners per reporter/frame/layer. Post-result CPU
analysis will compute the mixed difference
D_l=s(neither,either)-s(both,either)-s(neither,only)+s(both,only).
An additive cue-only plus context-only scalar has D_l=0. Exact finite-panel
least-squares residual over the four corners is |D_l|/2 in L2 and unavoidable
maximum corner error is >=|D_l|/4. Compare against the summed four native
scalar bridge error bounds before treating nonzero differences as numerical
evidence. This algebraic diagnostic has no post-hoc behavioral pass threshold.
It rules out only additive separation of the FIXED coordinates on these inputs;
nonlinear encodings, context-dependent readers and alternative circuits remain.

All545902902 native parameters, upstream activations, full contextual routing,
normalizers, signed value mixing, suffix and native head outputs stay required.
No independent upstream extraction or storage saving. Managed GPU only;
hash-bound sources/artifacts and immutable result.
