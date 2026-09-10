# Native folded correlative program v1

Freeze CORRELATIVE_REPLAYABLE_INTERFACE_V1_ARTIFACT.pt and the weight-derived
CORRELATIVE_WEIGHT_PORTS_V1_MAPS.pt. No new fitting. All original model weights,
native inputs and normalized/rotary two-QK routing remain required.

For each block l, compute s_l=sum_h sum_source p_h*(r_local_h*x_l+r_first_h*x_0),
using its26total selected head ports, then residual change
w_l*(donor_s_l-live_s_l). There are14 blocks, not one global scalar. Native
residual and suffix recompute after each edit. Compare this implementation to
the saved original head-space block-live q q^T edit, with all14sites installed.
Use actual native positional and head-normalization operations for routing.

Four fixed panels, each16rows: canonical held A1, canonical held A2, canonical
held P, and disjoint either/not held A1 used as C by v505. Use the original
held split rows[2::4]+rows[3::4]. No outcome filtering or fresh-OOD claim.

Seven body forwards per panel: folded-capture donor, folded-capture base,
independent original no-edit forward, original donor swap, folded donor swap,
original FIT-mean replacement, folded FIT-mean replacement. Total28forwards,
448sequences. Save full-vocabulary errors and per-row task/loss effects.

Frozen predicates:
A: folded-capture base versus original no-edit logits maxabs<=1e-3 AND
relativeL2<=1e-5 for every panel.
B: every computed block scalar versus direct q read of native head outputs
satisfies absdifference<=1e-4+1e-5*abs(reference), over donor/base/donor-edited/
mean-edited contexts and all14blocks. Record global maxabs and worst scalederror.
C: folded versus original donor and mean edits full-vocabulary logits maxabs
<=1e-3 AND relativeL2<=1e-5 for every panel/arm. Finite values throughout.
D: donor-oriented target margins between implementations differ maxabs<=.002
for every panel and both edits. This is a correspondence test, not recovery.
E: observed28bodyforwards448sequences, all56block scalar collections per panel,
and exactly14defined block writers/26head ports. Every target row is retained.

Preserve failures. The existing P-mean removal damage is expected to remain;
passing this compiler test cannot promote selective removal or semantic reuse.
Original source artifacts and previous metrics remain unchanged. Literal
folded core76032coefficients plus all upstream native program; no whole saving.
CPU trained-weight scalar/writer identities have already passed. ManagedGPU only,
hash-bound code/inputs/weights; immutable results and no bar/rank/gain rescue.
