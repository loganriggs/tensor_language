# Fixed correlative projector/complement across two behaviors

Original handoff/pilot authority. Reuse saved26heads and per-block rank-one q;
P=q q^T and R=I-P form an exact additive split even with finite-precision q.
q orthogonality maxerror must<=1e-5. Rank14 across14blocks; remaining3314 head
coordinates are a broad opaque complement, not a newly named semantic circuit.
No fit, head/rank/dose/gain selection, or claimed weight saving.

Prior v501/v505 found the same head set reaches disjoint either/not->or/but and
the both/neither scalar is nearly inert there. FIT loss constrained complement
inertness on the original task. Neither fact establishes held-row double
interchange AND selective mean removal across these two behaviors. Earlier
first/local and routing/value single-operand failures remain closed.

Use all existing recombined A1/A2 and disjoint C16rows each, seed9111280. All
are answer-changing within their own task: A1/A2 both/neither->and/nor, C
either/not->or/but. No model-based filtering, new-data or training-OOD claim.
Both native endpoint answers are checked explicitly. P answer-preserving
reporter rows are not used as unrelated removal controls.

At each LIVE block, swap P(donor-live), R(donor-live), or the complete selected
heads. Also replace P, R, or full selected heads by saved original FIT means,
with the other part left live. All downstream computation is recomputed.
Mean replacement is the specified removal operation, not a proof that the
feature has been erased on every possible input or task.

Ten forwards per panel: compiled native donor and base capture, independent
native base, full head donor swap, P donor swap, R donor swap, existing folded
scalar donor swap, full mean replacement, P mean replacement, R mean replacement.
Total30forwards480sequences, zero backward/fit/native updates. Reuse existing
g.forward_units q-dict/complement semantics and the folded scalar bridge.

Five frozen predicates:

A. Instrument: exact counts, finite logits, q orthogonality<=1e-5, native/folded
scalar checks<=1e-4+1e-5*abs(reference), native-base and P-vs-folded full-logit
bridges maxabs<=1e-3 AND relativeL2<=1e-5.

B. Capability and head-set reachability: both native endpoint margins>0 on
ALL48pairs, positive recovery denominators>1e-6, full selected-head mean raw
recovery>=.8 separately in A1/A2/C.

C. Double interchange: B; P raw recovery>=.8 and absolute R raw recovery<=.23
separately in A1/A2; R raw recovery>=.8 and absolute P raw recovery<=.23 in C.
Raw recovery is mean((base_margin-patched_margin)/(base_margin+donor_margin)),
with each native margin oriented toward its own endpoint's intended answer.
Do not substitute a ratio normalized by full-head recovery.

D. Selective mean removal: B; P mean-replacement CE damage>=.5 nats and R
mean absolute CE change<=.1 nats in each A1/A2; R mean-replacement CE damage>=.5
nats and P mean absolute CE change<=.1 nats in C. CE is correct base-answer
cross-entropy over the full vocabulary. Use mean ABSOLUTE per-row CE change
for preservation, so improvements cannot cancel harms. All bars prospective.

E. Additive endpoint effects: B and
||center(z_full-z_P-z_R+z_base)||/||center(z_full-z_base)||<=.10 in all3panels.
Orthogonal coordinate geometry does not guarantee this nonlinear endpoint
identity. Full mean-removal damage is reported as an additional reference.

A C pass supports a conditional interchangeable partition; D failure prevents
claiming independent selective removal. E failure prevents predicting joint
effects by adding singleton effects. Even C/D/E passing would not explain the
3314-coordinate complement's producer or make it structurally simple. Retain
all misses; do not train a new direction or choose another mean to rescue them.

Actual model count545902902nativeparameters stays charged. CPU post-result
paired bootstraps over16authored groups will describe uncertainty in CE damage,
raw transfer and endpoint interaction, with no threshold changes. Managed GPU
only, hash-bound sources and immutable receipt.
