# Routing and value operand swaps in the frozen correlative interface

Original handoff/pilot authority. Matched long-frame native capability failed;
do not rescue that test. Use all existing short-frame recombined A1/A2/P and
disjoint C rows,16groups each, seed9111280, whose both-endpoint answers passed
in the previous run. No new-data or training-OOD claim, no outcome filtering.
Recompute capability explicitly with circuit_endpoint_capability_v1.

Prior art: correlative first/local source tests retained complete native routing.
v173/v174 split the cue-token first-value versus residual-input route and shared
head-index value channels. They did not swap the two operands of this saved
26head/14block scalar interface. Induction factor failures remain closed; this
is a different supported behavioral interface, without head/rank/gain rescue.

At each block s=sum_{head,source} P*U. P is the actual masked native product
attention score at the semantic query. U is the scalar value read after folding
the saved direction through both local and first-layer value matrices, with
actual signed mixing. Capture donor P and U. At each LIVE recipient block:

    route swap: delta_s=sum((P_d-P_live)*U_live)
    value swap: delta_s=sum(P_live*(U_d-U_live))
    joint swap: delta_s=sum(P_d*U_d-P_live*U_live)

Write w_l*delta_s at the original output hook and recompute the suffix. Every
branch recomputes its own live recipient operands after earlier block edits.
Joint must match the existing frozen scalar donor executor. Partial writes
alter only the selected scalar port; no complete native head is overwritten.
No normalizer, rounded rotary table, mixture coefficient or upstream path is
removed. Exchanging full P is not a token-only selector; full U is not a token-
only payload. These remain conditional operand interfaces unless independently
explained and extracted.

Seven forwards per panel: factor donor capture, factor base capture, independent
native base, factor joint swap, original scalar joint swap, route swap, value
swap. Total28bodyforwards448sequences, zero fits/backwards/updates.

Five frozen predicates:

A. Instrument: exact counts and finite logits; native base and joint-vs-original
scalar full-vocabulary bridges maxabs<=1e-3 AND relativeL2<=1e-5; every scalar
bridge error<=1e-4+1e-5*abs(native reference). Natural product-rule identity below
must agree within1e-4+1e-5*abs(joint scalar change).

B. Native and joint reference: all64pairs have both intended endpoint margins
>0; all target recovery denominators>1e-6; joint mean raw recovery>=.8 separately
on A1/A2; P/C joint absolute margin movement divided by A1 midpoint median
native cue separation<=.23 each. P/C are same-answer and disjoint controls as
in the existing source test. No new unrelated-removal claim.

C. Route sufficiency: B; route raw recovery>=.8 and centered full-vocabulary
effect error relative to joint<=.15 separately on A1/A2; P/C movement<=.23.

D. Value sufficiency: identical C requirements for the value swap.

E. Additive endpoint effects: B and
||center(z_joint-z_route-z_value+z_base)|| / ||center(z_joint-z_base)||<=.10
separately on A1/A2. All errors aggregate rows and prediction vocabulary.

Opposing outcomes: a C/D pass identifies a conditional sufficient operand path;
both failing closes that factor-only simplification at fixed interface and
bars. E miss means singleton endpoint effects do not add at the stated tolerance,
not that the joint bilinear program is invalid. No dose/rank/head/mixture rescue.

Save per-row native-state scalar terms
delta_s = delta_P*U_b + P_b*delta_U + delta_P*delta_U, summed over sources/heads.
This exact identity localizes scalar interaction but is NOT an additive prediction
of final effects: live states differ between intervention branches. Post-result
CPU bootstraps and write-weighted term norms will be descriptive; no threshold
or candidate chosen from those statistics is retrospectively registered.

Reuse frozen maps and original executor as reference; the small operand executor
adds explicit P/U capture and factor swapping.28forwards includes its native and
joint positive controls. All545902902 native parameters plus76032corecoefficients
remain required; no weight/compute saving or independent upstream extraction.
Managed GPU only, immutable source hashes and result. Saved factors are transient;
compact scalar terms and per-row outcome/error statistics are durable.
