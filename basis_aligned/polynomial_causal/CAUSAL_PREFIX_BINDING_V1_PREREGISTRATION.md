# Early-memory × late-query binding in both sentence orders

Use all32 worlds in THIRD_NOUN_CAUSAL_ROLE_ALIGNMENT_V1. At attention9..17,
queries are the last3 positions; memory contains all earlier positions. Original
early factor e is object number (index2), late factor l is attractor kind (index4).
Fronted swaps those early/late roles. Other factors remain conditioned upon.
The late factor is not available in the prefix; require its flip to preserve
prefix K1/K2/mixed-V BITWISE on every call, as well as native-prefix invariance.

For normalized, positioned query lift Phi=q1 tensor q2 and prefix memory
M=sum(k1 tensor k2 tensor mixedV), write M=M0+e Me. The mixed prefix output is
new=Phi_l Me plus inherited=Phi_el M0, each with the native output projection.
Query characters are computed from the PRODUCT lift, not products of averaged
query factors. Coefficients condition on the other three factors. Use grouped
four-by-four query/memory contractions without materializing either large tensor.

At each live receiving attention, recompute these query features from CURRENT
queries across the four counterfactual rows. Prefix states stay native by causality.
Subtract new, inherited, full mixed prefix read, or new+inherited before the native
output projection. This defines operation interventions in the conditional program;
it is not a native weight deletion. All later computations and both RMS norms live.
The existing prefix context result licenses the interface, not branch dominance.
This is distinct from MLP8 new/inherited products or local-value routing splits.

A instrument: native full three-reader grids replay MATURE_VALUE_MLP8_CONSUMERS_V1;
zero-edit full vocabulary replays native; independent full mixed cut versus summed
new+inherited cut full vocabulary agrees. Output maxabs<=1e-3 AND relative<=1e-5.
FP64 local partition closure maxabs<=1e-8 AND relative<=1e-8; exact native prefix,
late-factor prefix invariance, first-value9 preservation, earlier output unchanged,
finite outputs and restored hooks/methods. CPU planted new-only, inherited-only,
both causal orders, leakage falsifier, native attention-class zero/full-joint controls.

Let z0 be native outputs; zF,zN,zH the full/new/inherited removals. E_j=z0-z_j.
Measure Q_oh correct margin, centered three readers and centered full vocabulary.
B full-prefix necessity: ||Q zF||/||Q z0||<=.25 in EVERY world/readout.
C new sufficiency: ||Q(E_N-E_F)||/||Q E_F||<=.10 in EVERY world/readout, with
||Q E_F||/||Q z0||>=.10. D inherited sufficiency: same rule for E_H.
E effect composition: ||E_F-E_N-E_H||/||E_F||<=.10 in both mixed and full-table
versions of all three readouts, EVERY world. This scientific additivity test is
distinct from the exact joint-operation/full-cut implementation replay.
Zero scientific denominators fail. No branch promoted from smaller error or task
readouts alone. If total weak, do not search individual layers/heads/positions.

Per world native2+zero2+full2+new2+inherited2+joint2 =384forwards6144sequences,
2880patched attention calls, zero fits, 600s cap, managed GPU. Counterfactual query
features and native prefix producers retained; all545902902 parameters charged,
saving0. Grouped cross-read temporary is only4groups×4×4×9×3×128 FP64 scalars
per16-row batch (1.6875MiB); other scores, factors, buffers and native weights extra.
Passing supports a conditional cross-token operation, not independent extraction,
unseen-text prediction or selective native-weight removal. Fresh generation and
joint reuse would still be required before a semantic circuit claim.
