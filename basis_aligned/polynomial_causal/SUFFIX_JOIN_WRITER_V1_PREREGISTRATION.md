# Upstream writers of the nominated suffix join

Parent CAUSAL_SUFFIX_JOIN_V1 passed the exact instrument, source necessity/control
gate in all12 positive groups, and fixed-point failure prediction. Strict full
distribution sufficiency failed: keepJ+Q answers all384 positive rows correctly
but does not reproduce native probabilities/errors. Preserve that distinction.

Hypothesis: an upstream attention operation joins the earlier suffix binding S to
the later suffix binding J. Source rule alone does not prove this dependency.
Intervene on all four edges from the two S token positions to the two J positions
in prefix attention layer0,1,2 separately, and all3 jointly, across all4 heads.
Recompute all downstream states and normalization. No head selection or fitting.
For each arm also cut a count-matched nonjoining source pair in the next binding
slot after S; it remains before J. Control lag differs by two tokens, explicitly.
The3 chain bindings occupy slots3/11/19, so this control is never a chain binding.

Fresh16 IID24-cycle worlds seed12909 and16 two12-cycle OOD worlds seed12910.
Counterbalance all6 orders at those slots and fork each into query hops0..3.
This is32 independent worlds,768 correlated query/order variants. Evaluate the
same S→J edges for every hop: hop0..2 are controls for a nominated hop3 cache.
All outputs are full29-way logits. Retain every example regardless of correctness.

Additionally, collect native K1/K2/V at J as an explicitly labeled intervention
reference. After cutting S→J in all3 prefix layers, restore only those final-layer
source ports at J. These native reference activations are for causal rescue, not
an extracted program or a permitted cache at ordinary evaluation. The source ports are taken from the hop0 member of the same binding/order world
and reused for all4 query hops, with a causal-equality check at J. Query ports and
all other source positions remain recomputed in the intervened model.

A instrument: synthetic explicit masked-attention/native-hook replay1e-9, live
cuts, identity port-rescue control, restored hooks; native hop3 accuracy>=.8 in
each population/orientation (B2-later vs B3-later), no per-order admission filter.
B selective upstream dependency: the same one of {layer0,layer1,layer2,joint} in
all4 population/orientation groups has hop3 gold-P loss>=.50, matched source-control
absolute loss<=.10, and absolute hop0/1/2 loss<=.10. All4 arms reported, no promotion
from choosing an arm on these same results; passing nominates a writer interface.
C rescue: native-port restoration recovers>=.8 of joint-cut mean gold-P loss in
every population/orientation, joint-cut loss>=.50 required. Also require query
distribution KL mean<=1e-3,p99<=1e-2 versus native in those groups. A rescue that
only fixes task labels does not pass this full-distribution gate.
D source reference: uncut model with its own K1/K2/V restored at J agrees with
native full logits1e-9. Persist per-row query logits for all arms and grouping.

If B fails, close this direct pair-to-pair writer hypothesis; inspect indirect
paths only if measured effects justify it. If B passes, identify within-layer
head/factor writers for both orientations and test the explicit EC+CE join,
not another source mask or a decoder-only claim. All400640 weights remain native
and charged. Batch4 FP64,1800s,<256MiB/tensor, managed GPU exclusively. Reuse the
counterbalanced generator/scorers with fresh seeds, no new training or rank sweep.
