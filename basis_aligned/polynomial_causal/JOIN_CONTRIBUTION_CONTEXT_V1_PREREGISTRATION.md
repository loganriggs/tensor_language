# Two join contributions: context interchange v1

The whole-head local-record equality kernel is rejected. Causal source, H1/H2
writer direction, middle matching and final source-port reuse remain supported.
Now test whether a precisely delimited additive write, rather than a whole head,
is a reusable context-independent two-edge computation for downstream readers.

Fresh16 IID24-cycle worlds17909 and16 OODtwo12-cycle worlds17910. Two disjoint
three-edge query chains start at permutation positions0 and12. Their three facts
occupy binding slots3/11/19 (chainA) and5/13/21 (chainB). Both arrangement cases:
A forward(order0,2,1),B backward(order0,1,2), then swap orientations. Remaining18
facts are uniformly reordered to make donor and recipient serializations with
identical full maps, six chain facts, selected source/destination identities,
positions, lag and query answers. No context/donor selection by model outcome.
The two selected routes terminate at different binding pairs. All eight query
forks (A/B × hop0..3) share each serialized binding context.

For route j, compute its exact additive L2 output contribution

 d_j(t)=scale * O_h sum_s 1[(t,s) in J_j×S_j] A_h(t,s) V_h(s),

with h=1 for forward and2 for backward, original RMS/RoPE/weights and four pair
cells. The native background remains live. Extraction here is only an explicit
native contribution; original weights and donor-prefix computation are retained
and charged. Donor d_A,d_B are computed once from its A/hop0 input, then reused
across all eight recipient query forks. They are intervention references, never
claimed as independent execution from recipient tokens.

Arms: native; removeA,removeB,removeBoth; ownRestoreBoth; donorRestoreA,
donorRestoreB,donorRestoreBoth. Restoration adds the selected d at the L2 output
after zeroing those same native score edges, then recomputes all downstream RMS
and nonlinear readers. No native source ports, downstream activations, or output
logits are patched. Additive L2 outputs permit exact joint own restoration.

A instrument: structural disjointness/map identity/causal masks, planted explicit
write vs native-minus-cut, nonzero and joint self replay<=1e-9; trained ownRestoreBoth
all-logit max<=1e-9; donor contribution unchanged across query forks<=1e-9; all
finite; native hop3 accuracy>=.8 in each population/arrangement/query group.
B selective causal use: removing each route loses>=.25 gold probability on its own
hop3 query, abs loss<=.10 on the other hop3 query and every lower hop group, each
population/arrangement. No filtering on native correctness.
C context interchange: each donorRestore arm vs recipient native full29-way
KL mean<=1e-3 andp99<=1e-2 on all positions and query separately, each population/
arrangement. Preserve native mistakes. Accuracy alone is insufficient.
D causal composition/reuse: centered full-logit (restored-cut) vector vs
(native-cut), singles and joint, relative RMS<=.01; absolute<=1e-8 when native
RMS<1e-6. Report all positions and query separately, each population/arrangement.

Null: identical selected local facts/positions do not define a context-independent
native write modulo downstream readers. A valid failure requires explicit extra
contextual dependencies before any quotient/extraction claim; no posthoc donor,
head, coordinate, threshold or fitting sweep. A pass licenses a fresh semantic
edit/independent execution test, not a smaller full model claim. B4 FP64,1800s,
<256MiB per tensor, GPU only through managed queue. No training or calibration.
