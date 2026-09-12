# Regional block: four-port interchange

Use the frozen32V2corrected regional prompts. No fitting. For each source position,
factor the block into gate (actual/reference), shared quadratic parent, two child
linear readings as a bundle, and private query-dependent writers. Pair UK/US
contexts of equal token length; swap each port bundle across the pair at the same
position. Retain recipient native background and all other computations.
Evaluate all16subsets of donor ports, summed across all causal sources, then
recompute actual final MLP/RMS/unembedding/cap. Store scalar regional/unrelated
margins and16write vertices, not16full-vocab arrays.

A: all-recipient write matches saved V2collapsed write<=1e-5relative; all-donor
write matches saved paired donor write<=1e-5relative; finite-grid interaction
reconstruction<=1e-10relative on FP64transforms. Existing CPUport controls pass.
B: whole-block donor transplant shifts regional margins toward donor by mean
>=10%of native cue gap with>=5/8positive pairs in EACHfamily. Define pair transfer
as half of [(mUK-mUK_after)+(mUS_after-mUS)] so swapping the complete cue signal
would equal the native UK-US margin gap.
C: A/B and parent-only transfer>=50%of all-port transfer in EACHfamily; unrelated
contrast meanabs parent-transfer<=.5regional meanabs parent-transfer. This tests
parent dominance, not whether every useful parent must itself encode regional
style. Preserve parent failure even if another port dominates. All16interactions
and each single-port effect are descriptive beyond these registered bars.

7bodyforwards32sequences4-12tokens,16suffix vertices,180secondsmanagedGPU. Pairwise
length/ordering tripwires required. No source-position alignment heuristic or
semantic refit. A pass establishes compiler consistency, not causal abstraction;
B/C remain limited to these templates pending independent-factor validation.
