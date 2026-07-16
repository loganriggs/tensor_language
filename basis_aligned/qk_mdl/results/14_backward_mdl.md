# Method E: backward (unembedding-relative) MDL — a careful null

Logan's conjecture: optimizing the decomposition relative to the UNEMBEDDING (what
downstream consumes) should find a different optimum than the forward, embedding-
relative metric. Tested on the flagship's stream-table quantization (windowed-D W=4
harness, where table coarseness has real ΔCE cost).

## Instantiations tested

1. **Empirical Fisher** (E-1): k-means assignments in the space whitened by diagonal
   E[(∂Loss/∂stream)²] (backprop through the live model, 96 sequences).
2. **Direct-U sketch** (E-2/3): assignments under the quadratic form M'M with
   M = JL-projection · unembedding — rows compared by their logit-space image.
   Sketch widths 512 and 2048, two k-means seeds each; plus an L2 seed control.

Centroid rule identical everywhere (mean of members in activation space); only the
partition changes. ΔCE at T=512, all untrained:

| k=64 arm | ΔCE | k=256 arm | ΔCE |
|---|---|---|---|
| L2, 3 seeds | +0.103 / +0.139 / +0.167 | L2 | +0.104 |
| Fisher | +0.171 | Fisher | +0.116 |
| unembed, 3 variants | +0.124 / +0.125 / +0.150 | unembed, 3 variants | +0.131 / +0.139 / +0.172 |

## Conclusion

**No resolvable advantage for either backward metric.** At k=64, within-metric seed
variance (±0.03) exceeds the between-metric difference (means: L2 ~0.137, unembed
~0.133 — indistinguishable). At k=256 the L2 partition looks better (+0.104 vs
+0.131…+0.172) though the L2 side is single-seed. An intermediate positive (E-2's
k=64 "crossover") was retracted after the stability check — a lucky draw.

Why the null is coherent with everything else in the program: the stream tables'
quantization error behaves like NOISE that the downstream model filters (vq1024 is
free; denoising H5's content helps; CE-polish of table values buys nothing). When
error acts as filtered noise, preserving activation-space geometry is already the
right objective, and output-relevance weighting — which can only see the direct
linear path to the logits — has nothing to add. A backward metric should matter
where errors are consumed adversarially (few atoms, systematic bias), but k-means
seed variance dominates precisely there.

**Untested, strongest instantiation** (needs training budget — Logan's call):
CE-refined assignments — behavioral Lloyd iterations where rows move between clusters
based on measured ΔCE, not any proxy metric. This optimizes the discrete structure
directly against the binding metric and is the only version that can see through the
nonlinear downstream paths.

Bits accounting (convention set tick 56: structural bits and estimation tokens side by
side, never mixed): every arm above = 34 tables × (k·1024 atom floats @32b + V·log₂k
index bits); estimation data = 524k tokens (early pile-10k slice) for the tables,
+65k tokens (96 seqs) for the Fisher variant.

Files: `../e1_backward_vq.py/.json`, `../e2_unembed_vq.py/.json`,
`../e2b_stability.py/.json`, `../e2c_l2seeds.py/.json`, `../stream_fisher.pt`.
