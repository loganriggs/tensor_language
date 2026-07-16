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

## The behavioral-Lloyd pilot (E-5): repairs, doesn't transcend

Logan approved the strongest instantiation: gradient-scored assignment refinement
against the binding metric itself (bottom 12 streams, k=64, first-order move scores
from backprop through the PATCHED model, 2%-damped moves, revert-and-halve trust
region, held-out audits). Result:

- Walked its starting partition from +0.142 to **+0.103** (late-region +0.106 — no
  overfit), i.e. exactly to the GOOD END of the L2 seed distribution (+0.103…+0.167)
  — and no further. Predicted gains decayed (−9.5 → −2.0) with half the steps
  reverting: the gradient signal converges INTO the L2-good basin, not past it.
- Mass moves (10% of rows) reliably backfire (+0.109 → +0.147): the first-order
  linearization only survives small steps — the trust-region protocol is mandatory.
- Discovery en route: the "seed floor" is not seeds — identical-seed runs differ
  (+0.109 vs +0.142) because GPU-atomic index_add makes k-means itself
  non-deterministic. Partition variance is chaotic, not just stochastic.

**E-arc verdict, final:** the backward objective — in proxy form (Fisher, direct-U)
AND in direct behavioral form — cannot beat the best activation-space partition.
Behavioral refinement is useful as a REPAIR mechanism (turns an unlucky partition
into a best-tier one without seed lottery), not as a better optimum. The stream
tables' content is token identity plus noise the downstream stack filters; activation
geometry was the right metric all along. Logan's conjecture is answered for THIS
object; it may still hold for objects whose errors are consumed adversarially
(none identified in this model so far).

Bits accounting (convention set tick 56: structural bits and estimation tokens side by
side, never mixed): every arm above = 34 tables × (k·1024 atom floats @32b + V·log₂k
index bits); estimation data = 524k tokens (early pile-10k slice) for the tables,
+65k tokens (96 seqs) for the Fisher variant.

Files: `../e1_backward_vq.py/.json`, `../e2_unembed_vq.py/.json`,
`../e2b_stability.py/.json`, `../e2c_l2seeds.py/.json`, `../stream_fisher.pt`.
