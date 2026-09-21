# Three-hour math and literature review — 2026-09-21 01:35 UTC

Performed just ahead of the 01:37 checkpoint. Next checkpoint:04:35UTC. The user-authorized two-day weights-first focus remains active. The last three hours moved from selected output-feature reconstruction to full-path evaluation, shared graph corrections, shared input dictionaries and explicit product selection. None of these establishes semantic circuit identity.

## Mathematical audit: pruning products is a joint-function problem

Write the current folded approximation as

$$
f(n,m)=\sum_k c_k(a_k^\top n)(b_k^\top m).
$$

For separable second moments $M_n,M_m$, the product dictionary Gram is

$$
K_{ij}=(a_i^\top M_n a_j)(b_i^\top M_m b_j).
$$

After choosing product indices $I$, the optimal unrestricted output coefficients in this metric solve

$$
D K_{II}=C K_{:I}.
$$

For a fixed positive-definite output metric the same solution applies, because its left factor cancels from the normal equations. A singular or restricted output metric requires care about unmeasured directions and nonunique solutions. We use a spectral pseudoinverse with an explicit cutoff, and record its retained rank and normal-equation residual.

The current greedy selection uses the output residual and conditional product variance, with rank-one updates of the dictionary Gram projection. It is checked against exhaustive candidate selection on a tiny dense problem. It is not a globally optimal subset search. The individual-energy comparator receives the same exact output refit, preventing refit capacity from being mistaken for a selection advantage.

Three limits remain explicit:

1. The tensor teacher here is the frozen approximate graph. We separately score against original native calibration outputs and test native interventions.
2. Separable moments assume independent input factors in the metric. Actual midpoint/source states are paired and correlated; their empirical product Gram is different.
3. Means are restored separately. Coefficient error of the bilinear portion is not automatically the error of the full affine computation or final normalized logits.

## Primary literature and implications

**Compensating for deletion matters.** Optimal Brain Surgeon uses off-diagonal curvature to assess pruning and compensate through remaining weights, rather than relying only on weight magnitude. Our fixed-product writer loss is exactly quadratic, so we can solve the conditional writer problem directly. This motivates the comparison, but our forward product selection is not the OBS algorithm and inherits none of its empirical results. [Hassibi and Stork, original paper](https://papers.nips.cc/paper/1992/file/303ed4c69846ab36c2904d3ba8573050-Paper.pdf).

**Hierarchy can change expressivity cost.** Cohen, Sharir and Shashua relate particular convolutional arithmetic circuit architectures to CP and hierarchical Tucker representations and analyze depth efficiency. Their architectural assumptions do not prove efficient recovery for our normalized transformer, nor do they identify semantic features. The useful implication is to retain hierarchical and shared candidates instead of treating flat product count as a universal measure of difficulty. [Original COLT paper](https://proceedings.mlr.press/v49/cohen16.html).

**Exact graph rewrites have a distinct role.** Tensat applies equality saturation to tensor computation graph optimization, reducing sensitivity to sequential rewrite order. It is relevant to exact distributive/factoring rewrites in the proposed second stage. Our lossy deletion, approximate merging and fitted corrections must remain outside an equality engine and be scored by reconstruction and native tests. We have not implemented Tensat or general equality saturation in this study. [Original MLSys paper](https://proceedings.mlsys.org/paper_files/paper/2021/hash/cc427d934a7f6c0663e5923f49eba531-Abstract.html).

## Research decisions

Keep exact algebraic rewrites separate from approximate graph edits. Continue charging shared computations once, but include all projection/output coefficients and additions. Preserve the isotropic-versus-data-informed distinction; a favorable covariance metric is not proof of global coefficient simplicity. Report successes and failures against their original targets, especially the selected-four-feature versus full-contribution distinction.

The current product-selection pilot may reduce multiplications but increase output storage unless its exact output subspace is retained. That cost is included. Its CPU results show a small gain over individual-energy selection, not a dramatic structural advantage. Native evidence remains the next discriminator at this review boundary.

For the next research cycle, prioritize actual graph edits that change cost, compare paired-input and separable metrics on the same representation, and test whether features remain identifiable and useful under interventions. Do not substitute low CE or a passed compression threshold for the original circuit objective.
