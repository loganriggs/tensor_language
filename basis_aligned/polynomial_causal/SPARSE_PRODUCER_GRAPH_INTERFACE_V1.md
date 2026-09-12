# Executable sparse producer graph

This is the exact execution/interface for a fitted interaction graph, not a
claim that the current graph is a circuit. The full-frame fit is still running.

Let $t(x)=(L_{16}x)\odot(R_{16}x)$, $p=\lambda D_{16}t(x)$,
and let $Q$ be the learned orthogonal frame in the producer metric $H$.
Its shared scalar nodes are

$$
z=Q^\top H^{-1/2}p=C^\top t(x),
\qquad C=\lambda D_{16}^\top H^{-1/2}Q.
$$

Each node is a quadratic function of the normalized MLP16 input. For the
selected unordered edge set $E$, the extracted partial write is

$$
\widehat w(x)=\frac{1}{\rho(y)^2}
\sum_{(i,j)\in E}d_{ij}\,s_{ij}\,z_i z_j,
\qquad s_{ii}=1,\quad s_{ij}=\sqrt2\ (i\ne j).
$$

The square-root factor is required by the orthonormal symmetric coefficient
basis; it is not a fitted scale. $d_{ij}$ is a residual-space writer, obtained
by contracting the actual MLP17 Down map with its two reader products. The
complete edge set exactly reconstructs the pure producer/producer numerator.
MLP biases, other source-pair terms and the direct residual remain outside it.

For a local graph intervention, removing node $i$ deletes its incident edges
while holding the background and native denominator fixed. For two nodes,

$$
\Delta_{\{i,j\}}=\Delta_i+\Delta_j-\Delta_{i\cap j}.
$$

Here the last term is the write of edges incident to both nodes. Summing two
single-node removals double-counts that interaction. This identity concerns
residual writes: CE and capped logits are nonlinear and must be evaluated on
the jointly edited state. A global upstream input ablation also changes other
paths and normalization, so it is a different intervention.

[CPU execution control](SPARSE_PRODUCER_GRAPH_EXECUTE_V1_CONTROL.json) passes:
reader replay3.59e-16, complete quartic replay7.28e-16, joint-removal identity
2.10e-16. The shared-edge effect is nonzero, so the overlap check is live.
[Implementation](sparse_producer_graph_execute_v1.py) reuses the existing
producer-reader folding formula and the symmetric coefficient convention.

## Correction: centering is a metric choice, not an exact preserved channel

The frozen sparse-frame preregistrations say that the common output channel
remains separate native background. The implemented full-residual graph does
**not** by itself enforce that statement. With vocabulary mean $\mu$,

$$
U=U_c+\mathbf1\mu^\top,\qquad
\|U\delta\|^2=\|U_c\delta\|^2+V(\mu^\top\delta)^2.
$$

The fit uses $U_c^\top U_c$, which is positive definite in these runs.
Inverting its Cholesky factor recovers residual-space writers; it does not
project their mean-token contribution away or preserve that contribution
exactly. The frozen implementation and predictions remain unchanged. Interpret
the objective as centered coefficient error, and validate the resulting physical
write through the complete native U, RMS and tanh. No exact common-output
preservation claim is supported by this fitting metric alone. In particular,
common pre-tanh shifts are not behaviorally irrelevant after logit capping.

For the4096edge/full1152node program, storing folded readers, native L16/R16,
and residual writers costs20,643,840 floats plus8192edge indices. Alternatively,
retaining native D16 plus the physical reader map costs21,970,944 floats. This
is a conditional pure-path price; full-model background and U are additional.
No native L17 reader matrices are needed after its edge writers are compiled.
