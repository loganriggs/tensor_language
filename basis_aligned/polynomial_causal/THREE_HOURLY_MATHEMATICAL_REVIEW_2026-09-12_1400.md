# 14:00 mathematical review: close the changing writer state

Recorded 12 September 2026, 14:06 UTC, at the first boundary after the unfinished
compiler check. This advances extraction and composition machinery, not discovery
of a new identified circuit. The four behavioral criteria remain unfinished.

## Decision and executed consequence

The common reader/writer frame unnecessarily forces immutable input readings into
the evolving state. A fixed writer span closes the dynamics even when readers
reach outside it. Use this exact representation for extracted quadratic programs;
do not infer native behavioral fidelity from its execution accuracy.

The existing shared subprogram has 12 parents, 25 mixed products, 17 parent-pair
products and 19 active group writers. Its 56-coordinate common-frame compiler
passes but costs more than its compact DAG. The writer-state compiler uses 19
changing coordinates. Both execute the SAME shared union of the unconverged
spectral fit; private-only terms and original native MLP remainder are excluded.

## Exact object, contraction and proof

Let all writes and biases lie in the columns of a fixed, possibly nonorthogonal
$W\in\mathbb R^{d\times k}$. Block $j$ reads through
$F_j\in\mathbb R^{r_j\times d}$, multiplies selected pairs of readings and writes
their mixtures using $M_j\in\mathbb R^{k\times m_j}$. For arbitrary real initial
state $x_0$, represent the current residual as

$$
h=\alpha x_0+Wa.
$$

Retain immutable $f_j=F_jx_0$, $c=W^\top x_0$, $q=\|x_0\|^2$ and $u=Ux_0$.
Precompute $C_j=F_jW$, $G=W^\top W$ and $H=UW$. Then

$$
F_jh=\alpha f_j+C_ja,\qquad
\|h\|^2=\alpha^2q+2\alpha c^\top a+a^\top Ga.
$$

Residual reentry $h\leftarrow\lambda_jh+\mu_jx_0$ becomes
$a\leftarrow\lambda_ja$, $\alpha\leftarrow\lambda_j\alpha+\mu_j$.
With native $\epsilon=1.1920928955078125\times10^{-7}$,

$$
\rho^2=\|h\|^2/d+\epsilon,\qquad
s=(\alpha f_j+C_ja)/\rho,\qquad
a\leftarrow a+M_j\big(s_{p_t}s_{q_t}\big)_{t=1}^{m_j}+b_j.
$$

Induction proves the state identity after each block: reentry preserves the
affine form and every new write is in $W$. This requires neither readers in
$\operatorname{span}(W)$ nor invertible $G$. Final outputs are exactly

$$
30\tanh\left(\frac{\alpha u+Ha}{30\sqrt{\|h\|^2/d+\epsilon}}\right).
$$

The numerator per block is quadratic; RMS makes the complete map nonpolynomial,
and final tanh is retained. This is an arithmetic DAG with shared norm and reader
nodes, not a dense high-order polynomial expansion. Product deletion zeros the
registered product before mixing; parent deletion zeros every dependent product
once, including shared pairs. Multiple removals propagate through subsequent
norms and blocks. Summing overlapping per-parent removal banks is incorrect.

Changing writer coordinates $W\mapsto WT$, $a\mapsto T^{-1}a$ for invertible $T$
preserves the function with corresponding transformed Gram/cross/mixing tensors.
Thus coordinates are not unique semantic units. No minimal-dimension or stable
identification theorem is claimed. Large cancellation can harm floating-point
norm evaluation; current FP64 tests do not certify arbitrary conditioning.

## Literature mapping and limitations

[Mastrogiuseppe and Ostojic (2018)](https://arxiv.org/abs/1711.09672) connect
structured low-rank recurrent connectivity with low-dimensional dynamics. The
relevant analogy is separating directions that write state from directions that
read it. Their random-plus-structured recurrent-network analysis is not an exact
reduction theorem for this finite, RMS-normalized transformer. We use the direct
induction above instead; it provides no statistical recovery or uniqueness claim.

[CLUE](https://arxiv.org/abs/2004.11961) computes minimal constrained linear
lumpings for polynomial differential equations, preserving specified linear
observables. Our discrete map has RMS and tanh, and its encoding includes the
quadratic initial norm and input-dependent readout. Its minimality theorem
therefore does not apply. The earlier CLUE review remains relevant to polynomial
restrictions, but invoking it cannot certify this representation as minimal.

Tensor-train/hierarchical-Tucker factorization concerns coefficient contractions;
it does not itself ensure closure of changing denominators. Hankel/weighted-
automaton realization would require a finite linear state update under symbols,
which is not the normalized quadratic transition here. These are not competing
exact solutions under the current assumptions. No new theorem from those fields
is claimed. Fixed-writer closure is the cheapest executable consequence today.

Construction of dense cross maps costs $O(\sum_j r_jdk)$, Gram $O(dk^2)$ and
output adapter $O(Vdk)$. Encoding a new input still costs
$O(d(\sum_jr_j+k+V))$. Each block costs
$O(r_jk+k^2+km_j)$ with dense mixing; final full-vocabulary output costs $O(Vk)$.
Input cache and full unembedding are retained and charged. These are arithmetic
operation counts, not measured speedups. Across native layers the union of
writer spaces may become full-dimensional; attention adds token coupling.

## Measurements and literal price

[Shared-component result](SHARED_GRAPH_WRITER_STATE_V1_RESULT.json) uses 72 saved
native pre-MLP inputs and all 50,304 output rows. Baseline, parent4 removal,
parent5 removal and both match direct evaluation: full-logit error at most
$1.57\times10^{-15}$, effect error $1.81\times10^{-13}$, joint-interaction error
$8.56\times10^{-13}$. Interaction norm is 34.38. This is surrogate equivalence on
previously inspected endpoints, not a fresh native/OOD validation.

| Representation | Core scalars | Extra output adapter | Changing coordinates |
|---|---:|---:|---:|
| Compact shared DAG | 64,543 | not compiled here | physical residual |
| Common frame | 71,626 | 2,817,024 | 56 |
| Writer state | 66,395 | 955,776 | 19 |

Both compiled forms retain the common fullU (57,950,208 scalars) and 50,361
initial encoded scalars/input. Writer-state core still exceeds compact DAG;
the improvement is against the common-frame executor, not a universal storage
win. Product indices cost 84 int64 entries, and dependency metadata is retained.

The next CPU consequence was actually executed:
[18-block composition control](WRITER_STATE_COMPOSITION_V1_RESULT.json), using
nonorthogonal writers, unrestricted readers, signed reentry, biases and two
separated product interventions. Three input scales and all four edit arms pass:
state/norm/output error at most $1.12\times10^{-14}$ and joint-interaction error
$7.79\times10^{-13}$. Dropping Gram offdiagonals causes 2.16–65.0% output error,
so the normalization control is live. This is synthetic, not 18 native layers.

## What this changes next

We can now execute proposed shared components and their joint edits without
carrying their original intermediate residual vectors. This resolves a specific
extraction implementation issue. It does not repair the source fit's lack of
convergence, unstable cross-start identities, native behavioral misses, or the
failed recursively pruned regional key generator.

Further compiler variants have lower information value now. The next scientific
decision is whether a weight-derived composed candidate remains predictive when
its producer and consumer are replaced together, with all native remainder costs
explicit. Use the existing component/behavior ledgers before selecting that
candidate; a smaller state alone is insufficient. Next math review due17:06 UTC;
hourly review remains due14:39 UTC.
