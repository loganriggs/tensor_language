# Mathematical review: mixed quadratic graphs and arithmetic reuse

12 September 2026, 02:00 UTC. Previous review: 11 September22:56. The user-directed objective remains weight-first discovery with OOD prediction, extraction, selective removal, and composition/reuse. No new circuit meets these requirements.

The completed comparisons show a material restriction in the present approximation family: all products of the learned quadratics capture55.76% more coefficient energy than separate squares. But the32-edge greedy graph captures only3.49% more, and neither passes native-effect fidelity. The next test should optimize readers for a mixed graph, rather than add interactions to readers optimized for squares. The kernel is implemented and its synthetic gradient check passes; native pricing is submitted through lane1.

## Actual mathematical object

The input is $x\in\mathbb R^{1152}$ at the normalized MLP16 boundary. Define the bias-free producer and downstream quadratic map by

$$
P(x)=\lambda D_{16}[(L_{16}x)\odot(R_{16}x)],\qquad
T_m(x)=P(x)^TS_mP(x),\quad m=1,2.
$$

Both native product widths are4608. The two fixed output directions $W\in\mathbb R^{1152\times2}$ obey $W^TG_cW=I$, where $G_c$ is the centered full-unembedding metric. This uses full residual inputs but only a frozen two-output group. It is not a full-U factorization or a closed model. Native final-MLP normalization, background, bias, final RMS and capped logits remain external and are retained for validation.

Each $T_m$ is a fully symmetric order-four coefficient tensor. The approximation introduces32 shared quadratics

$$
q_j(x)=x^TQ_jx,\qquad Q_j=B_j\operatorname{diag}(\nu_j)B_j^T,
\quad B_j^TB_j=I_{16},\quad\|\nu_j\|_2=1,
$$

and a graph $E$ of products:

$$
\widehat T_m(x)=\sum_{(i,j)\in E}A_{ij,m}q_i(x)q_j(x).
$$

The old family fixes $E=\{(j,j)\}$. The new fixed graph has32 edges,28 of them cross products, selected previously using coefficients only. Both have32 quadratic intermediates,512 input squares and32 outer products. Reader/writer storage is592704 floats plus the mixed graph's64 endpoint indices. The full528-edge core stores593696 floats plus1056 endpoint indices. Opaque boundary dependencies are additional, not free.

The discovery norm is Frobenius error of fully symmetric coefficient tensors over formal inputs. It is not a same-input Gaussian moment norm or a language-data norm. All four input slots are tied in actual evaluation. Gauge freedoms include signed eigenvector permutations, inner rotations at repeated eigenvalues, node exchange with matching edges, and simultaneous re-encoding of quadratics and core when rank constraints permit it. Those gauges preclude interpreting an arbitrary coordinate as an identified circuit.

## Exact restriction and derivative

For fixed readers and graph, output coefficients solve linear least squares. If $K$ is the edge-feature Gram and $C$ its target correlations, use

$$
A=K^{-1}C,\qquad
J=\operatorname{tr}(A^TKA)-2\operatorname{tr}(A^TC).
$$

This assumes an invertible, sufficiently conditioned Gram; rank loss needs an explicit alternative, not a silently unstable inverse. At the exact linear solution the envelope derivative is

$$
dJ=\operatorname{tr}(A^T(dK)A)-2\operatorname{tr}(A^TdC).
$$

For edges $(i,j)$ and $(k,l)$, exact symmetry gives

$$
K_{ij,kl}=\frac{\operatorname{tr}(Q_iQ_k)\operatorname{tr}(Q_jQ_l)
+\operatorname{tr}(Q_iQ_l)\operatorname{tr}(Q_jQ_k)
+4\operatorname{tr}(Q_iQ_kQ_jQ_l)}{6}.
$$

The selected-edge kernel evaluates only the required32-by32 Gram. Each target correlation uses256 four-slot contractions at rank16, so32 edges require8192 contractions, the same count as32 squares. Equal counts do not prove equal runtime or memory; the managed price test measures both. Its [CPU control](QUARTIC_SELECTED_EDGES_V1_CONTROL.json) matches the full mixed kernel and features exactly, gives tangent finite-difference relative error3.11e-10, and confirms descent. This checks the derivative, not global recovery.

## Existing mathematics: precise applicability

**Separable nonlinear least squares.** [Golub and LeVeque's variable-projection work](https://faculty.washington.edu/rjl/pubs/GolubLeVeque1979/GolubLeVeque1979.pdf), previously read for the proposal, motivates eliminating $A$. Here the linear restriction is small and exact; nonlinear variables are the constrained $B_j,\nu_j$. The formula does not solve graph selection or global nonconvex optimization. Local stationarity can coexist with a large approximation error, as our independent-start control already demonstrates.

**Symmetric matrix pencils.** The existing generic real-pair implementation maps two learned core forms $C_0,C_1\in\mathbb R^{32\times32}$ to scalar and2-by2 blocks by congruence. It assumes a finite, complete generalized eigenbasis and does not implement the general defective/singular canonical form. This is a restriction of the real-pair canonical-form setting of [Lancaster and Rodman (2005)](https://epubs.siam.org/doi/10.1137/S003614450444556X), already documented in the primary pencil note. Setup is cubic in core size, but forming and independently diagonalizing transformed1152-dimensional quadratics is a separate cost. No uniqueness or semantic interpretation follows from generic blocks.

The new [learned-core experiment](LEARNED_QUADRATIC_PENCIL_V1_RESULT.json) obtains22 scalar and5 pair blocks, condition666, and exact native-write replay2.09e-13. Rank16 truncation of transformed readers worsens native error30.62→50.79%, failing its prediction. The original quadratics can instead remain shared parents, with the new quadratics computed as linear combinations. That exact arithmetic DAG preserves the full mixed function, including its failed behavioral fidelity. Thus hierarchy can avoid reader-rank inflation, but generic algebraic hierarchy alone is not circuit discovery.

**Quartic spectral recovery.** [Hopkins, Schramm and Shi (2019)](https://proceedings.mlr.press/v99/hopkins19b.html) give robust order-four decomposition under algebraic nondegeneracy, with runtime $\widetilde O(n^2d^3)$ up to conditioning. Their recovered tensor components are not our rank16 quadratic intermediates. We have not established their assumptions here. A dense flattening spectrum cannot reject a small arithmetic graph: our [new counterexample](QUARTIC_FLATTENING_COMPLEXITY_V1_RESULT.json) has one quadratic square but flattening rank $r(r+1)/2$. Rank16 can therefore give136 matrix modes from one square node.

**Polynomial optimization.** [Nie and Wang](https://arxiv.org/abs/1308.6562) use semidefinite relaxations for best rank-one approximation, mapping symmetric tensors to polynomial optimization on spheres. Their feasible atoms are fourth powers of linear forms in the quartic case; our signed rank-limited quadratics define a different feasible set. A large moment/SOS lift is not currently priced at1152 inputs, and no recovery theorem transfers automatically. Our matrix-free residual-eigenmatrix initializer is a cheaper relaxation, already controlled but not natively fitted; rank truncation and retained-span normalization prevent interpreting it as the exact best atom.

Tensor trains/hierarchical Tucker and generic contraction-width methods remain broader alternatives from prior reviews. They do not immediately solve the newly observed small two-output quadratic-core problem: the original compact producer graph is already known, and the missing evidence is improved nonlinear readers and executable behavioral fidelity. No new theorem or implementation in those families is claimed in this review.

## Decision and falsifiers

1. **First: jointly optimize the selected mixed graph.** Native gradient/price/replay must pass before a bounded fit. Register stationarity separately from gain and preserve the same frozen effects scorer. If a well-optimized graph does not improve extraction fidelity, do not call coefficient improvement a circuit result. Fixed support is a remaining restriction; support exchanges and independent starts are distinct follow-ups.
2. **Next alternative: residual-derived node initialization.** Use the already validated matrix-free operator if the mixed graph remains constrained by its initialization. Explicit eigen residuals and actual conditional atom scores are required. A failed eigenvector truncation does not establish absent arithmetic structure.
3. **Do not repeat unchanged square continuation immediately.** The additional54-minute fit improved coefficients8.96% but worsened native write fidelity and still missed stationarity. This does not prove the square family has converged; it makes another identical continuation less informative than changing the objective's interaction graph.

The circuit-level decision is whether interacting quadratic intermediates yield a more faithful extractable component, followed by frozen manipulation and OOD tests. Pure rank or weight error is insufficient. The concrete continuation is the implemented selected-edge gradient and managed native-price job, not this review alone.
