# Mathematical review: exact coordinate blocks versus overlapping arithmetic reuse

21 September 2026, 10:50 UTC. Review due 10:44, performed at the next safe boundary after the requested report rewrite. The full goal remains OOD prediction, extraction, selective manipulation, reuse, simplicity and stable identification. This checkpoint includes a primary-literature search and an executed structural test.

## Current object and what changed

The model has 18 blocks, residual dimension 1152, bilinear width4608 and vocabulary50304. Exact QR of the unembedding reduces the linear output coordinates to1152. The broad last-two-MLP folded contribution remains inaccurate; current local experiments concern six symmetric source forms $Q_o\in\mathbb R^{1152\times1152}$ feeding three selected scalar components, not all MLP outputs.

$$
q_o(z)=z^\top Q_o z,\qquad
\phi_j(z,h)=\left(\frac{h^\top a_j-q_{2j}(z)/2}{s(h)}-\alpha_j\right)
\left(\frac{q_{2j+1}(z)}{s(h)}-\beta_j\right).
$$

Polynomial degree is at most four in independent z/h before the explicit RMS denominator. Centered affine and mean corrections accompany approximated source forms. Both native input ports remain supplied. A common fixed residual writer maps these scalars onward; final RMSNorm/unembedding/softcap are explicit in behavioral tests.

The latest 592-product programs store1342028floats. Matched pair programs use1152products/1340940floats. Dense projection and readout accounting removes the apparent total-multiplication advantage: shared1331088versusbaseline1330560 before common interface costs. Fresh32FineWeb/16code tests yield0absolute failures for allthreegraphs, but10/15/13comparisons against either baseline fail for mixed/isotropic/covariance fitting. The mixed coefficient metric helps FineWeb relative to covariance fitting but hurts code. Neither changes upstream extraction or feature identity.

Input-factor scaling, swaps and permutations are gauges. Tucker input bases and general intermediate linear bases introduce further freedom. Native and covariance-shaped coefficient errors are different metrics; a nonsingular coordinate change does not change exact algebraic decomposability.

## Literature mapping

[Fang, Huang and Huang, Theorem2.8](https://arxiv.org/html/2503.01166v1) relates simultaneous congruence block decompositions of symmetric matrices to orthogonal idempotents in their center

$$
Z(Q)=\{X:Q_oX=X^\top Q_o\ \text{for every }o\}.
$$

Our six quadratic outputs are exactly such a family. A nontrivial idempotent would expose independent groups of learned input coordinates shared by every source output. This is an exact direct-sum notion, stronger than an approximate low-rank fit. The paper's center computation is a linear system in $d^2$unknowns; materializing it here is too expensive. It does not identify semantic units, minimize multiplication count, or solve overlapping arithmetic-DAG discovery. Real reconstruction also matters when intermediate eigenvectors are complex.

[Maehara and Murota](https://optimization-online.org/2009/05/2292/) study simultaneous block diagonalization of matrix *-algebras through numerical linear algebra. That is relevant to invariant-subspace structure, but our allowed input transforms are general congruences, not only orthogonal similarity. Confusing these restrictions could falsely reject simple nonorthogonal programs.

[Grasedyck's hierarchical SVD](https://epubs.siam.org/doi/abs/10.1137/090764189) offers hierarchical low-rank approximation with bounded-rank storage scaling. Its object is a multilinear tensor with a chosen hierarchy; it does not equate coefficient rank with minimal arithmetic cost for repeated inputs or shared DAG nodes. [Balle and Mohri's weighted-automaton method](https://papers.nips.cc/paper/2012/hash/700fdb2ba62d4554dc268c65add4b16e-Abstract.html) uses Hankel structure for string-to-value functions. Our six static quadratic measurements do not provide the required concatenation-indexed observable family, so that realization result does not directly solve this object.

The [user's tensor-similarity paper](https://arxiv.org/html/2605.15183v1) supplies metric-based comparison of functions. It does not guarantee that minimizing a single chosen norm preserves every native intervention or produces unique internal arithmetic nodes. The fresh domain tradeoff makes that distinction operational.

## Executable consequence: a simple-pencil obstruction

We derived a cheaper necessary test rather than solving the full center system. Choose two fixed linear combinations $A=\sum_o a_oQ_o$ and $B=\sum_o b_oQ_o$. If A is invertible and $M=A^{-1}B$ has distinct eigenvalues, any center element X commutes with M and hence is diagonal in its complex eigenbasis V. It must also commute with each $M_o=A^{-1}Q_o$.

Writing $V^{-1}XV=\operatorname{diag}(x_i)$ and $K_o=V^{-1}M_oV$ gives

$$
(x_i-x_j)(K_o)_{ij}=0.
$$

Therefore, connect i,j whenever any corresponding matrix entry is nonzero. A connected graph forces all $x_i$equal, so the center contains only scalars. There is no nontrivial idempotent and no exact direct-sum congruence split. The shortcut costs $O(od^3)$work and $O(od^2)$memory. It requires a regular, simple, adequately conditioned pencil; repeated eigenvalues are **inconclusive**, not a negative result. It is our derived implementation, not a claim to implement the paper's full algorithm.

Five planted block patterns, hidden by nonorthogonal transforms, recover their known block sizes. Independent small dense-center nullity calculations agree. A degenerate-pencil control is rejected. Native execution then takes12.49seconds on CPU with2threads.

All four native screens (two fixed pencils, two congruent geometries) pass instrument checks and give one connected1152-dimensional component at thresholds1e-10,1e-8and1e-6. Eigen residuals are below3.7e-15; solve residuals below1.1e-12. Minimum relative eigengaps exceed8.2e-6. Graph connectivity bottlenecks are.00585–.00731, well above tested thresholds. This is numerical evidence, not an interval certificate; conditioning and entry perturbations prevent interpreting that margin as a certified distance to a decomposable tensor.

Registered outcomes: instrumentPASS, nontrivialsplitFAIL, coordinate-consistencyPASS. Covariance shaping cannot manufacture an exact algebraic split. This result does **not** rule out approximate decompositions, cheap products of overlapping linear forms, output sharing, or deeper DAG reuse.

## Decision, organization and next action

Exact independent-coordinate grouping is not the next promising route for these six outputs. Return to overlapping graph reuse, while preserving actual arithmetic accounting. The next bounded screen will factor the graph's dense input-reader map into shared linear intermediate nodes, refit the product readout at fixed topology, and test component fidelity plus both coefficient geometries. This is a different graph edit from deleting product nodes: it targets the dominant projection arithmetic. It must beat a literal total-operation baseline; fewer product nodes alone no longer suffice.

Rank-limited reader factorizations do not establish semantic identity. If they fail component fidelity at a meaningful arithmetic saving, retain the failure and do not promote a cheaper opaque approximation as a circuit. Successful candidates would still need frozen native interventions, cross-domain checks, and upstream closure.

Organization: the continuation dossier remains authoritative; this analysis and preflight live in direct_tensor_match. The parent explanation index contained stale latest pointers; update its top navigation instead of copying all reports. Queue is empty, managed runner PID8995 remains live, and no competing GPU job was started. Existing source/graph evaluators and exact readout solver will be reused. No new compiler or scoring framework is justified.

The user's two-day weights-first focus overrides hourly alternation. Handoff to later circuit work: use preserved component interfaces for removals and swaps, investigate whether constituent features survive fitting gauges, and close native-input dependencies. Handoff within folding work: search overlapping shared computations rather than assume disjoint coordinate blocks or that a chosen tensor norm determines causal fidelity.
