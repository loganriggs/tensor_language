# Common input blocks of the full composed output

12 September 2026. Weight-only structural screen; no identified circuit.

The full-output rank-32 experiment fails sufficiency. An alternative is to keep many
output directions but separate groups of inputs whose products rarely interact.
This could split one native module into independent operations without assuming
that each operation writes a single token or output direction.

## Object and relation to prior work

Let the MLP16 homogeneous producer be $p(x)$, and let $H$ be its exact
quadratic-coefficient Gram matrix, as derived in
[the producer-metric note](PRODUCER_METRIC_SPECTRAL_V1_MATH.md).
Write $z=H^{-1/2}p(x)$ and transform the MLP17 readers to
$L=L_{17}H^{1/2}$ and $R=R_{17}H^{1/2}$. Then

$$
f_v(x)=z^T S_vz,\qquad
S_v=\sum_j W_{vj} A_j,\qquad
A_j=\tfrac12(l_jr_j^T+r_jl_j^T),\qquad W=U_cD_{17}.
$$

$U_c$ subtracts the mean vocabulary row. This is the entire centered pre-head
output numerator of the pure MLP16/MLP17 path. RMS denominators, other residual
paths, bias and final tanh remain explicit external operations. Centering is a
coefficient convention, not a claim that common pre-tanh shifts are behaviorally
irrelevant after tanh. The norm is the paired coefficient norm, not the fully
symmetrized degree-four norm or a text-distribution metric.

A shared orthogonal projector $P$ separates this input space exactly when

$$
[P,S_v]=0\quad\text{for every }v.
$$

This is simultaneous block diagonalization. The commutant approach is established
in [Maehara and Murota (2011)](https://epubs.siam.org/doi/10.1137/090779966).
Their paper studies a numerical algorithm for the finest common block decomposition.
Our approximate graph search below is a restricted diagnostic, not an implementation
of the full theorem or a guarantee of finest-block recovery.

The repository already contains `toy_consumer_commutant_blocks.py`, and MLP0
rungs340/346 found no convincing unconditioned blocks. See
[that interpretation](MLP0_BILINEAR_STRUCTURE_IDENTIFIABILITY_2026-09-01.md).
The new object is the **full-U producer-weighted two-MLP composition**, not a new
name for that MLP0 test. MLP16/17 dossiers were checked before this proposal.

## Exact all-output contraction

Define $G=W^TW$ and $\Phi(X)=\sum_vS_vXS_v$. For symmetric $X$,

$$
4\Phi(X)=
L^T[G\odot(RXL^T)]R+
L^T[G\odot(RXR^T)]L+
R^T[G\odot(LXL^T)]R+
R^T[G\odot(LXR^T)]L.
$$

Consequently, with $K=\Phi(I)$,

$$
\mathcal L(X)=KX+XK-2\Phi(X),\qquad
\langle X,\mathcal L(X)\rangle_F=\sum_v\|[S_v,X]\|_F^2.
$$

This requires $O(m^2+md+d^2)$ storage and $O(m^2d+md^2)$ work per
application, with $m=4608,d=1152$. It avoids the roughly 534 GB float64
vocabulary-by-input-by-input tensor. The exact kernel has been checked against
explicit signed output forms, with errors below $4\times10^{-16}$ in the CPU control.

For a projector define $a=\operatorname{tr}(PK)$, $t=\operatorname{tr}K$,
and $c=a-\operatorname{tr}(P\Phi(P))$. Then

$$
\text{off-block energy fraction}=\frac{2c}{t},\qquad
\text{normalized cut}=\frac ca+\frac c{t-a}.
$$

The first can be misleading: the executed quiet-coordinate control has only
$8\times10^{-9}$ global leakage but normalized cut approximately one. Almost
all information in that small coordinate crosses the proposed boundary.
We therefore require substantial incident energy on both sides and small
normalized cut. These are algebraic conditions, not semantic independence.

## Registered native pilot

Compare identity input metric and the exact producer metric, retaining all1152
input dimensions and all centered output directions in final scoring. For each
metric, use two fixed seeds,120423 and120424. Generate16 Gaussian output sketches
through a Cholesky factor of $U_c^TU_c$; these are weight sketches, not text samples.
Diagonalize the first sketched symmetric form. In that basis, form the nonnegative
graph $E_{ij}=\sum_s(S_s)_{ij}^2$ for $i\ne j$. Use the second eigenvector of
the symmetric normalized graph Laplacian, split at its median into576/576
directions. Repeat the same optimized graph procedure in a random orthogonal
basis as a matched basis control. Final cut metrics use the exact all-output
kernel above, never the sketch objective alone.

Predictions, fixed before native execution:

- A: numerical identities and Laplacian eigenpair residuals below $10^{-8}$;
  finite metrics and projector ranks576.
- B: both producer-metric seeds have exact normalized cut at most0.1,
  incident energy on each side at least10%, and at least20% smaller normalized
  cut than their matched random-basis controls.
- C: producer-metric partitions agree across seeds with projector overlap at
  least0.9 after allowing the two sides to exchange. Report the identity-metric
  arm under the same metrics; it is not an outcome-selected replacement.

Null: approximate independent half-dimensional blocks are not found by this
specified search. Failure does not exclude unbalanced, nonorthogonal, overlapping,
hierarchical or multi-parent structures, or blocks missed by the sketches/basis.
Success supplies an input-group candidate for frozen extraction and joint-edit
tests; it does not satisfy OOD prediction or selective removal.

Discovery needs eigendecompositions and exact contractions, with a600second
managed GPU budget. No fixed CPU-hours restriction is assumed. A candidate
requires an explicit1152-by1152 basis plus576-dimensional block cores for every
output direction; retaining two dense blocks need not beat the existing CP
implementation. Price the executable factor graph before claiming simplicity.
Do not count a projector-only artifact as a cheaper executable model.
