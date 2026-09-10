# Three-hour mathematical review — 10 September22:49 UTC

Review initiated at the due boundary; calculations completed through22:55.

## Decision and actual object

Separate three questions: uniqueness of an exact fitted tensor, uniqueness or
stability of its approximation to native weights, and identification of a
reusable circuit. New executable checks answer parts of the first question;
they do not certify the other two. Next test an independent sparse-frame restart,
then compare functions and graph interfaces rather than raw parameter ordering.

Native MLP17 has inputs x in R^1152, L,R in R^(4608x1152), D in R^(1152x4608),
and U in R^(50304x1152). The folded quadratic is

$$
T_{vij}=\sum_k(UD)_{vk}\operatorname{sym}(l_kr_k^\top)_{ij}.
$$

This is degree two in the normalized MLP input. Down bias, incoming residual,
finalRMS and30tanh(score/30) remain outside this local tensor; the full model
is not a homogeneous polynomial in raw residual inputs. Separate the exact
common-output summand and centered remainder without discarding either.
The discovery norm is coefficient Frobenius over all input pairs and output
rows, not an activation-distribution norm. All current work uses weights only.

Signed squares represent the fitted tensor as [UW,A,A], with256columns in each
factor and A in R^(1152x256). Reader signs and permutations are immaterial,
and rescaling a reader is offset by its writer. The sparse-core representation
instead has an orthogonal128input frame and256quadratic edges. Rotations change
edge sparsity; absent readers and graph symmetries can retain ambiguity.

Prices: squares589824floats. Sparse centered component442368floats+512edge
indices, plus664128independent coefficients for an exact dense common quadratic.
Current full-model background remains native. Neither artifact is an independently
extracted text-to-output circuit. OOD prediction, extraction, selective removal
and compositional reuse remain program-level requirements.

## Exact factor identifiability and constructive recovery

Leurgans–Ross–Abel prove uniqueness when two factor matrices have full column
rank and no two columns of the third are collinear. For [UW,A,A], full column
rank of A and noncollinearity of UW are sufficient. This permits signed,
nonorthogonal input readers. [Primary paper](https://epubs.siam.org/doi/10.1137/0614071).

Kruskal gives the sufficient condition k_A+k_B+k_C>=2r+2, using column
independence ranks, not arbitrary ordinary ranks. Full column rank of all three
factors is a stronger easily checked special case. Our fitted square factors
have numerical singular-value ratios0.09438 for A and0.01899 for W; the native
unembedding is injective on residual-output space in the existing positive-definite
metric checks. Thus the numerical full-column-rank evidence supports768>=514.
This is numerical evidence, not a symbolic rank certificate.
[Original theorem](https://www.sciencedirect.com/science/article/pii/0024379577900696).

For two output mixtures, the reduced input matrices have the form

$$
M_1=R\operatorname{diag}(c_1)R^\top,\quad
M_2=R\operatorname{diag}(c_2)R^\top,
\quad M_2M_1^{-1}=R\operatorname{diag}(c_2/c_1)R^{-1}.
$$

Here A=ER with E orthonormal. Nonzero c_1 entries and distinct ratios recover
columns of R by a matrix eigendecomposition, up to scaling and permutation.
A nonsingular rank-r slice also supplies span(A); our numerical recovery test
uses that fitted input span directly. For compressed slices the eigensolve costs
O(r^3), with O(dr^2+r^2) storage/work terms for lifting, without materializing
Vxdxd. Constructing native compressed mixtures uses the existing4608products.

Three fixed random mixture pairs recovered every fitted reader with minimum
matched cosine>=0.9999999999999976. Polynomial replay error is at floating-point
roundoff. A counterexample with identical output columns permits rotating two
squares without changing their sum, showing why the output condition matters.
[Executed audit](SQUARE_PENCIL_IDENTIFIABILITY_V1_AUDIT.json).

These guarantees do not cover the native approximation: its relative squared
coefficient error is0.90355. Actual native slices projected into the same input
span differ from fitted slices by69–80% relative Frobenius norm. Their matrix
pencils have86.7–92.2%non-real eigenvalues, while the exact independent real-square
model has real ratios. This does not disprove overcomplete squares, richer
multi-output blocks, useful approximate structure or circuits. It says the exact
undercomplete diagonalization assumptions do not describe these native slices.
Robust uniqueness requires sufficiently small error and quantitative conditioning;
we cannot assume that regime here.
[Robust uniqueness research](https://arxiv.org/abs/1304.8087).

Distinct-reader products expand into paired CP terms with duplicate output
columns. The square-model theorem cannot be transferred to that expansion or
to the sparse graph without checking their different degeneracies.

## Closed blocks versus open computational interfaces

Simultaneous orthogonal block diagonalization of symmetric output quadratic
matrices corresponds to reducing subspaces of their generated matrix *-algebra.
General algorithms exploit this algebra through numerical linear algebra; they
solve a closed-block problem, not automatically a circuit-selection problem.
[General algorithm](https://optimization-online.org/2009/05/2292/).

For a proposed orthogonal projector P=EE^T and centered quadratic family Q_v,
let S=sum_v Q_v^2. A cheap exact consequence is

$$
\sum_v\|[Q_v,P]\|_F^2
=2\left[\operatorname{tr}(E^\top SE)
-\sum_v\|E^\top Q_vE\|_F^2\right].
$$

This is the energy of the mixed inside/outside tensor terms. It is zero exactly
when this proposed orthogonal subspace is reducing. The input marginal and
projected core are computed implicitly from L,R,D,U; no collection of50304dense
quadratics is allocated. Finding the entire commutant is a different problem,
with up to d^2unknown matrix entries; the fixed-projector test avoids that solve.
Nonorthogonal congruence blocks require a different metric/pencil treatment.

The native centered tensor splits across the learned128input span as5.882%inside,
24.136%mixed and69.982%outside energy. Its92active readers have4.626%inside and
16.555%mixed energy. The selected surrogate graph has components of sizes46,3,
four2s and35singletons, but its active union has a large native boundary.
Graph disconnectedness therefore does not license dropping native interfaces.
The formula passes an independent dense control and energy closure checks.
[Native boundary audit](SPARSE_CORE_NATIVE_PORTS_V1_AUDIT.json).

This is not a rejection of open reusable circuits: their inputs, outputs and
cross-boundary operations must be specified and retained. It changes the
extraction decision by quantifying interfaces omitted by a closed-block reading.

## What the current optimizer result establishes

Native sparse-frame RCG converged locally in74.30s/865iterations, increasing
centered capture1.401→3.6695%. Numerical and convergence predictions held;
the5%capture prediction failed. Support gap2.05e-6, tangent stationarity8.49e-5,
maximum gradient1.79e-8. This removes local stopping uncertainty for this run,
not initialization dependence or representation restrictions.
[Receipt](SPARSE_CORE_RCG_V1_RESULT.json).

The highest-information next action is an independently initialized run of this
same converged problem, comparing full fitted functions before parameter names.
If equally good runs disagree, candidate units are not yet stable. If they agree,
inspect repeated input/output interfaces and the native cross terms before
FineWeb validation or causal promotion. Orthogonal sparse cores, nonorthogonal
multi-output blocks and coupled path factorizations remain distinct alternatives.
The new exact pencil/port tools are immediately usable; another unmodified
iteration chunk of an already converged fit is lower-value.
