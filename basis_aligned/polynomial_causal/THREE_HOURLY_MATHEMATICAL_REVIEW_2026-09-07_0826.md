# Three-hour mathematical tensor-network review — 2026-09-07 08:26 UTC

## Decision

The weight-level object for comparing reused MLP computations is not a basis of residual vectors and
not a set of hidden units. It is the gauge-invariant subspace of **symmetric quadratic forms** obtained
by contracting an identified response subspace through the exact `Down`, `Left`, and `Right` tensors.
This object can be compared across tasks without materializing a `1152 x 1152 x r` tensor: an implicit
hidden-space Gram kernel gives its principal angles exactly. The immediate executable consequence is a
reusable implementation plus a synthetic identity test, followed by a temporal-versus-is/was
task-specific quadratic-reader overlap experiment.

## Current model and target as explicit objects

The promoted empirical program has one changed cue token, a frozen 48-component source graph spanning
heads and MLPs in layers 0--10, and eight response interfaces

\[
S=\{\mathrm{MLP1,MLP3,MLP4,MLP6,L8H1,L9H1,L9H4,L11H3}\}.
\]

At each site a rank-eight response projector is represented by an orthonormal matrix `Q_s`; its gauge
is `Q_s -> Q_s O` for `O in O(8)`. The four attention response spaces have ambient dimension 128 and
the four MLP response spaces dimension 1152. Attention responses are compiled through exact local
current-value `W_V/W_O` pullbacks. MLP responses are compiled through rank-64 activation-conditioned
input covectors and the native bilinear weights. The clean rank48 forward OOD program has minimum
coordinate projection `.9445`, behavior projection `.8015`, and zero control flips; exact 10% cue-source
noise also passes. Reverse OOD execution is a registered near-null: coordinates and controls pass, but
one cross-fit's behavior projection `.7982-.7998` misses the frozen `.80` bar.

The whole restricted transformer is not a polynomial: RMS normalization introduces inverse square
roots, attention normalization is nonlinear, and later modules compose recurrently. A single MLP at a
fixed normalized input boundary is exactly quadratic. Let `d=1152`, hidden width `h=4608`, response
rank `r=8`, and

\[
z(x)=(Lx)\odot(Rx),\qquad y(x)=D z(x),\qquad c(x)=Q^T y(x),
\]

where `L,R in R^(h x d)`, `D in R^(d x h)`, and `Q in R^(d x r)`. Define

\[
A=D^TQ\in\mathbb R^{h\times r}.
\]

For coordinate `j`,

\[
c_j(x)=\sum_k A_{kj}(l_k^Tx)(r_k^Tx)=x^TB_jx,
\qquad B_j=L^T\operatorname{diag}(A_{:j})R.
\]

Only the symmetric part

\[
S_j=\tfrac12(B_j+B_j^T)
\]

is functionally observable because `x^T(B_j-B_j^T)x=0`. Thus a task response space induces the
quadratic-form subspace `span{vec(S_1),...,vec(S_r)}`. The gauges `Q -> QO` and `A -> AO` merely rotate
this span. Hidden-factor exchange `L<->R` leaves `S_j` unchanged, and reciprocal hidden-unit scaling
also cancels. This is precisely the cross-boundary, within-module object the circuit target asks for.

Allowed inputs are normalized activation states from sealed temporal/is-was constructions. Outputs to
preserve are the eight response coordinates, two physical final-state modes, signed answer effects,
and full-vocabulary control distributions. The present implementation still uses a native background
and 48 native component computations; therefore it is an executable causal interface, not yet a
standalone cheaper model. Empirical basis storage is 40,960 floats per rank-eight split, and the four
rank-64 MLP input bases add 294,912 floats per split, but those counts do not price the retained native
component weights or background state. Literal extraction price remains open.

## Exact implicit contraction

Materializing each `S_j` costs `d^2 r`. Instead define hidden-space matrices

\[
P=LL^T,\quad R_0=RR^T,\quad C=LR^T,
\]

and the positive-semidefinite quadratic-form metric

\[
G=\tfrac12\left[(P\odot R_0)+(C\odot C^T)\right]\in\mathbb R^{h\times h}.
\]

Direct trace expansion gives, for coefficient vectors `a,b in R^h`,

\[
\langle S(a),S(b)\rangle_F=a^TGb.
\]

Consequently two task-specific coefficient matrices `A_1,A_2 in R^(h x r)` have within- and
cross-Gram matrices `A_1^TGA_1`, `A_2^TGA_2`, and `A_1^TGA_2`. Whitening the within-Grams and taking
the singular values of the whitened cross-Gram yields the principal cosines between the two quadratic
reader subspaces. This calculation is exact up to floating-point arithmetic and never constructs a
`d x d x r` tensor. It also exposes null directions: any `a` with `a^TGa=0` represents no scalar
quadratic function even if its hidden coefficient norm is large.

## Theorem/algorithm mapping and violated assumptions

Björck and Golub give the stable SVD computation of principal angles between subspaces; principal
angles are invariant even when principal vectors are nonunique
([Björck & Golub 1973](https://rainbow.ldeo.columbia.edu/~alexeyk/BjoerckGolub1973.pdf)). Here the
Euclidean bases are replaced by bases whitened in the exact `G` metric. The algorithm applies exactly
after quotienting the nullspace of `G`. It yields a finite weight-function comparison, not population
identifiability. Wedin's singular-subspace perturbation theorem would turn cross-fit tensor
perturbations into angle bounds only with an appropriate singular gap and perturbation norm
([Wedin 1972](https://link.springer.com/article/10.1007/BF01932678)); those quantities must be measured
before invoking its guarantee.

The native bilinear MLP tensor has a known CP expression with up to 4608 terms. Kruskal's uniqueness
condition for a three-way rank-`H` decomposition requires the three factor k-ranks to sum to at least
`2H+2` ([Kruskal 1977](https://www.sciencedirect.com/science/article/pii/0024379577900696)). For the
task-contracted `1152 x 1152 x 8` tensor, the sum is at most `1152+1152+8=2312`, far below
`2*4608+2=9218`. Therefore Kruskal cannot identify the 4608 hidden products, and unit-level uniqueness
must not be claimed. The quadratic-form span is meaningful because the native factors are already
known and only their induced function is being compared.

Bilinear realization theory relates Volterra kernels to reachability, observability, and minimal
state-space realizations; shift-operator formulations give finite-dimensional realizability criteria
for genuine bilinear dynamical input-output systems
([Frazho 1998](https://epubs.siam.org/doi/10.1137/0318049)). The local MLP quadratic form is analogous
to a second-order Volterra kernel, but the full transformer violates the required bilinear state
dynamics through normalization, attention, discrete token lookup, and finite-depth residual
composition. No global minimal-realization theorem transfers. Its useful consequence here is only to
separate an observable weight kernel from a particular hidden realization.

## Executable consequence and route comparison

The first implementation consequence is a function that computes `G`, metric Grams, and principal
cosines, with a synthetic check against explicitly materialized symmetric quadratic forms. The real
experiment will fit temporal-only and is/was-only response subspaces at each of the four MLP sites,
contract both through `D^TQ`, and report:

1. cross-fit principal cosines in activation space and in quadratic-function space;
2. the dimension of their stable intersection under a frozen cosine threshold;
3. held-out coefficient prediction through the shared versus task-specific quadratic spans; and
4. causal swaps using the shared span alone and each task-specific complement.

Opposing predictions are decisive. If activation overlap is weak but quadratic-function overlap is
strong and the shared span transfers causally, native output coordinates were gauge-misaligned views
of one reused computation. If both overlaps are weak, the two tasks merely share downstream sites. If
weight overlap is strong but shared-span swaps fail, static weights are an incidence prior rather than
a live shared circuit—the same lesson already seen for raw head scores.

The empirical reverse-support ladder remains cheaper and more immediately causal, so it should run
before the full task-specific tensor capture. But the implicit metric implementation is the best
mathematical consequence and should land now; it converts the user's proposed “shared subspace into
actual weights” step into a gauge-invariant, testable object. Further CP/rank decomposition is lower
information because uniqueness assumptions fail. Global Volterra/minimal-realization work is also
lower information until the nonlinear boundaries are explicitly approximated or removed.

Next mathematical review due around **2026-09-07 11:26 UTC**.
