# Three-hour mathematical review — 2026-09-02 11:14 UTC

## Goal and current boundary

The target is a smaller executable account of bilin18 whose parts correspond to computations: what they read, what
operation they perform, what they write, and which later computation uses the write. The parts must predict held-out
and shifted inputs, compose when installed together, and support selective removal or editing. Storage and compute are
prices on an identified program; low rank, quantization, reconstruction, or CE alone do not identify its parts.

Rung480 found reproducible, gauge-safe sensitivity directions inside the continuous attention0 approximation, but
their 32-circuit labels did not transfer across document/source views. Rung481 then applied the exact MLP0
`T/C/I/S` decomposition. `T` and `I` were the two largest branch effects and were fairly similar within each half,
but the 62 member-minus-control circuit averages were not stable or selective enough to decide whether they are one
variable. The next mathematical object should therefore be the actual immediate consumer maps, not another rank or
circuit-label fit.

## Exact object

For one token sequence `x`, let `w0(x) in R^(256 x 1152)` be MLP0's write and let

`v_b(x) in R^(256 x 1152),  b in {T,C,I,S}`

be one exact branch contribution from rung401. Removing branch `b` with strength `alpha` means

`w0_b(alpha;x) = w0(x) - alpha v_b(x)`.

The three immediate observed consumer maps are:

- `F_A(w0)`: attention1's 1,152-dimensional write at every position;
- `F_D(w0)`: MLP1's write when attention1's write is restored to its native value, isolating MLP1's direct reading
  of the changed incoming residual stream; and
- `F_M(w0)`: MLP1's write when attention1 recomputes normally, measuring the complete attention1-to-MLP1 path.

For consumer `c`, branch `b`, and sequence `x`, the tangent response is

`r_c,b(x) = d/dalpha F_c(w0_b(alpha;x)) at alpha=0
          = - J_c(w0(x)) v_b(x)`.

The finite physical response is

`q_c,b(x) = F_c(w0(x)-v_b(x)) - F_c(w0(x))`.

The joint reader is the stacked map `F=(F_A,F_D,F_M)`. Its Jacobian has three output blocks. Locally unobservable MLP0
directions are the intersection of their kernels. Two branch directions are operationally the same up to scale on
the tested data when `J v_I = alpha J v_T` for one fixed scalar that transfers to new documents. They are
consumer-specific when this proportionality holds for one output block but fails for another.

The measured tensor is the branch Gram tensor

`G[h,c,b,d] = sum over documents, positions, and output coordinates of r_c,b * r_c,d`,

with dimensions `2 x 3 x 4 x 4` for document half, consumer, branch, and branch. An analogous tensor is accumulated
for the full removals `q`. It is only a 4-by-4 Gram per consumer because no native hidden coordinate is being selected;
the complete 1,152-dimensional outputs enter the contraction. Orthogonal changes of output coordinates leave `G`
unchanged.

## Contraction graph and degree

MLP0 is exactly bilinear after normalization:

`w0 = Down0((Left0 z0) elementwise-multiplied-by (Right0 z0)) + bias0`.

Its output enters the block1 residual mixture. Attention1 forms normalized Q and K vectors, contracts every query with
earlier keys, squares and normalizes the scores, contracts them with values, and applies its output matrix. MLP1 then
applies another bilinear map after RMS normalization. Thus `J_c v_b` is computed by differentiating the actual
contraction graph through these operations. Because RMS normalization and attention-score normalization divide by
input-dependent norms and row sums, this complete consumer map is a smooth rational map on the observed inputs, not a
fixed-degree polynomial tensor. Calling it merely a higher-order tensor decomposition would lose that fact.

Parameters are tied across all 256 positions in the usual transformer way. MLP hidden-product rescalings and
permutations are gauge freedoms of a factorization; attention head coordinates also admit compatible internal basis
changes. None changes the observable maps `F_c` or the response Gram tensor. The approximation norm is the Frobenius
norm over documents, positions, and all 1,152 output coordinates. The experiment saves no parameters and stores no
raw activations; it prices model/prefix evaluations and stores only contracted sums.

## What existing mathematics gives exactly

The constant-rank theorem says that a smooth map of locally constant rank has coordinates in which it is a projection
onto its observable coordinates. Consequently, nearby level sets of `F` are smooth fibers and `ker J_F` is their
tangent space. This gives an exact local meaning to the proposed quotient: divide MLP0 perturbations by directions
that the chosen downstream consumers cannot distinguish. See John M. Lee, *Introduction to Smooth Manifolds*,
[Springer, DOI 10.1007/978-1-4419-9982-5](https://link.springer.com/book/10.1007/978-1-4419-9982-5).

Hermann and Krener's nonlinear observability theory similarly factors states by equality of their possible outputs
and uses differentials of observations to test local distinguishability; under regularity assumptions the quotient
inherits the same input-output behavior. See R. Hermann and A. J. Krener,
[“Nonlinear Controllability and Observability,” IEEE TAC 22(5), 1977](https://www.math.ucdavis.edu/~krener/1-25/10.IEEETAC77.pdf),
DOI 10.1109/TAC.1977.1101601.

The mapping is useful but limited. Bilin18 here is a finite feed-forward computation, not the controlled dynamical
system in Hermann--Krener, so Lie derivatives over control trajectories are not needed. The constant-rank theorem is
local and assumes rank is constant in a neighborhood. It does not guarantee that `T` or `I` follows one global fiber,
that the relation survives OOD text, or that an observable coordinate is semantically simple. A tangent equality can
also fail for the finite removal `alpha=1`. Those violations are exactly what the empirical gates must test.

## Executable consequence

Rung483 will compute exact forward-mode Jacobian-vector products through the three consumer maps for all four MLP0
branches. It will also make the complete branch removals and all six two-branch removals. It will:

1. verify the automatic derivative against a symmetric finite difference;
2. require the tangent to predict the complete physical removal before using it as a reader description;
3. distinguish three preregistered outcomes for `T` and `I`: one shared variable at all consumers, two distinct
   variables at all consumers, or a consumer-specific relation;
4. fit any proportionality constant only on the first 250 documents and test it on the second 250;
5. compare same-position agreement with fixed position-shuffle controls; and
6. open documents500:1000 only if exactly one relation is identified on documents0:500.

This is a direct consequence of the observable-quotient mathematics. A null would say that first-order immediate
readers are not sufficient to identify the MLP0 branches; it would not invite a lower-rank refit. The next distinct
object would then be task-conditioned reader functionals or finite behavioral interchange, not another compression
sweep.

## Decision

The local observability quotient is cheaper and more directly tied to circuit boundaries than fitting another
coupled tensor factorization. It can group `T` and `I` across their different exact input formulas, split them by
consumer, or falsify both descriptions. Proceed with rung483 now. The theorem supplies the correct local object and
its gauge-free meaning; held-out and finite interventions remain necessary for circuit identification.
