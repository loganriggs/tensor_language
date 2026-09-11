# Sparse token groups with single-product input functions

Authority: user weights-first search, bilinear handoff and unembedding notes.
Prior turn: unrestricted512-function dictionary converged; strongest functions
were dense. Native free-product fits used dense output writers. This combines
the previously separate restrictions, without text/data fitting.

For all50,304centered token functions scaled by their global RMS coefficient norm:

$$
\min_{A,H}\frac{1}{2V}\sum_v\left\|Q_v-\sum_{j=1}^{512}A_{vj}H_j\right\|_F^2
+\frac{\lambda}{V}\|A\|_1,
\qquad H_j=\operatorname{sym}(p_jq_j^\top),\quad\|H_j\|_F=1.
$$

The exact vocabulary-common function remains separate and charged. Norm/tanh,
bias and residual background are not replaced by this quadratic approximation.
No restriction confines atoms to the native output-function span; this matters
because low-rank atoms can combine to represent higher-rank native functions.

## Exact update and relationship to literature

With fixed codes, define the residual correlation excluding atomj:

$$
R_j=\sum_v A_{vj}Q_v-\sum_{k\ne j}(A^\top A)_{jk}H_k.
$$

The atom update maximizes $\langle R_j,H_j\rangle$. Keep the largest positive
and most negative eigenvalues of the symmetricR_j and normalize their joint
Frobenius norm to1. This is the global fixed-code conditional optimum under
single-real-product inertia, including rank-one semidefinite endpoints.
The objective reduction is exactly $\langle R_j,H_j^{new}-H_j^{old}\rangle/V$.
All512updates are sequential. For fixed atoms, solve the convex L1 code problem
with the previously controlled accelerated proximal routine, at most2000steps,
stationarity tolerance1e-6. Store exact conditional Gram/cross matrices.

The close primary reference is
[Ravishankar et al., Efficient Learning of Dictionaries with Low-Rank Atoms](https://web.eecs.umich.edu/~fessler/papers/lists/files/proc/17/web/ravishankar-16-elo.pdf).
Their low-rank matrix-atom model provides a block-coordinate template with exact
updates. Here samples are token quadratic coefficient matrices, not image patches.
We use real symmetric single-product inertia and L1 codes, whereas their displayed
formulation uses generic matrix rank, L0 codes and coefficient bounds. Their
critical-point theorem is therefore not asserted for this adaptation. Exact
conditional descent is derived and tested directly; joint global recovery is not.

## Frozen initialization, budget and measurements

Initialize all512atoms by projecting the previous debiased dictionary functions
to their best real product, then unit normalize. Multiply initial codes by the
projection norm to preserve that truncated approximation. Freeze lambda as the
median ninth-largest absolute target/atom correlation after projection; then
solve codes. This recalibration uses weights only and is explicit, not the old
unrestricted dictionary's penalty. No token-name selection.

Fit540seconds after initialization, allowing a whole sweep to finish. Save each
completed sweep in ephemeral shared memory, durable history/checksum in JSON.
Expiry is optimization_unfinished, not a negative about the representation.
Continuation can reuse the full codes/readers/penalty state.

A: original FP64 direct8token-matrix/implicit-loss agreement, energy replay,
unit-atom norms and maximum outer objective increase all<=1e-8.
B: two consecutive frozen-point checks with sum of all512exact conditional
atom improvements/V<=1e-8 and code stationarity<=1e-4. Run these audits when
outer improvement<=1e-5; prior per-coordinate convergence cannot substitute.
This is coordinate convergence, not a global optimum or unique identification.
C: afterA/B, centered capture>=.10 and medianactive<=16. This is a modest coverage
screen for an explicit input computation, not the four-property success criterion.
D: afterA/B, median per-token support Jaccard with previous debiased dictionary
>=.25. This tests continuity along initialized atom indices, not canonical matching.
Also report signed writer cosines and all nonzero links.

Price:512products,2*512*1152input-reader coefficients, sparse output coefficients
plus their indices, retained common function and background. The old opaque
1152-function basis is not required to execute these new atoms.

Null/red-team: a converged miss rejects only this fixed budget, penalty and start.
Before broader interpretation test shrinkage/support sensitivity, an independent
start, and small multi-product atoms as a distinct relaxation. Unconverged fits
require continuation or a diagnosed numerical repair. Exact dense CPU controls
cover signed projection, coefficient objective, atom descent and residual gap.
