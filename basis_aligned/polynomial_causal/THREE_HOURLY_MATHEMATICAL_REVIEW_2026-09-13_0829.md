# Mathematical review: shared output subspaces and balanced coefficient geometry

Due 08:29 UTC, completed 13 September 2026 after the executable control. The user's interaction-compression direction remains primary. The decision is to test a different weights-only error geometry at unchanged graph capacity; the ordinary-objective fit and its failed behavioral criteria remain recorded.

## Actual object and graph

The selected final-head/last-MLP mixed numerator is

$$
y_o=\sum_{i=1}^{1152}\sum_{a=1}^{128}T_{oia}z_i h_a,
\qquad T\in\mathbb R^{12\times1152\times128}.
$$

The twelve outputs are six existing British/American token pairs. This is degree two in independent residual and head ports, not a polynomial claim about the entire normalized attention circuit. RMS denominators, the residual route, pristine background and final softcap remain in the native conditional validator. Folding the two input legs produces the symmetrized mixed bilinear expression; the ports have different dimensions and are not interchangeable.

Flatten the coordinate products into rows of $X\in\mathbb R^{147456\times12}$. Our representation assigns each row to one of 32 shared eight-dimensional output spaces:

$$
x_e\simeq Q_{g(e)}c_e,\quad Q_g\in\mathbb R^{12\times8},\quad Q_g^TQ_g=I.
$$

Here $x_e$ is written as a column, while $X$ stores its transpose. Execute

$$
f_{g\ell}(z,h)=\sum_{e=(i,a):g(e)=g}c_{e\ell}z_i h_a,
\qquad \hat y=\sum_gQ_g f_g.
$$

These are sparse bilinear forms with potentially full input rank. There are 256 intermediate scalar forms, 147456 product assignments and up to eight coefficients per product. All native input coordinates remain. Orthogonal changes of basis inside a group and permutations of groups are gauges, not newly identified circuits. Shared writer parameters save storage; distinct group features still require computation. Causal reuse across behaviors remains untested.

FP32 codes and decoders plus uint8 assignments cost 4,878,336 nominal bytes; actual serialized ordinary fit costs 4,880,988 bytes. Charge metadata, working state and execution separately. A grouped removal has precise algebraic semantics but needs native selective-intervention evidence before interpretation.

## Algorithm mapping and alternatives

This is a **union-of-linear-subspaces fit to coefficient vectors**, commonly called K-subspaces. For fixed bases the closest-subspace assignment and projection are exact. For fixed assignments, leading eigenvectors of each group's uncentered second-moment matrix minimize its squared residual. One sweep costs roughly $O(NKdr+Nd^2+Kd^3)$, with $N=147456,d=12,K=32,r=8$. It monotonically decreases the chosen objective in exact arithmetic. Small conditional gain certifies only approximate blockwise stationarity; it does not certify a global optimum, uniqueness, or recovery of circuit nodes.

[Vidal, Ma and Sastry's GPCA](https://arxiv.org/abs/1202.4002) offers an algebraic approach: represent a union through homogeneous polynomials and recover subspace normals from derivatives. Its exact-union algebraic model is not established for these approximate coefficient vectors. At 32 groups in twelve dimensions, a naive degree-32 homogeneous lift has $\binom{43}{11}=5,752,004,349$ monomials. Materializing the corresponding data matrix is infeasible here. Dimensionality reduction or special structure would need a justified recovery argument; the paper is not a turnkey global solver for our current fit.

In tensor terminology these groups are output-low-rank blocks with disjoint coordinate-product support, while allowing unrestricted input matrix rank. General block-term decomposition is broader; LL1's rank-one output blocks impose a different restriction. [Tensorlab's authoritative BTD documentation](https://www.tensorlab.net/doc/btd.html) supplies generic nonlinear least-squares and minimization solvers, and distinguishes special LL1 algorithms. Substituting a generic BTD optimizer would discard our cheap exact support/projection updates unless those constraints were implemented. It supplies no automatic uniqueness or global-recovery certificate for this support-constrained object. The useful match today is conditional PCA plus a better declared metric, not a new general-purpose solver implementation.

## Why rebalance the metric?

The target's paired common component holds 85.22% of squared coefficient energy; differences hold 14.78%. The converged ordinary fit has 2.23% common-component error and 32.00% difference error. Differences account for 97.27% of squared residual. This diagnoses a tradeoff in the objective; it does not prove that this alone causes native behavioral failures.

Let $P_+$ and $P_-$ be orthogonal projectors onto paired sums and differences. Set

$$
E_+=\|XP_+\|_F^2,\quad E_-=\|XP_-\|_F^2,
\quad \gamma=\sqrt{E_+/E_-}=2.40119723,
\quad W=P_++\gamma P_-.
$$

Fit $XW$ with the same subspace algorithm. Its normalized objective is exactly

$$
\frac{\|(X-\hat X)W\|_F^2}{\|XW\|_F^2}
=\frac12\left(\frac{\|(X-\hat X)P_+\|_F^2}{E_+}
+\frac{\|(X-\hat X)P_-\|_F^2}{E_-}\right).
$$

No text, activation covariance or loss gradients enter this metric. However, the paired output scope is task-selected: this is not wholly unsupervised vocabulary discovery.

Decode to original coordinates with $D_g=W^{-1}Q_g$. Store $D_g$ in place of $Q_g$: no new runtime adapter, scalar coefficients or edges are necessary. The decoded bases are nonorthogonal in Euclidean geometry and obey $D_g^TW^2D_g=I$. Do not incorrectly reorthogonalize them during execution without transforming their codes.

## Executed consequence and decision

[The CPU control](interaction_balanced_output_metric_v1.py) uses the actual frozen tensor and saved ordinary fit. It checks the balanced identity, transforms the existing subspaces, makes fresh optimal assignments and one conditional eigenvector update, and folds the inverse transform into the decoder. [Receipt](INTERACTION_BALANCED_OUTPUT_METRIC_V1_CONTROL.json):

- Balanced identity error $6.94\times10^{-18}$; decoded replay error $1.77\times10^{-16}$.
- Squared balanced objective decreases from 0.0471143 after transformed reassignment to 0.0458711 after one update; the old stored fit has balanced error 22.69% before this reassignment.
- Difference error becomes 28.00%, while sum error rises to 11.56% and total error to 15.16%. This is a real tradeoff, not adoption.
- 1.10 CPU seconds; nominal program size unchanged. No convergence or native-behavior claim.

The next informative comparison is a bounded converged balanced fit at identical K32/r8 capacity, followed by the existing own-term regional validator. Opposing predictions: protecting differences improves own-term fidelity at unchanged size, or the loss of common components/structural restriction leaves the native failures. Retain total, sum, difference and native errors together; do not replace the earlier 10% total-error bar after seeing results. If behavior remains poor, prefer the existing sparse-entry baseline and change graph structure rather than repeatedly tuning this metric.

This consequence is cheaper and more directly discriminating than building a massive polynomial lift or generic topology solver. Full OOD prediction, extraction, selective manipulation and behavioral reuse remain incomplete.
