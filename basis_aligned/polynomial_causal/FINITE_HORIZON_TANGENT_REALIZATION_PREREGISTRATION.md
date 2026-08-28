# Preregistration: finite-horizon tangent minimal-realization discriminator

Date: 2026-08-28

Status: CPU mathematics and proof kernel only. This document opens no model or row and
authorizes no GPU measurement until a source-closed matrix-free consumer is separately
committed.

## Exact object

Use the admitted rank640 complete program. At early MLP write interfaces $i=0,1,2$,
inject deterministic natural-covariance-shaped residual directions $u_i$ at one frozen
position per row. Let registered categorical-score/Fisher tests of all later logits be
$y$. The same-forward tangent responses form blocks

$$
H_i=\frac{\partial y}{\partial u_i}.
$$

For a depth cut $k$, horizontally concatenate every upstream response:

$$
\mathcal H_k=[H_0\mid\cdots\mid H_{k-1}].
$$

Every implementation that passes all upstream linearized intervention effects through
a state $z_k\in\mathbb R^r$ must factor $\mathcal H_k=D_kE_k$; therefore
$r\ge\operatorname{rank}(\mathcal H_k)$. Conversely, the SVD supplies the optimal
rank-$r$ factorization in squared response error. This is the finite-depth
operator-Schmidt/cut rank and the time-varying analogue of a Hankel minimal-realization
rank. RMSNorm, residual addition, attention, and later MLPs are inside the measured
Jacobian rather than treated as separable Euclidean modules.

## Frozen pilot

- Program: immutable admitted shared-QK rank640 shell with exact MLPs.
- Fit role: the already authorized skip80 attention-fit rows only.
- Sites: MLP0, MLP1, MLP2; no site selection after observation.
- Inputs: 32 fixed-seed covariance-shaped residual-write directions per site, assigned
  evenly over rows and scored positions.
- Tests: 16 fixed-seed categorical Fisher score probes of all later positions and the
  full 50,304-logit vocabulary. Accumulate only float64 response outer products and
  source/target hashes; graph-bearing tensors may not escape the consumer.
- Replication: split by source document before any spectrum is computed.
- Cut rule: smallest rank retaining 95% squared singular energy, only when the singular
  gap is at least 2 and the rank is strictly below full measured support.

## Gates and consequence beyond reconstruction

1. Primary/replication selected ranks differ by at most two; normalized spectra have
   $L^1$ distance at most 0.10; normalized cut-projector chordal distance is at most
   0.15.
2. Eight typed orthogonal gauge replays preserve every cut spectrum to relative
   $10^{-8}$ and physical projected responses to $10^{-6}$.
3. The selected cut factor predicts heldout single-direction Fisher responses with
   pooled $R^2\ge0.5$ and median relative error at most 25%.
4. Sixteen heldout two-site mixtures have median relative prediction error at most 30%.
5. Replacing discarded directions in the complete rank640 program changes natural-row
   CE by at most 0.005 nat and preserves at least 90% context recovery; retained matched-
   RMS directions must cause at least five times the discarded median response.

Failure of rank stability or nonlinear mixtures falsifies a shared linear state, even
if local reconstruction is good. A pass licenses only a three-site tangent-state
compiler; it does not certify finite edits or the remaining fifteen MLPs until repeated
without rank selection.
