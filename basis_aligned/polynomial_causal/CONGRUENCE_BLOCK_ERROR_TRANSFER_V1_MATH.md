# From a congruence witness to original-coordinate block error

The current native search evaluates the entire family of symmetric token forms
$Q_v$ using

$$
\mathcal L(Z)=2\sum_vQ_v(Q_vZ-Z^\top Q_v),\qquad
\mathcal E(Z)=\langle Z,\mathcal L(Z)\rangle_F.
$$

The operator is self-adjoint and positive semidefinite. Scalar identity is always
in its kernel. Its superoperator trace is

$$
\operatorname{Tr}(\mathcal L)=2(d-1)\sum_v\|Q_v\|_F^2.
$$

Thus, after dividing by total coefficient energy and removing one scalar identity
direction, the exact mean eigenvalue is$2/(d+1)$. The native near-null bar is1%
of this mean, not an unexplained absolute cutoff. This is a screening threshold,
not a theorem that such modes correspond to useful blocks.

## Real block-scalar witness

Suppose$Z=W\Lambda W^{-1}$ where$W$ is invertible and$\Lambda$ is scalar on each
proposed block. Let$\delta$ be the minimum difference between distinct block
labels. Define$C_v=W^\top Q_vW$ and$R_v=Q_vZ-Z^\top Q_v$. Then

$$
W^\top R_vW=C_v\Lambda-\Lambda C_v.
$$

Every off-block entry on the right is multiplied by a label difference of
magnitude at least$\delta$. Consequently,

$$
\|\operatorname{offblock}(C_v)\|_F
\leq\frac{\|W\|_2^2}{\delta}\|R_v\|_F.
$$

Delete those transformed cross-block terms and map back:

$$
\widehat Q_v=W^{-\top}\operatorname{blockdiag}(C_v)W^{-1}.
$$

The original-coordinate error satisfies

$$
\frac{\sum_v\|Q_v-\widehat Q_v\|_F^2}{\sum_v\|Q_v\|_F^2}
\leq\frac{\kappa_2(W)^4}{\delta^2}
\frac{\mathcal E(Z)}{\sum_v\|Q_v\|_F^2}.
$$

This shows why the eigenvalue alone is insufficient. Poor conditioning or tiny
label separation can amplify the error. Rescaling$Z$ and adding scalar identity
changes numerator and separation together, leaving the bound unchanged. The
block-diagonal deletion is a specific approximation, not necessarily the best
original-Frobenius block-core refit for a nonorthogonal$W$.

The CPU control tests a condition-three mixing matrix, exact and noisy block
families, plus witness scaling and scalar shifts. All checks pass. At the larger
noise level the actual relative squared error is0.19475while the bound is25.144:
the bound is conservative and can be uninformative. Candidate promotion must
measure actual original-coordinate error, rather than merely quote the bound.
[Receipt](CONGRUENCE_BLOCK_ERROR_TRANSFER_V1_CONTROL.json),
[implementation](congruence_block_error_transfer_v1.py).

## General real invariant blocks

If$Z=W\operatorname{blockdiag}(D_1,\ldots,D_b)W^{-1}$, cross-block constraints
become$C_{ij}D_j-D_i^\top C_{ij}$. Replace$\delta$ with the smallest singular value
of these Sylvester maps over all cross-block pairs. Complex-conjugate eigenvalues
must stay together when representing a real invariant block. The earlier
nonorthogonal toy demonstrated this necessity. A numerical implementation of the
general Sylvester-separation bound has not been tested here.

This is the algebra underlying the nonorthogonal block formulation of
[Cai and Liu](https://arxiv.org/html/1607.00716v2), with explicit accounting for our
original coefficient metric. A block separation of these quadratic numerators
does not remove coupling through actual RMS denominators, residual paths or tanh.
The four circuit properties therefore remain separate requirements.
