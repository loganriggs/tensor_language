# Value-coherent attention inputs: one-source and cross-source channels

10September, following the O17-only pullback. This is a new algebraic tool with
executed synthetic controls, not a trained-model structure result.

Let $x_j$ concatenate the normalized current-layer source residual and the
normalized source residual used by the block0 value stream. The native value
mixture is linear in this tuple. For head$h$ define

$$
W_h=[(1-\lambda)V_{17,h},\ \lambda V_{0,h}],\qquad M_h=O_hW_h.
$$

For a fixed query, let $a_{hj}$ be the actual two-QK routing product to source$j$,
including normalization, RoPE and causal masking. No independence or fixed-route
assumption is required for the identities below. The attention write is

$$
y=\sum_{h,j}M_h a_{hj}x_j,\qquad
Z_{hi}=\sum_j a_{hj}x_{ji},\qquad y=\sum_h M_h Z_h.
$$

For one output quadratic$Q_v$ of MLP17 with U folded in, pull it through the
value/output maps:

$$
T_{v,h i,k j}=(M_h^\top Q_v M_k)_{ij},\qquad
y^\top Q_vy=\sum_{h,i,k,j}T_{v,h i,k j}Z_{hi}Z_{kj}.
$$

The tensor is symmetric when the complete$(h,i)$ and$(k,j)$ slots are exchanged.
Define partial head exchange$\tau T_{h i,k j}=T_{k i,h j}$ and

$$
T_+=\frac{T+\tau T}{2},\qquad T_-=\frac{T-\tau T}{2}.
$$

The first piece is separately symmetric in heads and source features; the
second is separately antisymmetric in both. They are orthogonal in coefficient
Frobenius norm. With one source, $Z=a x^\top$ has rank1, so every2×2minor is zero
and $T_-$ evaluates to zero. This is the degree-two rank-one matrix constraint;
the [Segre variety's minor equations](https://arxiv.org/abs/0705.1942) give the
standard algebraic setting. It does not say attention with many sources has rank1.

For multiple sources, Cauchy–Binet gives the executable decomposition

$$
Z_{hi}Z_{kj}-Z_{hj}Z_{ki}
=\sum_{p<q}
(a_{hp}a_{kq}-a_{hq}a_{kp})
(x_{pi}x_{qj}-x_{pj}x_{qi}).
$$

Thus the antisymmetric contribution combines routing contrasts across two
heads/sources with value-feature contrasts across those sources. It can be a
real cross-source operation. Deleting it because it vanishes for one source
would be wrong: two heads independently selecting two coordinate sources give
$Z=I_2$ and determinant1. If all head routing rows are proportional, this channel
vanishes even with several sources; that is an additional explicit condition.

The current control verifies native-shaped value mixing, full pullback and
one-source equality to below4e-16, and an orthogonal energy split. On a fixed
two-source toy, the antisymmetric contribution is live (relative norm0.06564)
while the two-piece reconstruction still agrees to3e-16. The source-pair
determinant formula agrees to6e-17.
[Control receipt](SHARED_SOURCE_ATTENTION_QUADRATIC_V1_CONTROL.json).

There is also an efficient exact coefficient-energy computation. Write
$A_{hk}=O_h^\top Q_vO_k$ and $G_{hk}=W_hW_k^\top$. Then

$$
\|W_h^\top A_{hk}W_k\|_F^2
=\operatorname{tr}(A_{hk}G_{kk}A_{hk}^\top G_{hh}),
$$

$$
\operatorname{tr}[(W_h^\top A_{hk}W_k)^2]
=\operatorname{tr}[(A_{hk}G_{kh})^2].
$$

Half the sum/difference gives the symmetric/antisymmetric source-feature
matrix energy. Sum diagonal head blocks once and off-diagonal ones twice.
This needs only128×128 head blocks, avoiding a20736×20736 lifted matrix for the
9heads and2304-dimensional shared source tuple. The low-rank contraction is
implemented and compared against the dense toy.

Next trained-weight measurement: apply this contraction to the four already
fixed output-energy-ranked varimax quadratics (indices0,1,4,2), reporting each
separately. Compare native shared value coordinates with independent signed
source-coordinate permutations per head, preserving each head's Gram while
scrambling shared-source alignment. This is not a whole-tensor estimate or an
independent circuit: four selected scalar forms, QK gate structure and actual
multi-source usage remain explicit limitations. Do not infer behavioral
importance from coefficient kernel energy. Residual/mixed MLP terms, RMS,
final tanh and the other output components remain required.
