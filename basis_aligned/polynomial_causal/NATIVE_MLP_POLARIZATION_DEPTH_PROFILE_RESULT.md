# Native MLP polarization depth profile — result

## Outcome

The exact native MLP coefficient slice is full numerical rank at all 18 layers, and
its rank-768 tail changes smoothly through layer 10.  It therefore does **not** explain
the shipped table program's causal rank knee between layers 9 and 10.

For MLP $j$, the computation formed the $1152\times1152$ matrix

$$
A^{(j)}_{e_0}=\frac12D_j\left[
\operatorname{diag}(R_je_0)L_j+\operatorname{diag}(L_je_0)R_j
\right]
$$

from the exact BF16 checkpoint coefficients converted to float64.  It then computed
all singular values.  Every layer has numerical slice rank 1,152 under the same
conservative roundoff bound used in the earlier MLP0--2 certificate.

The relevant quantity for the shipped rank decision is the optimal rank-768 relative
Frobenius tail,

$$
\epsilon_{768}(A)=
\frac{\sqrt{\sum_{k>768}\sigma_k(A)^2}}
     {\sqrt{\sum_k\sigma_k(A)^2}}.
$$

It ranges only from `0.1210` to `0.1325` across all 18 layers.  At the proposed
boundary:

- MLP10 / MLP9 tail ratio: `1.01128`;
- median MLP10--17 / median MLP0--9 tail ratio: `1.05537`.

Both miss the prospectively frozen `1.20` coefficient-knee threshold.  The rank-512
tails are similarly smooth (`0.2838--0.3078`).  MLP0--2 reproduce the earlier
certificate to floating SVD roundoff: decisive quantities differ by at most about
`1.3e-12` relatively.

## Interpretation

The raw native tensors are not algebraically simpler before layer 10 and suddenly
harder afterward, at least not in this deterministic polarization slice.  All of them
need at least 1,152 products for global exact equality, and all discard a similar
fraction of coefficient energy at rank 768.

Therefore the shipped program's layer-10 knee must be conditional on computation:
which residual states natural text reaches, what later components read, and how those
errors combine through RMSNorm, attention, and loss.  This supports consumer-weighted
or reachable-state simplicity and prunes raw coefficient rank as an explanation of
the knee.

This remains a descriptive, weight-only result.  It does not certify natural-text CE,
semantic variables, selective removal, extraction, OOD transport, or the shipped
allocation itself.

Runtime was `11.6703` seconds.  Numerical artifact:
`native_mlp_polarization_depth_profile.json`, SHA-256
`0061545777c81e0589f3dd05c5c82185638ee0540c64cad792b809fc2a43980a`.
