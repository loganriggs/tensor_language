# Suffix-transport v1 deployed-N write semantics

Status: **prospective, nonauthorizing semantic closure.** This note resolves the
meaning of the frozen preregistration's deployed complement before any fresh role is
loaded. It changes no row, basis, candidate, optimizer, selector, or gate.

For site (l), let (B_l\in\mathbb R^{1152\times64}) have orthonormal columns,
let (m_l^N) be the live write produced by the frozen current-ship surrogate at the
current autoregressive state, and let \(\widehat p_l\) be the executable predicted
code. The student state (P) means exactly

$$
m_l^{P_B[N]}
=m_l^N+\left(\widehat p_l-m_l^NB_l\right)B_l^\top
=m_l^N(I-B_lB_l^\top)+\widehat p_lB_l^\top.
$$

Therefore

$$
m_l^{P_B[N]}B_l=\widehat p_l,
\qquad
m_l^{P_B[N]}(I-B_lB_l^\top)=m_l^N(I-B_lB_l^\top).
$$

The preserved complement is the deployed **N-surrogate** complement, not the
native-original MLP complement. Preserving the latter with zero native calls is
impossible in general: two native functions can have identical (B_l) coordinates
and arbitrary different values in (B_l^\perp).

The fit state written compactly as `P/P/N` is consequently
`P_B0[N0] / P_B1[N1] / N2`. Frozen means frozen surrogate parameters, not cached
activations: every N write must be recomputed at its live same-forward state. A P arm
is a conditional slice correction and never a standalone native-module replacement.
Standalone prices must include the complete deployed N producer; only the incremental
rank-64 correction may be priced conditionally.

## Required runtime provenance

- Student scopes produce exactly one live deployed-N write at MLP0, MLP1, and MLP2,
  and make exactly zero native-original MLP0/1/2 calls.
- `P_B[N]` accepts a one-use typed N-write handle bound to its site, current tensor
  object, forward nonce, and broker issuer. Raw tensors and native-O handles fail.
- Coordinate and autonomous OON teacher scopes use separately licensed native-O
  capabilities and exact native-call ledgers.
- Exact O/E restorations are explicitly O-dependent and carry their native-call
  allowance; they cannot be relabeled N or P.

The pure runtime's typed handle is deliberately nonauthorizing by itself. Only a
source-closed observed bilin18 adapter can prove that the handle was minted by the
frozen ship's live N producer, that N2 also executed, and that the original-call,
outer-forward, return, restoration, and inertness ledgers closed exactly.
