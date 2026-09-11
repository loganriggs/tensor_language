# Shared square nodes across LL1 output groups

The implemented model computes a bank of squared linear readers once, then lets several output groups consume them:

$$
z_j=(a_j^\top x)^2,\qquad
q_g=\sum_{k=1}^{r}s_{gk}z_{I_{gk}},\qquad
F=\sum_g c_g q_g.
$$

The integer matrix I describes the graph. Several groups can refer to the same square node. Unlike independent LL1 blocks, the reader bank is optimized jointly with these ties enforced. All output vectors are eliminated by the existing exact conditional least-squares solve, with the same whole-group energy penalty. Gradients for reader slots are summed back into their shared bank rows. This is a restricted arithmetic DAG, not a general hierarchy learner.

The small dense control passes: gradient error 1.01e-15, directional finite difference 2.43e-11, and executable graph replay 4.24e-16. [Kernel](shared_square_ll1_v1.py), [control](SHARED_SQUARE_LL1_V1_CONTROL.json).

## Native proposal result

Before looking for repeated square directions, each fitted LL1 quadratic is converted to its signed eigendecomposition using QR and an eigendecomposition of the small core. This preserves the function while removing arbitrary internal factor choices. Repeated or nearly repeated eigenvalues still make individual axes ambiguous.

The tested proposal rule groups axes only when every pair within a cluster has absolute cosine at least 0.995, 0.999, or 0.9999. Its representative is an energy-weighted leading direction; slot coefficients are projected onto that square. All thresholds were fixed before inspection. Proposals are scored by exact change to the frozen surrogate, normalized by whole native tensor energy.

Canonicalization replays to 2.4e-15. The spectral start merges no readers at any threshold. The native start merges one reader at the first two thresholds and none at 0.9999. Its saving is 1,152 of 1,254,400 coefficients, or 0.0918%, while squared function change is 6.83e-6 of native energy. The registered 5% saving and 16-node reduction criterion fails for both starts. No native optimization was queued from these proposals. [All results](SHARED_SQUARE_LL1_NATIVE_V1_PREFLIGHT.json).

Our generic graph layout also adds 1,024 integer indices, exceeding the tiny saving in this case. That is a price of this encoding, not a lower bound: a sparse exception/alias encoding could avoid most of those indices. The original 5% saving criterion would still fail.

## Why the miss does not rule out shared linear computation

Consider two output functions

$$
F_1(x)=uv,\qquad F_2(x)=uw.
$$

Here u, v, and w are orthogonal scalar input coordinates. An explicit graph reads three values, computes two products, and reuses u. Separate signed eigendecompositions instead use

$$
uv=\frac12\left[\left(\frac{u+v}{\sqrt2}\right)^2-
\left(\frac{u-v}{\sqrt2}\right)^2\right],
$$

with the analogous expression for uw. There are four square directions, and every cross-group direction cosine has magnitude 0.5. The 0.995 merge rule detects no sharing even though the shared parent is explicit and exact. The executed coefficient and graph identities pass below 5e-16. [Counterexample](SHARED_SQUARE_LL1_MIXED_PARENT_V1_CONTROL.json).

Thus near-identical eigen-square merging is incomplete. Shared **linear** parents can be hidden in different combinations of eigenvectors. The next proposal method should use shared input subspaces or mixed products and allow refitting the bases. The earlier pairwise shared-parent/arrowhead graph already implements one such move; scaling that kind of proposal is distinct from merely relaxing a square-cosine threshold. No conclusion about absent native DAG structure follows from this preflight.
