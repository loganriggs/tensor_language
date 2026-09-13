# Share output writes across input products

13September2026. This is a different structural restriction from deleting input readers. Every coordinate input product remains available, but groups of products reuse output directions or output subspaces. It targets the retained head17.2/MLP17 interaction, not the earlier full-vocabulary token-function dictionary.

For $T\in\mathbb R^{12\times1152\times128}$, collect each input product's output weights into $x_e\in\mathbb R^{12}$, where $e=(i,a)$ and there are147,456products. The underlying scalar feature is $z_i h_a$.

## One output direction per product

The first restriction approximates

$$
x_e\approx c_e d_{g(e)},\qquad \|d_g\|=1,\quad g(e)\in\{1,\ldots,K\}.
$$

For fixed directions, choose the largest squared inner product and set $c_e=d_{g(e)}^Tx_e$. For fixed assignments, each direction is the leading eigenvector of its cluster's coefficient Gram matrix. Both steps solve their conditional problems exactly and do not increase coefficient loss. Signed coefficients make the direction an unoriented line; negative writes are allowed.

Execution forms shared bilinear features

$$
f_g(z,h)=\sum_{e=(i,a):g(e)=g}c_e z_i h_a,
\qquad y=\sum_g d_g f_g.
$$

Each $f_g$ can have a high-rank input interaction matrix. It is not one rank-one input product, so this model does not impose the earlier small LL1 input-rank budget. All coordinate products remain charged.

With$K=32$, one FP32 coefficient and one uint8group index per product, plus32twelve-dimensional directions, cost738,816bytes versus7,077,888dense. That attractive price is not meaningful without fidelity. A planted control and exact executable reconstruction pass, but the native fits have roughly60%coefficient error.

Ten starts received20alternations; the best three were refined through300alternations. Total measured CPU time was51.90seconds. Final errors are60.034%,60.053%and60.069%. Assignments still change for60–149products on the last iteration, so the registered exact-assignment convergence condition is **not met**. Last50-step relative squared-loss improvements are0.0035–0.0137%. This is an unfinished optimization result, not a global impossibility result or an adopted compression.

## Relaxation: one small output subspace per product group

Instead let $Q_g\in\mathbb R^{12\times r}$ have orthonormal columns:

$$
x_e\approx Q_{g(e)}c_e,\qquad c_e\in\mathbb R^r.
$$

Assignment maximizes $\|Q_g^Tx_e\|^2$; the conditional optimum for each group is the top$r$eigenvectors of its assigned Gram matrix. The executable features become$r$sparse bilinear forms per group, followed by the common output matrix$Q_g$. This allows more varied writes without projecting away input directions.

The planted two-plane control passes. One actual-weight conditional update, starting from fixed random orthonormal subspaces, gives:

| Group dimension $r$, with $K=32$ | Initial error | After one update | Stored bytes |
|---|---:|---:|---:|
|2|67.27%|52.43%|1,330,176|
|4|53.27%|36.74%|2,512,896|
|8|23.92%|17.05%|4,878,336|

Executable shared-feature readout agrees with the reconstructed tensor within1.6e-14relative. These are operator controls and a one-update feasibility check, **not converged fits**. The eight-dimensional case has approximately the same storage budget as the earlier10%-error sparse-entry representation; it is therefore a useful next matched-cost fit. Its current17.05%coefficient error does not yet beat that baseline.

This representation shares actual output computations among coordinate products. It does not yet establish semantic reuse across behaviors, native-effect preservation, freshOOD prediction or a runtime benefit. Group labels and within-subspace bases have gauge freedom; arbitrary labels cannot be interpreted as stable circuit identities. Further fitting must check convergence and compare against the existing sparse-edge behavioral baseline before promotion.

[Ray algorithm and control](interaction_shared_write_rays_v1.py) · [Ray screen](INTERACTION_SHARED_WRITE_RAYS_V1_SCREEN.json) · [Ten-start refinement](INTERACTION_SHARED_WRITE_RAYS_REFINE_V1_RESULT.json) · [Subspace implementation](interaction_shared_write_subspaces_v1.py) · [Subspace control](INTERACTION_SHARED_WRITE_SUBSPACES_V1_CONTROL.json).
