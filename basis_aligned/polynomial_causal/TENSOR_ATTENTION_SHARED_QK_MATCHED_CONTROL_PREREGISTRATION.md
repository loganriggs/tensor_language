# Matched weighted independent-versus-shared QK control

Date: 2026-08-28 09:23 UTC

Status: frozen before implementation is committed or any row is opened. Discovery
roles only; no final, promotion, or new-data authority.

## Question

The tensor-preserving frontier found 99.46%/99.43% attention recovery with one
rank-384 encoder shared by Q, K, Q2, and K2. The historical routing-384 control used
four separate ridge-map plus ordinary-SVD fits. The shared arm instead used the exact
activation-weighted simultaneous factorization, so its advantage is confounded between
the program class and the optimizer.

This control gives both classes the same weighted objective. At every site, fit
bottom-up on each arm's own deployed trajectory:

$$
\min_{E_j,D_j}\|A^{1/2}(C_j-E_jD_j)\|_F^2
$$

independently for $j\in\{q,k,q',k'\}$, versus

$$
\min_{E,D_j}\sum_j\|A^{1/2}(C_j-ED_j)\|_F^2
$$

with one shared encoder. Here $A$ is the arm's deployed-state covariance and $C_j$ is
the same registered ridge coefficient. Both are weighted Eckart--Young solutions.
Values and output maps remain dense; the complete squared-attention operator is
unchanged.

## Frozen roles and execution

- fit: `.rowcache/fineweb_n480_skip80.pt`, batch 8 per arm;
- coverage: `.rowcache/fineweb_n96_skip80.pt`;
- held-out: `.rowcache/fineweb_n192_skip7000.pt`, production batch 4;
- replication: `.rowcache/fineweb_n192_skip11000.pt`, production batch 4.

The two arm trajectories may be concatenated only along batch during covariance
collection. Covariances, value buses, prefixes, and fitted programs remain separate.
Evaluation must poison all 18 native attention objects and require complete ordered
block/v1 transaction closure, total support, zero tables, and zero native calls.

## Frozen comparisons

Let $R_S$ and $R_I$ be shared and independent normalized recovery.

1. **Replay control:** shared recovery must lie within 0.003 of the committed frontier
   on each role (0.994635 skip-7000; 0.994342 skip-11000).
2. **Sharing is fidelity-free at rank 384:** $R_S\ge R_I-0.005$ on both roles.
3. **Shared program dominates:** prediction 2 holds and shared complete stored values
   and multiply-adds are both lower than independent. The QK factor price is $5Dr$
   versus $8Dr$; complete attention price, not factor-only price, is reported.

If independent improves by more than 0.005 recovery, withdraw the claim that sharing
itself explains the fidelity advantage. The shared class may remain Pareto-optimal,
but the improvement is partly fitter quality and a rank/cost sweep is required. If the
two are matched, the common interface is a certified simplification under the tested
distribution: fewer degrees of freedom with no material executable consequence.

The output is create-only and binds checkpoint, source, parent result, and role hashes.
