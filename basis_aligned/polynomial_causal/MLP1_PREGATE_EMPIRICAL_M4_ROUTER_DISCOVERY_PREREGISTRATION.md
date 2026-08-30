# MLP1 pre-gate empirical-fourth-moment router discovery

**Frozen:** 2026-08-30 03:34 UTC, before any empirical-router optimization or
empirical-router SELECT outcome was computed.

## Prior evidence and claim boundary

The already-opened P512 sparse-Down program computes

$$
c+A\operatorname{TopK}_{32}(E((Lx)\odot(Rx)))
$$

and recovers 0.865084 of the CE damage from deleting MLP1's input-dependent write on
its 96-document SELECT role. It failed its frozen 0.90 gate, so FINAL is sealed.

The already-completed coefficient-Frobenius control folded each encoder row into

$$
Q_a=\tfrac12(L^T\operatorname{diag}(e_a)R+
R^T\operatorname{diag}(e_a)L)
$$

and truncated its signed eigendecomposition at ranks 1, 2, 4, and 8. Rank 8 recovered
0.731306 of the deletion stake and retained 0.845359 of the P512 stake. That control
optimized $\|Q_a-\widehat Q_a\|_F$, not error on real MLP1 inputs.

This experiment is discovery-only. It may use the already-opened disjoint FIT and
SELECT roles from the P512 lifecycle, but it must not request, deserialize, hash anew,
or evaluate FINAL. It cannot revive P512's failed admission gate and licenses no
composition, OOD, semantic, extraction, or removal claim.

## Fixed question

Does optimizing the same fixed signed symmetric rank-8 grammar in the empirical
fourth-moment metric materially improve its frozen-SELECT routing and causal
performance at the same literal price?

For real pre-MLP1 states $x_n$, the primary FIT loss for rank $r$ is

$$
\mathcal L_r=
\frac{\sum_{n,a}[x_n^T(Q_a-\widehat Q_{a,r})x_n]^2}
{\sum_{n,a}[x_n^TQ_ax_n]^2},
\qquad
\widehat Q_{a,r}=\sum_{j=1}^r\lambda_{aj}v_{aj}v_{aj}^T.
$$

This is the sample contraction with the empirical fourth-moment tensor. The runner
must evaluate it by streaming states and scores; it must never materialize an
$1152^4$ tensor.

## Frozen data, initialization, and optimization

- Target: the exact P512 encoder scores at MLP1, with the exact pinned bilin18
  checkpoint and exact P512 program bundle already used by the coefficient control.
- FIT: 96 documents, positions 64--255, from the frozen P512 FIT role.
- SELECT: the disjoint 96-document SELECT role, same positions.
- FINAL: zero documents opened.
- Candidate rank: exactly 8. This is the largest coefficient-screen grammar and is
  fixed before fitting. The experiment does not use SELECT to choose rank.
- Initialization: the deterministic coefficient-Frobenius signed eigenmodes produced
  by the v3 control algorithm (search width 64, five $Q^2$ iterations, QR after every
  iteration).
- Gauge: after every optimizer step, each vector is normalized and its squared norm is
  absorbed into its signed scalar weight, preserving scores exactly. At finalization,
  QR plus an $8\times8$ signed eigendecomposition produces an orthonormal canonical
  representation; factor signs are oriented deterministically and pre/post scores must
  replay. No orthogonality penalty or hidden unpriced transform is allowed.
- Optimizer: Adam, 1,200 fixed minibatch steps, batch size 256, initial learning rate
  0.003, cosine decay to 0.0003, seed 73031, no validation-based stopping, no SELECT
  gradient, and no CE gradient.
- Monitoring uses only the first 2,048 FIT positions.

The primary target is the analytical float32 homogeneous quadratic computed from
captured states and frozen float32 $L,R,E$. The runner also captures the finite-precision
deployed P512 scores made from native bf16 Left/Right products and reports their
relative discrepancy. It must not silently call bf16 rounding an exact quadratic.

Before real optimization, tests must show: direct sample score MSE equals explicit
fourth-moment contraction on a small tensor; a planted distributionally low-rank
quadratic is recoverable; a coefficient-Frobenius truncation can be wrong on a
low-support input distribution; and score evaluation requires only residual states
plus stored factors.

## Frozen evaluation

After the rank-8 factors freeze, SELECT reports for both coefficient and empirical
rank-8 factors:

1. uncentered relative signed-score MSE;
2. positive TopK32 precision, recall, and Jaccard;
3. top-1 atom agreement;
4. relative sparse-code MSE;
5. relative frozen-decoder write MSE;
6. physical whole-model CE;
7. recovery relative to deletion and retention of exact-P512 recovery;
8. paired document-bootstrap 95% intervals for empirical-minus-coefficient CE and
   empirical-minus-P512 CE, using 20,000 fixed-seed resamples;
9. literal stored reals and per-token multiplies; and
10. exact native Left/Right/Down call censuses for every arm.

`ZERO` removes $Dg(x)$ but retains native Down bias. Every empirical and coefficient
candidate must consume only `event.state` and stored factors, signed weights, decoder,
and intercept. Its native MLP1 Left, Right, and Down call counts must all be zero.

## Frozen interpretation

- **Strong same-grammar pass:** at the largest clearly cheaper rank, retention of the
  exact-P512 stake is at least 0.98 and the paired upper 95% bound on
  $CE_{empirical}-CE_{P512}$ is at most 0.02 nat.
- **Metric helps but is insufficient:** empirical rank 8 has a paired upper 95% bound
  below zero versus coefficient rank 8, but misses the strong-pass gate. This supports
  consequence- or CE-weighted fitting next, not promotion.
- **Independent-router grammar is pruned:** empirical rank 8 fails to improve
  coefficient rank 8 in score, write, and paired physical CE, or retains less than
  0.90 of the P512 stake. Move to finite downstream response or a jointly coupled
  router/decoder grammar.
- **Functional but noncanonical routing:** CE passes despite low support agreement.
  This supports functional compression but not stable atom identities.

Regardless of outcome, the strict whole-model storage/causal/terminal ledgers do not
change in this discovery.
