# Preregistration: trajectory-complete MLP1 physical-gate response

Date: 2026-08-28

Status: CPU mathematical contract only. No rows, targets, model responses, or GPU
authority are opened by this document.

## Exact object

For MLP1 gate $n$ and token position $q$,

$$
h_n(z_{c,q})=(\ell_n^\top z_{c,q})(r_n^\top z_{c,q}).
$$

Insert one shared scale vector $\alpha\in\mathbb R^{4608}$ before Down at every
position. For suffix score $s_{c,a}$, the published response must equal

$$
E_{(c,a),n}
=\left.\frac{\partial s_{c,a}}{\partial\alpha_n}\right|_{\alpha=1}
=\sum_q h_n(z_{c,q})d_n^\top g_{c,a,q}.
$$

The sum over every position is mandatory. A position-128-only edit is a different
object and cannot justify pruning a gate shared across the trajectory. The collector
must obtain $E$ directly through the shared $\alpha$ leaf; full write gradients need
not and must not escape its one-use transaction.

## Two different selection questions

1. **Ridge column subset selection** asks which physical gate interventions preserve
   the span or regularized covariance of measured responses. For target rank $r$, use
   $\lambda=\lVert E-E_r\rVert_F^2/r$ and

   $$
   \tau_n=e_n^\top E^\top(EE^\top+\lambda I)^+Ee_n.
   $$

2. **All-on sparse approximation** asks which gate scales approximate
   $E\mathbf 1$. It is a separate group-sparse objective with one support and
   coefficient rule shared across contexts, probes, and positions.

Column-span preservation does not imply that retaining those gates reconstructs the
native MLP. A column interpolant may mix omitted columns, and refitting Down changes the
response matrix whose leverage was measured. Native hard retention and selected
Left/Right plus refitted Down must therefore be separate program families.

Ridge leverage and projection-cost preserving column selection are established
matrix-approximation tools; their guarantees concern the measured matrix, not the
nonlinear transformer ([Cohen, Musco & Musco 2017](https://arxiv.org/abs/1511.07263),
[Boutsidis, Drineas & Magdon-Ismail 2014](https://arxiv.org/abs/1103.0995)).

## Frozen pilot comparison

- Site: MLP1 in the admitted rank640 complete shell.
- Gate budgets: $K\in\{32,128,512\}$.
- Independent categorical-Fisher probe halves and source-document splits are required.
- Selectors: ridge leverage; response energy; factor-product derangement; local
  activation-times-Down norm; equal-$K$ hash-random gates.
- Report both raw Fisher-energy weighting and context-balanced weighting.
- Report paired support Jaccard, score-rank stability, cross-half projection capture,
  per-document capture distribution, and fresh-document replication.
- The fixed-grammar standalone price of a retained native support is
  $3456K+1152$ values plus support-index/precision metadata, and $K$ bilinear
  multiplications per token. Response rank, bytes, MDL, and causal equivalence remain
  separate currencies.

Advance a budget only if ridge selection beats response-energy and every negative
control on both independent halves and fresh documents, with support Jaccard at least
0.5 and no preregistered document-stratum failure. The first finite experiment is a
small global scaling, $\alpha_n=0.9$ on selected packages, with predicted-versus-observed
Fisher/KL response. Full removal is forbidden until that calibration passes.

## Required consequence stages

Any candidate surviving the response pilot must subsequently pass, on untouched rows:

- finite CE, KL, top-1 agreement and accuracy, including coverage strata;
- the frozen causal intervention bank and mixture/Möbius composition tests;
- selective target removal with collateral bounds;
- native-hard-retention versus refitted-Down arms at their complete prices;
- zero native calls and source/artifact closure.

No response result alone licenses an arithmetic-rank, semantic, finite-removal,
bisimulation, or complete-program compression claim.
