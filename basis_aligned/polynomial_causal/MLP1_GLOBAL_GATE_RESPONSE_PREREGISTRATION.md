# Preregistration: trajectory-complete MLP1 physical-gate response

Date: 2026-08-28

Status: CPU mathematical contract only, prospectively amended at 13:12 UTC before
any rows, targets, model responses, authority, or GPU result were opened.

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

The response tensor layout is frozen as `[context, probe, gate]`. Cross-half CSS must
fit an interpolant only on the fit half,

$$
X_{\rm fit}=\arg\min_X\|E_{{\rm fit},S}X-E_{\rm fit}\|_F^2,
$$

then report $\|E_{{\rm eval},S}X_{\rm fit}-E_{\rm eval}\|_F/\|E_{\rm eval}\|_F$
without refitting on evaluation data. The span recomputed from evaluation columns is
retained only as a non-promotive diagnostic; at a support as wide as the response row
dimension it can be vacuously perfect.

## Frozen pilot comparison

- Site: MLP1 in the admitted rank640 complete shell.
- Gate budgets: $K\in\{32,128,512\}$.
- Independent categorical-Fisher probe halves and source-document splits are required.
- Selectors: ridge leverage; response energy; factor-product derangement; local
  activation-times-Down norm; equal-$K$ hash-random gates.
- Report both raw Fisher-energy weighting and context-balanced weighting.
- Report paired support Jaccard, score-rank stability, fit-frozen cross-half CSS error,
  fit-frozen all-on error, non-promotive in-half span capture, per-document errors, and
  fresh-document replication. Fit-half support and coefficients define the only
  candidate bundle; validation never reselects, unions, intersects, or changes it.
- The fixed-grammar standalone price of a retained native support is
  $3456K+1152$ values plus support-index/precision metadata, and $K$ bilinear
  multiplications per token. Response rank, bytes, MDL, and causal equivalence remain
  separate currencies.

Advance a budget only if ridge selection beats response-energy and every negative
control on both independent halves and fresh documents, with support Jaccard at least
0.5 and no preregistered document-stratum failure. Exact numerical margins, split
counts, target-rank rule, bootstrap unit, and multiplicity rule remain launch blockers
until frozen in a serialized no-outcome plan.

For fitted coefficients $\widetilde\beta_n=\beta_n$ on $S$ and zero off $S$, the
first finite candidate calibration follows

$$
\alpha(\epsilon)=\mathbf1+\epsilon(\widetilde\beta-\mathbf1),
\qquad \epsilon=0.1.
$$

Thus omitted gates move from 1 to 0.9 and selected gates move toward their proposed
endpoint. Scaling only selected gates to 0.9 is retained as a sensitivity control; it
does not test whether omitted gates are dispensable. Compare the signed tangent
prediction with observed teacher-forced Fisher/KL using the $\tfrac12\epsilon^2$
normalization. This is not an autoregressive-rollout Fisher claim. Full removal is
forbidden until candidate-path calibration passes.

## Required consequence stages

Any candidate surviving the response pilot must subsequently pass, on untouched rows:

- finite CE, KL, top-1 agreement and accuracy, including coverage strata;
- the frozen causal intervention bank and mixture/Möbius composition tests;
- selective target removal with collateral bounds;
- native-hard-retention versus refitted-Down arms at their complete prices;
- zero native calls and source/artifact closure.

No response result alone licenses an arithmetic-rank, semantic, finite-removal,
bisimulation, or complete-program compression claim.

## Prospective launch closure (13:31 UTC)

Before any global-gate model response was computed, the previously open numerical
choices were frozen in `mlp1_global_gate_plan.json` (file SHA-256
`4eefcc28ec3ed9fda09b047bb122aa47bc314e29f6a3857bc1da541bf7f5f8b1`). The plan
binds 32 registry-fresh one-row documents, split 16/16 by source document, and two new
32-probe halves in a $2\times2$ fit/validation design. Only fit-wave/first-half data
may choose support or coefficients. Every other cell is evaluation-only.

The promotive metrics are fit-frozen CSS and fit-frozen all-on approximation at
$K\in\{32,128,512\}$. Every candidate and control uses the same CPU float64
SVD/Tikhonov solver, cutoff, ridge, condition gate, and coefficient-norm gate. A
candidate must beat each of four controls separately on both fresh validation probe
halves and on both metrics, with a 20,000-draw shared-document max-error bootstrap
lower band over the frozen 48-comparison family, no excessive per-document harm, and
support Jaccard at least $0.5$. The factor-product control is a true fixed-point-free
permutation of gauge-canonicalized product/Down pairs, measured through an auxiliary
zero leaf at the unchanged native baseline, not a relabeling of already computed
scores. Its $+1$ cyclic pairing order and scale/sign replay check are frozen.

The parent row receipt's training prohibition remains in force. The separate
`mlp1_global_gate_row_use_authority.json` permits support selection and regularized
linear analysis-coefficient fitting on wave A as scored analysis, while forbidding
optimization of model parameters or buffers and forbidding all wave-B adaptation.
Neither artifact grants GPU execution or finite gate scaling; a committed,
source-closed, create-only collector and no-outcome execution authority are still
required.
