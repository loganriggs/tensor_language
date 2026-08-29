# MLP2 error-Rayleigh validity pilot — prospective preregistration

**Frozen:** 2026-08-29 23:05 UTC

**Status:** design only; no rows, model responses, or outcomes have been opened

**Purpose:** test whether downstream-consequence geometry predicts the finite failure
of rank-512 MLP2 programs better than local write MSE does.

This is deliberately **not** a low-rank certificate. A scalar consumer has a one-row
Jacobian and therefore trivial rank at most one. A successful pilot licenses an
equal-price weighted fit; it does not prove that the native MLP2 tensor has low rank.

## Frozen experimental object

Allocate 64 registry-fresh FineWeb documents with no reuse from current MLP2 fit or
evaluation roles. Split by source document into 32 `DESIGN` and 32 one-shot `HELDOUT`
documents.

For background

$$
b\in\{\text{native MLP0},\ \text{frozen MLP0-C512}\}
$$

and frozen program

$$
P\in\{\text{FULL512},\text{CONTINUE512},\text{ROBUST512}\},
$$

capture the normalized MLP2 input $z_b$ and full-sequence MLP2 write error

$$
E_{P,b}=P(z_b)-f_2(z_b)
$$

on positions `0:256`. The prefix must not be discarded: an error before position 64
can change later causal attention. Inject

$$
w_{P,b}(\alpha)=f_2(z_b)+\alpha E_{P,b}
$$

through an otherwise native suffix. The $\alpha=1$ arm must exactly replay the
corresponding physical MLP2-program arm before any scientific metric is admitted.

Capture capped final logits on positions `64:256` and the complete native attention-5
and attention-6 write fields. Estimate each directional response $J_cE$ by symmetric
finite differences at

$$
\alpha\in\{-1/8,-1/16,0,1/16,1/8\}.
$$

For final logits, use the exact categorical-Fisher quadratic

$$
q_{\mathrm{logit}}
=\mathbb E_t\left[
\sum_v p_{t,v}\,\delta \ell_{t,v}^2
-\left(\sum_v p_{t,v}\,\delta \ell_{t,v}\right)^2
\right].
$$

This is the local second-order KL predicted by the model's output distribution.
Attention-5 and attention-6 use separately reported native-energy-normalized squared
response norms, $q_5$ and $q_6$. They may enter a fitted predictor as separate
features, but they must never be summed with an arbitrary hand-chosen weight.

## Controls

- Local baseline $\|E_{P,b}\|_F^2$.
- Whole-document derangement of $E$ within a background, rescaled to the recipient
  document's norm. This preserves the complete temporal field but breaks matching to
  the source state.
- Covariance-shaped random errors with matched Frobenius norm.
- Both signs of every error and exact $\alpha=0$ native replay.
- Source document is the common inference unit for every program, background,
  amplitude, and control.

One shared document bootstrap/max band covers programs and backgrounds.

## DESIGN-only fitting

Freeze feature normalizers and one ridge grid using only `DESIGN`. Compare three
prospective document-level predictors:

1. local $\|E\|_F^2$ only;
2. the CE linear directional term plus $q_{\mathrm{logit}}$;
3. the same terms plus separate $q_5$ and $q_6$ features.

No model choice, normalization, or threshold may be changed after opening `HELDOUT`.

## HELDOUT gates

All gates are required:

1. The aggregate `1/16` and `1/8` tangent estimates agree within 20%, with
   document-level Spearman correlation at least 0.8.
2. At $|\alpha|=1/8$, observed teacher KL divided by
   $\tfrac12\alpha^2q_{\mathrm{logit}}$ lies in $[0.8,1.25]$, with document Spearman
   correlation at least 0.6.
3. The full predictor improves clustered held-out MSE by at least 25% over local error
   and at least 10% over the final-logit-only predictor, with Spearman correlation at
   least 0.5.
4. It predicts the same-wave finite $\alpha=1$ MLP0-C512 by MLP2 CE interaction with
   the correct sign and absolute error at most 0.0025 nat for each frozen program.
5. Deranged and covariance-random controls fail the corresponding predictive gates.

## Rulings

- If the small-amplitude gates fail, reject this adjoint/Fisher metric.
- If small amplitudes pass but $\alpha=1$ fails, attribute the remaining problem to
  finite suffix/RMSNorm nonlinearity and do not train from the metric.
- If attention-5/6 do not beat final logits alone, remove them from the objective.
- If every gate passes, authorize one fresh equal-price weighted rank-512 fit. Its
  deployed price remains 1,770,624 float32 coefficients, 512 products, three dense
  multiplies, and zero native MLP2 calls. Any runtime gate or projection is extra and
  must be charged.

## Claim boundary

The shipped-program rank-128 result is conditional on an all-table downstream
program. It does not imply exact CE additivity: residual summation is followed by
RMSNorm, capped logits, and log-softmax, all of which create mixed terms. Only a
same-support Möbius factorial with an explicit tolerance can measure additivity.
