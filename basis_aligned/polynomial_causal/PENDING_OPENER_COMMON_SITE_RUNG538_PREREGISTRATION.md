# Rung 538: common causal location for the pending-opener state

**Frozen before any site-interchange outcome:** 2026-09-03 14:42 UTC

## Decision

Rung 537 established only that the model can answer all frozen counterfactual
families. This rung asks whether replacing one complete internal state moves the
answer from the base prompt toward the donor prompt for both independently built
answer-changing families. If no common location works, fitting a small DAS
subspace is not authorized.

## Exact intervention

Every base/donor pair has equal token length. Let $a_s(x)$ denote the complete
activation at the final prompt position at site $s$. The base-to-donor run uses

$$
a_s(b) \leftarrow a_s(d),
$$

while every other activation follows the base run. The reverse run substitutes
$a_s(b)$ into the donor run. This is a full activation interchange, not a mean
ablation, rank reduction, gradient approximation, or fitted projector.

The fifteen sites, in frozen causal order, are:

1. residual stream entering each block 8 through 14 (seven 1,152-dimensional states);
2. the 4,608 bilinear products immediately before the down-projection in MLPs 8
   through 14 (seven states, interleaved after their corresponding residual entry);
3. layer-13 attention head 8's complete 128-dimensional pre-output-projection
   vector (between the layer-13 residual entry and MLP13 product).

Thus the exact order is `resid8, mlp_product8, ..., resid13, attn13h8,
mlp_product13, resid14, mlp_product14`. If several sites pass, select the earliest
one in this order. The rule prefers the earliest causally adequate location and
does not choose the largest observed effect.

## Measurement

For base answer $y_b$, donor answer $y_d$, and base-to-donor patched logits
$\ell_{b\leftarrow d,s}$, define donorward movement

$$
\Delta^{b\to d}_s =
\left[\ell_{b\leftarrow d,s}(y_d)-\ell_{b\leftarrow d,s}(y_b)\right]
-\left[\ell_b(y_d)-\ell_b(y_b)\right].
$$

Define $\Delta^{d\to b}_s$ analogously with the preferred answers reversed.
For each site, family, split, and direction, report the mean, the fraction of
individual pairs with $\Delta>0$, and a 2,000-resample group-bootstrap 95% lower
bound on the mean.

A cell passes only when both directions separately have mean movement above zero,
bootstrap lower bound above zero, and at least 70% of pairs moving donorward. A
site is common and live only if every cell for both `opener_type_substitution`
and `closed_then_reopened_type` passes on both FIT and SELECT. FINAL_TEST, OOD,
the invariance rows, and controls remain unopened.

## Instrument and price gates

- Frozen R537 input hashes and the successful capability-result hash must match.
- The pinned bilin18 weight hash must be
  `680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3`.
- The run must use 128 FIT/SELECT pairs, 16 baseline forwards, and two patched
  directions for each of 15 sites: exactly $16+2\times15\times16=496$ model
  forwards, zero backwards.
- Every intended source-target activation difference must be finite and nonzero.

## Opposing outcomes

- A common live site authorizes fitting shared and family-specific DAS projectors
  only at the frozen selected site.
- Sites live only for the one-token family support a punctuation-token shortcut,
  not the proposed shared pending-state variable.
- Different sites for the two families imply distinct computations or a bad site
  vocabulary; do not force a shared low-dimensional fit.
- No live site means this site ladder cannot mediate the behavior. Preserve the
  null and redesign the causal location before any rank search.

This rung targets cross-family circuit grouping and a real manipulation site. A
small dimension or good reconstruction cannot pass it.
