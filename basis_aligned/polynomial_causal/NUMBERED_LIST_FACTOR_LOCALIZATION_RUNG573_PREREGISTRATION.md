# Rung 573 preregistration: exact numbered-list label-factor localization

## Computation being tested

At layer 8, final query position $q$, head $h$, and source position $k$, the exact residual-stream contribution is

$$
t_{h,k}=p_{h,q,k}\,W_O^{(h)}v_{h,k},
$$

where bilin18's unnormalized attention weight is

$$
p_{h,q,k}=\frac{(q_1\cdot k_1)(q_2\cdot k_2)}{128^2},
$$

and the layer-8 value is

$$
v_{h,k}=(1-\lambda_8)c_v^{(8)}(x_k)+\lambda_8v^{(0)}_k,
\qquad \lambda_8=4.
$$

The two registered heads are L8H7 and L8H3. The tested arms are fixed before outcomes:

1. complete output of both heads at the final query (positive ceiling);
2. joint score-and-value contribution from all semantic label positions;
3. joint contribution from only the final visible label;
4. score-only transplant at the final label;
5. complete value-only transplant at the final label;
6. layer-0 cached-value part only at the final label;
7. layer-8 own-value part only at the final label;
8. layer-0 cached-value parts at all label positions.

“Score-only” uses donor $p$ with base $v$; “value-only” uses base $p$ with donor $v$; “joint” uses donor $p$ and donor
$v$. Semantic label positions are mapped by line identity, never by assuming a fixed absolute token index.

## Rows and splits

Only R567 numbered-list FIT and conditional SELECT rows are eligible. The two answer-changing state-shift families are
targets. Surface changes, the earlier-middle-label edit, repeated labels, and the step-two conflict are required
answer-preserving controls under the list-index claim. FINAL_TEST/OOD stay closed.

## Decisions

The complete two-head ceiling must move the donor-versus-base answer margin in the donor direction in at least 75% of
groups in every target family/direction, with a positive 2,000-group-bootstrap lower mean effect. Otherwise the legacy
site is rejected on R567 and no factor is selected.

For each factor arm and target cell, recovery relative to the complete-head effect is

$$
r=\frac{M_{\mathrm{arm}}-M_{\mathrm{base}}}{M_{\mathrm{complete\ heads}}-M_{\mathrm{base}}}.
$$

Every target cell must have median recovery at least 0.5, at least 75% positive recovery, and a positive bootstrap
lower mean recovery. On each control family/direction, the arm's full-vocabulary logit RMS and registered-answer margin
change are divided by the median corresponding complete-head target effect across FIT. Both normalized control effects
must be at most 0.25. The native correct answer must remain the best registered digit candidate in at least 75% of
control groups.

FIT selects the first passing arm in the fixed order: final-label cached value, final-label complete value, all-label
cached values, final-label joint, all-label joint, final-label score, final-label own-layer value. This ordering is by
semantic support and exact tensor terms, not hidden dimension or rank. SELECT tests only the chosen arm and the complete
two-head ceiling. It cannot choose a replacement. A null forbids tuning thresholds or widening to arbitrary subspaces.

The maximum planned price will be computed from the frozen dry-run after the semantic-position audit. No model weight is
updated. A held activation result is still not a weight-level circuit; it licenses a later exact compilation through
the layer-0 value matrix, layer-8 scores/output projection, and downstream MLP consumers.
