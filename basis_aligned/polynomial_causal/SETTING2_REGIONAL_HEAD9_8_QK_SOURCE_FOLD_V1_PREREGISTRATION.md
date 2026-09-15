# Setting2 regional head9.8 QK-source × MLP16 fold V1 preregistration

`ACTIVE_TRACK: WEIGHT_FOLDING`

## Question

Which block-8 source interactions feed the two QK scores of head9.8 along the
now-validated regional head9.8 × MLP16 × MLP17 path?

At every query/key position, split the raw block-9 attention input exactly as

$$
r_9=c_8+a_8+m_8,
$$

where $c_8$ is the propagated pre-block-8 carry including the learned embedding
skip, $a_8$ is the propagated attention-8 write, and $m_8$ is the propagated
MLP-8 write. Native residual-RMS and per-head Q/K RMS denominators are held
fixed. Because all remaining maps and rotary position transforms are linear,
each head9.8 score factor has the exact nine-term expansion

$$
s_f(t,j)=\sum_{u,v\in\{c_8,a_8,m_8\}}
\left\langle R_tQ_f\widehat u_t,\;R_jK_f\widehat v_j\right\rangle/128,
\qquad f\in\{1,2\}.
$$

For one factor at a time, multiply each ordered source term by the other native
score factor and the native mixed value, sum over source positions, apply the
native head9.8 output projection and residual propagation, then fold the result
against MLP16 through MLP17 and each row's UK-minus-US unembedding reader. Thus
each of the two independent nine-term censuses must sum exactly to the previously
localized head9.8 folded term.

## Frozen data and price

- Reuse all 96 rows in `FIRST_TOKEN_PATH_FRESH_V1_ROWS.json`; no new rows.
- Batch by native sequence length with at most eight rows: exactly 14 physical
  prefix forwards and 96 sequences.
- Two QK factors, nine ordered carry/attention8/MLP8 source interactions each.
- Zero behavioral-logit access, fits, backwards, gradients, parameter updates,
  or quantization.

## Predictions and fixed gates

1. **A — exact instrument.** Raw block-9 source reconstruction and each QK
   score-factor sum have relative error at most $10^{-8}$; each nine-term folded
   sum closes at most $10^{-8}$; the recomputed native head9.8 write bridges to
   the captured model head write at most $10^{-6}$.
2. **B — selective source-pair concentration.** In at least one QK factor, the
   top three frozen ordered interactions replay the paired head9.8 folded change
   with relative error at most $.50$.
3. **C — attention8 enters routing.** In at least one factor, the sum of all five
   ordered interactions containing $a_8$ has paired-change norm at least $.10$
   of the complete head9.8 folded change.
4. **D — stable leading interaction.** The globally leading ordered interaction
   has change-norm ratio at least $.20$ in every one of the four frozen prompt
   families for its factor.

Rank terms only by global paired-change norm. Freeze the top three after seeing
no outcomes other than this registered census. Report signed aligned fractions
and cancellation; ratios may exceed one. Failure of concentration retains the
exact term census and rejects only this three-source simplification.

## Scope

This is exact path attribution with native denominators and values held fixed.
It is not a QK intervention, causal sufficiency claim, value-source
decomposition, unseen-text result, or complete circuit extraction.
