# Setting2 regional QK1 edit downstream-response census V1 preregistration

`ACTIVE_TRACK: WEIGHT_FOLDING`

Fresh recursive removal of the three late-touching head9.8 QK1 carry blocks is
material, but its actual target-logit effect has cosine `-.92536` with the negative
of the frozen MLP16×MLP17 folded prediction. Keep that intervention fixed and
decompose the response it induces downstream.

Reuse the 48 `ODD_FRAMING_FRESH_V1_ROWS` rows, now as an opened diagnostic panel.
Run native and the unchanged QK1 late-touching removal. Capture the edited-minus-
native attention and MLP writes at layers 9–17. After multiplying each write by
its exact learned residual propagation coefficient, their 18-term sum must equal
the final pre-RMS residual change:

$$
\Delta x_{18}=\sum_{\ell=9}^{17}\gamma_\ell
\left(\Delta a_\ell+\Delta m_\ell\right),
\qquad
\gamma_\ell=\prod_{j=\ell+1}^{17}\lambda_{j,0}.
$$

Contract every term with the row-specific UK-minus-US unembedding reader before
final RMS and softcap. Rank paired British-minus-American change vectors by norm,
report aligned fractions, family ratios, and top-five replay. Separately compare
the complete pre-RMS numerator change with the actual final target-logit edit to
test whether final normalization changes the direction.

Predictions:

1. **A — exact census.** Carry reconstruction and final residual-response closure
   are at most $10^{-6}$, manual attention9 reconstruction is at most $10^{-6}$,
   all values are finite, and exactly 12 length-bucketed executions occur.
2. **B — direct routing write is insufficient.** The propagated attention9 write
   term alone has relative replay error at least `.50` for the complete pre-RMS
   numerator change.
3. **C — immediate MLP response is prominent.** MLP9 ranks in the top three terms
   and has change-norm ratio at least `.25`.
4. **D — a small suffix response set is adequate.** The five largest frozen-grain
   terms replay the complete pre-RMS numerator change with error at most `.35`.
5. **E — numerator direction survives the output nonlinearity.** The complete
   pre-RMS numerator paired change has cosine at least `.70` and sign agreement at
   least `.70` with the final target-logit edit effect.

This is an exact response attribution for a fixed causal edit, not an independent
intervention or fresh validation. A pass can select downstream terms for a new
fold; it cannot promote an individual module without a later selective edit. A
failure localizes the missing explanation without changing the QK1 group or
retuning any threshold. Price: 48 rows, two arms, 12 executions, zero fits,
backwards, gradients, parameter updates, or quantization.
