# Setting2 regional head9.8 QK1 carry-source fold V1 preregistration

`ACTIVE_TRACK: WEIGHT_FOLDING`

## Question

Which embedding and layer-0–7 module-write interactions make the dominant QK1 carry8-query × carry8-key term inside the regional head9.8 × MLP16 × MLP17 path?

Expand the block-9 carry exactly as

$$
c_8=\gamma_E E+\sum_{\ell=0}^{7}\left(\gamma_{A,\ell}A_\ell+\gamma_{M,\ell}M_\ell\right),
$$

where the $\gamma$ values are products of the checkpoint's learned residual coefficients through the raw block-9 attention input. This gives 17 sources: embedding and the attention/MLP write from each of layers 0–7. Hold the complete raw block-9 RMS denominator, QK1 head denominators, QK2 score, native mixed value, MLP16 state, MLP17 denominator, and row-specific UK-minus-US reader fixed. Then expand only the selected QK1 carry×carry score as

$$
s_{1,cc}(t,j)=\sum_{u,v\in\mathcal S}
\frac{\langle R_tQ_1\widehat u_t,R_jK_1\widehat v_j\rangle}{128},
\qquad |\mathcal S|=17.
$$

Fold all 289 ordered query-source × key-source terms through the unchanged QK2/value branch, head9.8 output projection, residual propagation, MLP16 × MLP17 interaction, and row reader. The 289 terms must sum to the parent QK1 carry×carry folded term. Rank by paired cue-change norm; retain query/key order and report aligned fractions plus four-family ratios.

## Frozen data and price

- Reuse the 96 rows in `FIRST_TOKEN_PATH_FRESH_V1_ROWS.json`, batched by length at no more than eight rows: exactly 14 physical prefix forwards.
- Seventeen propagated sources and 289 ordered QK1 source interactions.
- Zero new rows, behavioral logits, fits, backwards, gradients, parameter updates, or quantization.

## Predictions and gates

1. **A — exact instrument.** The 17 sources reconstruct carry8 with relative error at most $10^{-6}$; their 289 QK1 scores reconstruct the parent carry×carry score at most $10^{-8}$; their downstream folded sum closes at most $10^{-8}$; and the recomputed parent carry×carry change-norm ratio matches the bound V3 value within $10^{-6}$.
2. **B — sparse ordered concentration.** The ten globally largest ordered interactions replay the paired parent carry×carry folded change with relative error at most `.50`.
3. **C — cross-module interaction is material.** The sum of all ordered terms whose query and key sources differ has change norm at least `.20` of the parent carry×carry change.
4. **D — typed earlier writes matter.** In at least one of the attention-containing or MLP-containing aggregates, the paired change norm is at least `.10` of the parent.
5. **E — stable leading term.** The globally leading ordered interaction has change-norm ratio at least `.10` overall and at least `.03` in every prompt family.

Failure of B preserves the exact census and rejects a ten-term simplification. Failure of C–E rejects the corresponding grouping claim without changing closure. Do not merge ordered pairs or choose a wider top set after outcomes.

## Scope

This is exact task-matched path attribution under fixed native denominators and partner branches. It does not establish causal sufficiency, QK necessity, value transport, unseen-text transfer, or a complete circuit. QK2 attention8 routing and the established head8.2 value path remain separate registered branches.
