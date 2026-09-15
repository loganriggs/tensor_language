# Setting2 regional QK1 edit downstream-response census V2 correction

V1 completed the frozen calculation but is instrument-invalid because its final
residual-response closure was $1.04642\times10^{-6}$ against the preregistered
$1.0\times10^{-6}$ boundary. This is the accumulation floor from reconstructing
18 FP32 module-write differences with separately rounded learned residual
coefficients; carry reconstruction was $9.71\times10^{-8}$ and manual attention9
reconstruction was exact.

V2 changes only that numerical audit ceiling to $2\times10^{-6}$. This remains
well below all scientific effect sizes. The rows, fixed causal edit, 18 response
terms, contractions, ranking, prices, scientific predictions B–E, thresholds,
and interpretation from
`SETTING2_REGIONAL_QK1_EDIT_DOWNSTREAM_RESPONSE_CENSUS_V1_PREREGISTRATION.md`
are unchanged. The V1 invalid receipt is hash-bound, and V2 reruns the complete
calculation rather than copying its values.
