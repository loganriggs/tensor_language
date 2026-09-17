# Endpoint batching equivalent replay

Engineering gate, opened fresh-removal artifact. Freeze exact eight unique
execution groups and preserve each pair/context's RNG key and source/destination
masks. Score six endpoint pairs and four controls from the same final logits,
then restore original 48-row order. Same20arms;160forwards,300seconds maximum.
No row, seed, program, intervention, control or scientific threshold changes.

pred_a: independent CPU packing/unpacking from saved original artifact is exact;
all repeated unrelated scores agree exactly and group-pair ordering survives.
pred_b: native expanded margins differ from original artifact by <=1e-5 maxabs
AND <=1e-6 relative Frobenius. pred_c: each nonzero intervention/readout's effect
relative L2 discrepancy <=1e-3 and forward count exactly160. Scientific outcomes
are inherited only if these equivalence gates pass. This is replay, not fresh
confirmation. Null: hidden endpoint-dependent intervention makes grouping invalid.
Report actual total walltime and compare to original16.29s, without asserting
sixfold wall speedup from the sixfold model-call reduction.

Before enqueue: retain the eight reference native-write inputs/outputs for a
subsequent isolated-process CPU package replay at <=1e-5 relative write error.
No additional model forwards or new behavioral promotion.
