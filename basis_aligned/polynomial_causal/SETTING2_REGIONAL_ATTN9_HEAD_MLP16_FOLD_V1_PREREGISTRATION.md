# Setting2 regional attention9-head × MLP16 fold V1

## Question

Attention9 is the largest individual upstream source in the task-matched regional
earlier-residual × MLP16 path. Does the contribution localize to head9.8, which is
already the central producer in the earlier setting1 regional circuit?

Capture attention9's nine native pre-projection head values on the same 96
British/American prefixes. Project each head through its native output slice and
the exact learned residual propagation coefficient into block 17. Fold each head
write's symmetrized cross term with the propagated MLP16 write through native
MLP17, its shared input denominator, and the row-specific UK-minus-US reader.

## Frozen predictions

1. The nine projected heads reconstruct the propagated attention9 write to
   relative error at most $10^{-6}$, and their folded terms reconstruct the whole
   attention9 × MLP16 term to relative error at most $10^{-8}$.
2. The top two head terms reconstruct the 48-pair whole-term change with relative
   error at most `.40`.
3. The leading head has change-norm ratio at least `.25` and at least `.05` in
   every construction family.
4. **Cross-setting overlap:** head9.8 is the leading head or has change-norm ratio
   at least `.20`.

Freeze the nine-head ranking and report aligned fractions, family ratios,
cancellation, and top-two replay. This is an exact circuit-path overlap screen,
not causal reuse or sufficiency. No behavior/logit outcomes, fitting, new rows,
rank search, gradients, updates, or quantization. QK/value generation, other
upstream sources, native weights, normalizers, bias, final RMS, and softcap remain
charged.
