# Setting2 earlier-residual/attention term census V1

## Question

The exact three-source census found that the combined background self term `gg`
dominates changes in twelve selected MLP17/unembedding readers, while `gp` and
`pp` are also live. Split that background without fitting to determine whether the
signal comes from the earlier decoder residual, the other attention17 heads, or
their interaction.

Write the MLP17 input as

$$
x=e+p+o+a,
$$

where $e$ is the incoming layer-17 residual excluding the scaled MLP16 output,
$p$ is the scaled MLP16 output, $o$ is attention17's output excluding head17.2,
and $a$ is head17.2. Expand all ten unordered bilinear source pairs through native
MLP17 and the same twelve frozen token readers.

## Frozen checks

1. The ten terms reproduce the direct absolute numerator and 64 paired donor/base
   changes to relative FP64 error at most $10^{-10}$.
2. Their aggregation must reproduce every prior three-source term: `gg=ee+oo+eo`,
   `gp=ep+op`, `ga=ea+oa`, with `pp`, `pa`, and `aa` unchanged, each to relative
   error at most $10^{-10}$.
3. Report each term's change-norm ratio, aligned fraction, per-reader ratios, and
   `.05` live verdict. Freeze the descending ranking.
4. Report cancellation. Signed or norm ratios are not additive causal shares.

The highest-ranked live term determines the next algebraic split. If it is `ee`,
split earlier decoder writes by layer or circuit-selected source in a later folding
step. No fit, rank search, model forward, behavior, fresh/OOD data, threshold
change, gradients, updates, or quantization. Normalizers, bias, direct residual,
softcap, all source generation, and native factors remain charged.
