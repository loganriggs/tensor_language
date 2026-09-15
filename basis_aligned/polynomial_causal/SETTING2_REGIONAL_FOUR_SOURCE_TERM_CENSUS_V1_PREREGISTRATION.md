# Setting2 task-matched regional four-source term census V1

## Motivation

The cached four-source MLP17 census used grammatical activation endpoints with
fixed UK/US unembedding readers. It is valid generic path algebra, but it cannot
show which terms matter at the regional circuit's own decision point. Repeat the
frozen decomposition on all 96 controlled British/American prefixes, using each
row's own UK-minus-US unembedding contrast.

At the final prediction position, write the raw MLP17 input as

$$
x=e+p+o+a,
$$

where $e$ is the layer-17 incoming residual excluding the scaled MLP16 output,
$p$ is that scaled MLP16 output, $o$ is attention17 excluding head17.2, and $a$
is head17.2. Expand all ten self/cross terms through native MLP17. Divide every
term by the shared native input denominator

$$
R(x)=\operatorname{mean}(x^2)+\epsilon
$$

before applying the row-specific reader. Pair each British row with its adjacent
American control and analyze the 48 term changes.

## Frozen predictions

1. **Instrument:** raw and normalized ten-term sums reproduce the direct native
   MLP17 numerator to relative error at most $10^{-8}$; the row checker passes;
   exactly 96 sequences execute in 14 physical prefix batches.
2. **Head17.2 circuit term:** the sum of all normalized terms containing $a$ has
   change-norm ratio at least `.05` on the regional pairs.
3. **Earlier-residual self term:** `ee` remains the top individual term and has
   change-norm ratio at least `.50`.
4. **Earlier-residual × MLP16 term:** `ep` has change-norm ratio at least `.05`.

Report every term's raw and normalized change-norm ratio, aligned fraction,
family-level normalized ratios, ranking, and cancellation. No fit, rank sweep,
behavior/logit outcome, threshold change, new row selection, backward pass,
parameter update, or quantization. This is task-matched path selection, not causal
identification. Down bias, direct residual, final RMS, softcap, source generation,
and all native weights remain charged. A failing head17.2 prediction closes it as
the center of this regional MLP17 path; the highest live non-head term becomes the
next folding target.
