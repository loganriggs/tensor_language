# Execute the retained polynomial by subtracting its omitted tail

13 September 2026. Same approximation as the existing three-term composed
interaction; simpler execution. This changes the implementation, not the fitted
function or the available behavioral evidence.

Write the actual pre-MLP10 input as $z=y+r(a)$ and its bilinear numerator as
$B(z)=D[(Lz)\odot(Rz)]$. The previous executor explicitly computed background
and mixed products, then replaced the self product $B(r)$ by its three-term
approximation. Equivalently,

$$
\widehat B(z)=B(z)-\left[B(r(a))-\widehat B(r(a))\right].
$$

From the [five-coefficient numerator](CONIC_AMPLITUDE_POLYNOMIAL_V1_MATH.md),
with dimensionless amplitude $t=a/s$, the omitted self contribution is

$$
E(a)=\frac{a^2t^3}{2\rho(a)^2}
\left(\widetilde N_3+t\widetilde N_4\right).
$$

Thus execute the native bilinear numerator and subtract $E(a)$, before the
unchanged MLP10 RMS division, bias and residual addition. All vectors have
output width 1152. This identity holds for arbitrary background $y$ in real
arithmetic. It does not assume the self approximation is accurate for every
background; the previously observed cancellation failures remain.

The direct tail compiler stores two output vectors plus the amplitude scale,
2305 scalars, instead of six width-4608 reader projections, three output vectors
and a scale, 31105 scalars. That is a 92.6% reduction in this **extra prepared
state**, not total circuit or model storage. Shared upstream response context,
native Left/Right/Down weights, bias, attention and suffix remain required.
The executor avoids reconstructing the residual/background and their separate
hidden products. It retains the same three large dense maps per input.

The matched CPU benchmark uses the same actual weights and synthetic inputs as
the existing cost check, two CPU threads and seven alternating timing trials.
FP64 agreement with the old approximation is at most $2.41\times10^{-15}$;
FP32 agreement is at most $6.96\times10^{-7}$, below the declared $2\times10^{-5}$
countercheck tolerance. The first FP32 attempt exposed the upstream helper's
FP64 output; explicit casting of the shared prepared context repaired this
instrument error before collecting results.

| Batch | FP64 warm speedup versus old executor | FP32 warm speedup |
|---|---:|---:|
| 1 | 1.05× | 1.22× |
| 3 | 1.04× | 1.05× |
| 16 | 1.23× | 1.21× |
| 64 | 1.21× | 1.26× |

The registered 1.1× criterion at **every** batch fails in both precisions.
The broad speed claim is rejected; measured larger-batch improvements and
literal state savings survive. Cold preparation remains slower than direct
native execution. Warm execution is approximately native speed, not evidence
of a faster model. This is a useful complementary representation: a compact
description of a modification can cost less than reconstructing its retained
pieces. No new native text/OOD validation or autonomous extraction is claimed.

Primary code: `polynomial_tail_mlp10_v1.py` and
`check_polynomial_tail_mlp10_v1.py` (optional `--fp32`). Receipts:
`POLYNOMIAL_TAIL_MLP10_V1_RESULT.json` and
`POLYNOMIAL_TAIL_MLP10_V1_FP32_RESULT.json`.
