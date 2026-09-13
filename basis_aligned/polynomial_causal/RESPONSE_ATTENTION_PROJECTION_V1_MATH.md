# Reusing the generated response basis through both QK factors and values

13 September 2026. The three-vector response interface can be folded through all five attention10 projections. It saves repeated projection work across enough edit strengths, while retaining the changing QK1, QK2, V and normalization computations.

At source/query position $t$, write the generated block10 residual change as

$$
\delta_t(a_t)=\sum_{j=0}^2 u_{tj}(a_t)v_{tj}.
$$

For each actual projection matrix $W\in\{W_{Q1},W_{K1},W_{Q2},W_{K2},W_V\}$, prepare the pristine projection $p_{W,t}=Wr_t$ and three vector projections $b_{W,tj}=Wv_{tj}$. Then every new amplitude field uses

$$
p'_{W,t}=p_{W,t}+\sum_j u_{tj}(a_t)b_{W,tj}.
$$

The same response coefficients serve all five consumers. This is identical-function reuse on the same context, not a claim that QK1 and QK2 are separate tasks. The coefficient field can differ by position.

The input squared RMS is computed explicitly from $r_t+\delta_t$. The existing raw-attention executor retains projected Q/K normalization including its epsilon-times-input-norm term, native rounded rotary tables, the product of both QK scores, causal masking, changing current values, first-layer value mixing and the output projection. The optimization changes only how the five raw projections are obtained.

## Evidence and honest break-even

On actual model weights, one synthetic17-position context and21 signed position-dependent edit fields, projection replay agrees within1.60e-15 and complete attention outputs within2.03e-15. Input norms match exactly, and zero edits give exactly zero projection change. This checks equivalence to the already validated FP64 raw-attention executor; it is not new native-text or OOD evidence.

The projected response bank adds17,280scalars per position (five matrices × three vectors ×1152coordinates), besides the pristine projections and three response vectors. For N edit evaluations, dense projection work changes from5(1+N) input-vector projections to5(1+3), followed by cheap vector combinations. Full attention routing, values/output work and background preparation remain.

An executed break-even countercheck includes projection-bank preparation, using two CPU threads and median five repeats. The baseline is given its generated deltas; the shared version also forms deltas for input norms.

| Edit evaluations | Original projections | Shared projections | Original / shared time |
|---|---:|---:|---:|
| 1 | 6.00ms | 10.68ms | 0.56× |
| 2 | 9.32ms | 11.24ms | 0.83× |
| 4 | 14.98ms | 12.70ms | 1.18× |
| 8 | 28.26ms | 15.60ms | 1.81× |
| 21 | 70.11ms | 25.42ms | 2.76× |

Thus it helps repeated amplitude sweeps in this test, but loses for a single child/remainder pair when preparation is charged. Do not claim a two-branch speedup. Query count, available cache memory and kernel implementation determine whether to use it. These are projection-only timings; no whole-model speedup follows.

The native six-product run did not use this new projection cache and must not be cited as native validation of it. A useful next integration would reuse the same prepared context across a genuine multi-strength removal/composition panel, comparing full execution and cache costs.

[Executable projection cache](response_attention_projection_v1.py) · [Control](RESPONSE_ATTENTION_PROJECTION_V1_CONTROL.json) · [Break-even countercheck](RESPONSE_ATTENTION_PROJECTION_BREAKEVEN_V1_RESULT.json) · [Existing attention equations](RAW_ATTENTION_RESPONSE_V1_MATH.md).
