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


## Two levels of reuse: fixed model vectors and changing context vectors

The three-vector basis is economical within one context, but it hides another sharing opportunity. The original four ingredients are $w,m_0,Jz,Jw$. The first and last are fixed for this writer and model; only $m_0,Jz$ change with context. A computation graph can exploit that distinction even though the local response span has only three dimensions.

For each attention projection $W$, store $Ww$ and $WJw$ once globally. At each new context, compute $Wm_0$ and $WJz$ once. Apply the exact response coefficients:

$$
W\Delta(a)=-aWw+
\frac{2\beta a-\gamma a^2}{\rho_a}Wm_0
-\frac{a}{\rho_a}WJz+
\frac{a^2}{2\rho_a}WJw.
$$

The implementation includes the actual block10 re-entry scale in these projected vectors. The same scalar coefficient calculations serve all five projection consumers. The pristine projection and changing raw-state norm remain explicit.

The extra dynamic projection cache falls from **17,280 to 11,520 scalars per position**, plus **11,520 scalars stored once globally**. At one position there is no total storage advantage; across more than two positions the shared global cost is repaid. This counts the projected banks only, excluding common response contexts and pristine projections. It is reuse across contexts of fixed functions, not new evidence that different semantic behaviors share a circuit.

Actual-weight controls across one/four synthetic 17-position contexts and 21 signed strength fields pass: projection error1.47e-15, complete attention error2.27e-15, raw-norm error1.60e-16. No native-text promotion has yet tested this new cache.

The registered cold-start speed criterion fails: including model-global preparation, the two-query workload is not at least10% faster in both context sizes. The executed amortization countercheck separately measures reuse with that global cache already prepared:

| Contexts | Queries | Earlier cache | Two-level cache, global setup excluded | Speed ratio |
|---|---:|---:|---:|---:|
| 1 | 2 | 10.64ms | 9.14ms | 1.16× |
| 1 | 21 | 24.28ms | 20.57ms | 1.18× |
| 4 | 2 | 33.84ms | 27.38ms | 1.24× |
| 4 | 21 | 45.48ms | 40.75ms | 1.12× |

The one-time global setup is1.69–1.83ms in that run. These are descriptive CPU measurements of projection updates, not a repaired passing cold-start registration or a whole-model speedup. They do not compare against direct uncached projection on the same new panel. The positive conclusion is narrower: fixed/global and context-dependent sharing can reduce cache size and improve repeated use of the previously proposed projection cache.

[Two-level executor](response_attention_projection_v2.py) · [Original control and cold verdict](RESPONSE_ATTENTION_PROJECTION_V2_CONTROL.json) · [Warm-cache countercheck](RESPONSE_ATTENTION_PROJECTION_V2_WARM_RESULT.json).


## Integrated real-text check: changed attention and joint norm generated internally

The [composed v2 executor](composed_mlp10_inputs_v2.py) now combines the three-vector response, two-level attention projection cache and six-product residual/residual bank. It generates child/remainder attention changes from the pristine context and scalar edit fields, then forms the complete MLP10 cross-product including residual/attention and attention/attention terms. It computes its own joint squared RMS from the generated pre-MLP10 changes. No observed changed-attention vectors or changed joint denominator enter this candidate.

This integrates earlier generator functionality with the newly simplified graph; generation of those dependencies was already available in composed v1. It is not a new claim to have removed all background dependencies. Pristine states, bias-free MLP9 output, first-layer values, scalar edit fields and native weights are inputs. The behavioral evaluation still supplies the additive post-MLP10 background and native suffix.

The original two-prefix CPU pilot passes: maximum old-generator replay error1.32e-12 and complete product replay2.47e-14. An additional fixed eight-prefix panel uses indices12,36,60,84,104,120,136,152, one from each existing regional/FineWeb group. Maximum generator replay error1.74e-12; product replay3.01e-14. Execution takes7.80CPU seconds for that eight-prefix check. All measured predicted target/control effects and all native reference effects exactly match the previously saved full-panel v1 receipt on those same rows. The comparison was actually performed, not inferred solely from the formulas.

The native model discrepancies remain unchanged. Regional per-prefix target absolute errors are0.95–1.91e-6 in the eight-row panel; on row12 that is1.64% of its small target effect. FineWeb row104 has native CE effect-1.01e-6 and predicted-2.38e-7: about76.5% relative error, despite only7.75e-7 absolute difference. The saved v1 receipt has exactly that discrepancy. Therefore the new reuse introduces no measured additional error on these checks, but cannot be advertised as uniformly accurate prediction of tiny native effects. No new OOD evidence or full-panel native validation of v2 is claimed.

The weight inventory also limits the compression claim: J/w, attention10 projections/output and MLP10 L/R/D still cost25,216,128scalars before the additional global projection cache. The exact simplification reduces prepared per-context intermediates and repeated operations; it does not eliminate those dense weights. A runtime comparison of the whole integrated executor, including preparation, remains unmeasured. A single edit pair need not benefit from constructing a six-product bank.

[Two-prefix receipt](COMPOSED_MLP10_REAL_TEXT_V2_CONTROL.json) · [Eight-group receipt](COMPOSED_MLP10_REAL_TEXT_V2_EIGHT_GROUP_CONTROL.json) · [Executed prior-outcome comparison](COMPOSED_MLP10_V2_PRIOR_OUTCOME_AUDIT.json) · [Native test source](check_composed_mlp10_real_text_v2.py).
