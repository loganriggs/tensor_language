# Closing the recent MLP16-to-MLP17 path

v627's frozen-write approximation missed7.5–9.3% of the component's actual
MLP16-ablation effect. v628 explicitly folds MLP16 into the parent component's
readers and recomputes attention17. The extracted executor uses no native model
objects at runtime and does not materialize an expanded high-order tensor.

Let n16=N(h16), m=(L16 n16)(R16 n16), and w16=D16 m+b16. At intervention
scale g, live17=lambda17*(h16+g*w16)+beta17*x0. Attention17 consumes N(live17)
with its actual Q/K normalization, RoPE, two-score product, causal mask and
first-value mixture v1. Then h17=live17+attention17 and its norm remains explicit.

For the frozen parent readers P and signed coefficients c, define

F=lambda17*D16^T P,

a=g*m^T F,

b=(lambda17*h16+beta17*x0+attention17+g*lambda17*b16)^T P.

The component is sum_k c_k (a_k+b_k)^2/(mean(h17²)+eps). The runtime retains
background²,2*background*MLP16, andMLP16² separately. Holding background
fixed, the last term is quartic in n16, represented by shared intermediate
products rather than an expanded tensor. The actual complete path is not a
fixed polynomial: background attention and normalizers remain explicit.

## Native validation

Two previously used8-document panels,64 tokens, scales0,.5,1,1.5. No fitting
or rank selection. The comparison is with actual model interventions scaling
the whole MLP16 write; downstream attention17 is allowed to change.

* Maximum recent-state replay error:1.19e-7.
* Maximum replay error against the frozen256-feature component:8.59e-7.
* A CPU test covers two sequence lengths and all four scales against the native
  module implementation, including signed coefficients and nonzero biases.

| MLP16 scale | Error predicting native component change: calibration / validation |
| --- | --- |
| 0 | .02272 / .02051 |
| .5 | .02805 / .02523 |
| 1.5 | .03116 / .02897 |

Baseline component approximation error remains .04592/.04165. Under1.5x
scaling, absolute component errors rise to .06101/.05690 while changes still
predict within3.2%. The parent spectral approximation, not recent-state
replay, is now the dominant discrepancy.

At baseline, the background–MLP16 cross term has norm1.16/1.28 times the target
component norm; at1.5x it reaches2.22/2.52. Signed cancellation makes ratios
above one possible. Dropping those terms would destroy fidelity.

## Extracted artifact and honest price

[Package and usage](../bilinear_quotient/circuits/followups/recent_folded_component_v628_package/README.md)
contains an independent Torch runtime, three weight shards, and a hash
manifest. It runs in a separate isolated Python CPU process without importing
the native model or loading its checkpoint. That package smoke uses synthetic
ports; native fidelity comes from the managed v628 replay above.

The package requires h16,x0,v1 from upstream generators. It stores24,235,715
tensor values, about97 MB. Pulling back farther has expanded the required
program, so this is not yet a simple circuit. All native MLP16 and attention17
weights are charged; a small parent core does not make these dependencies free.

Next decomposition should simplify this complete local program while preserving
its intervention tests, rather than compressing only the numerator and using
an uncharged norm/attention oracle. Preserve the three input ports as explicit
remaining dependencies. Semantic selectivity, useful reuse, upstream extraction
and a materially simpler executable remain unproved. The full goal is active.
