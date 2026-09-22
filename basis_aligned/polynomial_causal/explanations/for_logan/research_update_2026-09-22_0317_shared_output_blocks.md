# Shared output directions help, but the quadratic programs remain inaccurate

22 September 2026, 03:17 UTC.

**Letting a computed feature write to a learned combination of outputs improves the decomposition.** At 4,608 products, changing the output basis lowers centered pooled error from 17.44% to 12.92%. Equalizing the importance of the original outputs produces a more balanced alternative whose worst-output error is 18.51%. Neither is yet a simpler faithful circuit: the native computation is exact at the same product count and nearly the same coefficient storage.

## What is being decomposed here?

This is a bounded return to the **single last MLP**, block 17, rather than our two-layer quartic candidate. Its bilinear input has 1,152 coordinates and 4,608 channels. Folding its output projection through 16 fixed readers gives

$$
F_v(x)=x^\top T_vx,\qquad v=1,\ldots,16.
$$

The fixed readers select output effects from earlier work. These are neither 16 datapoints nor 16 identified semantic concepts. This test covers all input coordinates but only those selected outputs; normalization and biases remain outside this quadratic target. The reader choice has prior data context, while the new factorization uses weights without fitting text-state output labels.

The question is whether the scalar computations need to correspond to those particular output coordinates. There is no mathematical reason they must. A computed feature can instead have an effect on several coordinates.

## How this represents shared effects

Form the small output Gram matrix

$$
M_{uv}=\langle T_u,T_v\rangle_F,
\qquad M=Q\Lambda Q^\top.
$$

Rotating the outputs is an exact change of coordinates:

$$
A_g=\sum_v Q_{vg}T_v,\qquad
h_g(x)=x^\top A_gx,\qquad F(x)=Qh(x).
$$

We then approximate each quadratic form $A_g$ with a small number of signed bilinear products, and allocate a fixed total product budget across them. Each $h_g$ writes through the whole output vector $Q_{:,g}$. This is a block-term construction: individual output-shared quadratic forms use their own input directions. It supplies candidate computations for a later shared arithmetic graph, but does not perform arbitrary graph search.

```mermaid
flowchart LR
  A[Last MLP weights and 16 fixed readers] --> B[16 quadratic coefficient matrices]
  B --> C[Learn shared output directions from the coefficient Gram]
  C --> D[Factor each quadratic form into signed products]
  D --> E[Allocate product budget across forms]
  E --> F[Map predictions back to original 16 coordinates]
  F --> G[Measure pooled and every-output error]
```

The native products already have cross-output reuse. We first tried retaining a subset of them and refitting their output weights. Residual-aware selection barely improved on selecting channels by individual contribution energy: **68.56% versus 68.70% centered error at 512 products**. Its registered improvement criterion failed. Allowing new input and output directions is a different, more successful construction.

## Results and the effect of weighting

All numbers below are centered isotropic-Gaussian/coefficient errors, not text-state errors. Centering removes the large constant-mean contribution that distorted the earlier uncentered Gaussian comparison. “Equal-output” normalizes each original quadratic matrix to equal coefficient energy before choosing the basis and allocating products; its output map undoes that scaling.

| Basis/weighting | Products | Natural pooled error | Equal-output error | Worst original output |
| --- | ---: | ---: | ---: | ---: |
| Rotated, natural | 512 | 47.53% | 72.56% | 92.74% |
| Rotated, natural | 2,048 | 28.69% | 46.73% | 68.67% |
| Rotated, natural | 4,608 | 12.92% | 21.15% | 31.54% |
| Rotated, equal_output | 512 | 54.12% | 68.10% | 81.00% |
| Rotated, equal_output | 2,048 | 34.34% | 40.56% | 44.47% |
| Rotated, equal_output | 4,608 | 17.56% | 16.86% | 18.51% |
| Native shared-channel program | 4,608 | 0% | 0% | 0% |

Under natural weighting, rotating outputs improves error by approximately **26% relative** versus the optimally allocated original-output spectral baseline at both 512 and 4,608 products. Both registered improvement predictions pass. This comparison keeps the product count fixed. The equal-output version was a descriptive follow-up prompted by poor weak-coordinate errors; it does not change those registered predictions.

The weight setting changes the scientific conclusion. Natural weighting at 4,608 products gives the best pooled error, 12.92%, but a worst-output error of 31.54%. Equal weighting raises pooled error to 17.56% while reducing the worst output to 18.51%. We should preserve that tradeoff rather than summarize both as simply “better.”

At 4,608 products, these programs contain about 10.53–10.61 million reader/output floats, versus 10.69 million in the exact native selected-output program. Signs and routing metadata are additional. At 512 products, they use about 1.18 million floats, but errors remain much too large. No candidate is exported for native replacement on the strength of this screen.

## What we learned about the two-stage proposal

The factorization stage benefits from allowing learned output effects and output-specific input subspaces. This gives a concrete reason to include block terms alongside one global sparse Tucker dictionary. However, output rotation does not by itself identify a semantic feature, and independent quadratic factors still leave reuse between those factors unexplored.

The arithmetic-graph stage remains a separate task: find useful common inputs and intermediate computations, charge their computation once, and validate the resulting program. The previous reader-reuse edit demonstrated a small storage saving while preserving a fitted parent; it did not resolve that parent's native fidelity failures.

The queued learned-quartic residual experiment and removal-geometry diagnostic concern the two-layer path and remain separate. The present single-layer result should not be substituted for their text, finite-response, or intervention evidence. The full goal of predictive, reusable, selectively manipulable circuits remains unmet.

## Checks and reproducibility

Five planted shared-output mixtures recover exactly to numerical precision. The full native output transformation replays the original bilinear weights within $3\times10^{-15}$ relative error. The signed-product construction and budget allocator reuse the preceding explicit factor and exhaustive allocation controls. Float64 CPU calculations, two threads; the result JSON records total runtime. The chosen output SVD is an initialization rule, not a global optimum for compact block terms.

[Rotated-block plan](../../direct_tensor_match/OUTPUT_ROTATED_BLOCK_PLAN_V1.md) · [Results and all original-output errors](../../direct_tensor_match/OUTPUT_ROTATED_BLOCK_V1.json) · [Implementation](../../direct_tensor_match/audit_output_rotated_blocks.py) · [Shared-channel selection failure](../../direct_tensor_match/SHARED_CHANNEL_GREEDY_INTERPRETATION_V1.md) · [Independent-output baseline](../../direct_tensor_match/OUTPUT_LOCAL_BILINEAR_INTERPRETATION_V1.md).
