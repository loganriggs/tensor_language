# Conditional arithmetic programs and a graph simplification

22 September 2026, 01:34 UTC.

**Latest result:** the simplified rank-256 program uses **0.85 million coefficients and 1,536 variable products**, versus 2.39 million coefficients and the same product count for its CP parent. Opened-panel accuracy remains similar; native finite-removal testing is pending. The construction and subsequent graph edit are explained below.

We now have a constructive baseline that reduces the input directions used by a fitted CP program and **analytically accounts for the discarded directions**. At 256 retained directions, storage falls from **2.39 million to 0.85 million coefficients**, with similar value and same-token response errors on opened panels. Native finite-removal testing is next; this is not an adopted circuit.

The target remains the 16-coordinate pure quartic contribution through the final two MLPs. The starting object is our fitted 512-term CP approximation, not the exact native tensor. Each term multiplies four linear input features and writes into shared output directions.

Assume the input follows the Gaussian mean and covariance used in the weight-matching experiments. Select a smaller input subspace from the parent's exact average derivative information. Retain those coordinates and average over the remaining Gaussian coordinates. This is a conditional expectation.

Averaging a product is not the same as multiplying averaged factors. Discarded directions contribute quadratic corrections and a constant. For retained affine features $l_i$ and discarded-factor covariances $c_{ij}$,

$$
E[F_{\mathrm{term}}\mid\text{retained inputs}]
=l_1l_2l_3l_4+\sum_{i<j}c_{ij}\prod_{k\notin\{i,j\}}l_k
+c_{12}c_{34}+c_{13}c_{24}+c_{14}c_{23}.
$$

Compute all six pair products once and reuse them. Two also form the quartic. This takes seven variable multiplications per term rather than nine for separate evaluation. Constant corrections combine into the output bias.

```mermaid
flowchart LR
    X[1152 input coordinates] --> P[Shared projection: 64, 128 or 256 directions]
    P --> L[Four affine features per CP term]
    L --> Q[Six cached pair products]
    Q --> H[Quartic plus weighted quadratic corrections]
    H --> Y[16 output coordinates]
    B[Constant correction] --> Y
```

The table below describes the initial conditional programs, before the graph simplification explained at the end.

| Program | Stored coefficients | Variable products | Native value error, two starts | Feature-1 response error |
| --- | ---: | ---: | ---: | ---: |
| Original CP parent | 2.39 million | 1,536 | 7.38 / 7.58% | 9.72 / 11.78% |
| 64 directions | 0.24 million | 3,584 | 8.72 / 8.83% | 13.23 / 13.27% |
| 128 directions | 0.44 million | 3,584 | 7.79 / 7.93% | 11.37 / 12.18% |
| 256 directions | 0.85 million | 3,584 | 7.31 / 7.53% | 10.26 / 10.54% |

Value errors use the opened 256-document panel; responses use earlier same-token pairs. Small apparent improvements are exploratory, not established superiority.

These programs have more variable products but smaller dense linear projections. At rank 256, coefficient multiplications fall from about 2.37 million to 0.83 million, excluding the common output writer. This is a tradeoff under our different complexity penalties, not a win under every cost definition.

Matching the fitted parent under its Gaussian gives below 1% error at rank 256. That does not mean below 1% error against the native model. Component sensitivity errors remain above 10%, smaller outputs remain inaccurate, and semantic selectivity is unproved.

The construction passed independent integration controls on five toy structures. Float32 exports reproduce float64 predictions to below 4.6e-7 relative error. The next native test checks actual feature-removal effects with original normalization and softcap retained.

This is an explicit alternative to Tucker/HT fitting and generic graph edits: use a defined projection to propose a smaller input dictionary, then compile its correction terms with reuse. Whether those dictionaries contain interpretable computations remains a separate question.

[Derivation, operation accounting and limitations](../../direct_tensor_match/CONDITIONAL_CP_PROGRAMS_INTERPRETATION_V1.md) · [Measurements](../../direct_tensor_match/CONDITIONAL_CP_PROGRAMS_V1.json) · [Executable evaluator](../../direct_tensor_match/conditional_quartic_cp.py).

## Graph simplification follow-up: remove the extra product cost

A subsequent graph edit keeps only the two quadratic corrections whose pair products are already needed to form each quartic term. It deletes the other four pair-product nodes per term. The constant correction stays.

The resulting rank-256 programs use **1,536 variable products**, matching the original CP parent, while retaining about **0.85 million stored coefficients** instead of 2.39 million. Their same-token response errors are **10.27% and 10.55%**, essentially unchanged from the full conditional programs. Dense linear operation counts also remain substantially lower.

The choice among three possible pairings used artificial Gaussian probes only; text labels did not choose the pairing. Independent artificial probes show a small increase in parent reconstruction error. This is an approximate graph edit, not an exact algebraic identity. At rank 64, the accuracy penalty is larger.

This is a concrete instance of the second stage we discussed: propose features through a decomposition, then simplify the executable graph by reusing intermediates and deleting low-value nodes. Native finite-removal checks are registered separately for the full and simplified versions. [Graph-edit results and limitations](../../direct_tensor_match/LEAN_CONDITIONAL_CP_INTERPRETATION_V1.md).

## Measured speed and a limitation of the Gaussian approximation

The simplified rank-256 program is **1.48–2.46× faster than its CP parent** in the tested warm CPU batches. This uses two threads, float32, and batches of 1, 64 or 2,048 states. It measures only the selected polynomial's 16 outputs—not the entire transformer or GPU performance. The full conditional program was only about 1.1× faster at the largest batch and slightly slower at batch 64, so the graph edit matters in practice. [Benchmark details](../../direct_tensor_match/CONDITIONAL_CP_CPU_BENCHMARK_INTERPRETATION_V1.md).

An algebraic stress test shows the limitation of fitting around a shifted Gaussian. The native pure quartic has exactly the same output at $x$ and $-x$. The simplified conditional programs do not: on 256 selected states, error rises from about **8.4%** at the original inputs to **13.8–13.9%** at their negatives. Negated states preserve input norm but are not established text examples, so this is not a text-OOD measurement.

This is expected to be possible under conditional averaging; it is not a failure of the integration code. The approximation gains efficiency in a data-informed region while losing global polynomial identities. Enforcing sign symmetry by averaging the two predictions worsens the original-input error in this test. [Symmetry audit and limits](../../direct_tensor_match/CONDITIONAL_SYMMETRY_INTERPRETATION_V1.md).

## 01:59 mathematical review: normalization does not make the quartic disposable

A further CPU check tested a tempting simplification. On a fixed-radius input surface, the native quartic can be decomposed exactly into a genuinely quartic harmonic part plus quadratic and constant parts. “Harmonic” here means its Laplacian is zero; it does not mean a semantic feature. We can calculate the lower-degree parts directly from weight contractions without storing the enormous tensor.

However, dropping the quartic remainder gave **138.47% value error** on one state from each of 256 opened documents, and **112.55% error** in the existing root1 same-token responses. Independent small-model derivative controls passed. Thus this particular mathematically well-defined truncation fails: normalization does not make the remaining fourth-degree computation negligible. This does not rule out a different learned compact program.

The [three-hour review](../../THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-22_0159.md) connects Gaussian conditional programs, hierarchical tensor representations and graph rewrites to their precise assumptions. The immediate priorities remain balanced feature fitting and testing the smaller conditional programs inside native finite interventions.
