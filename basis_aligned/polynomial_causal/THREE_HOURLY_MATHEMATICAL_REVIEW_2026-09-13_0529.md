# Mathematical review: exact joint-QK/shared-source coefficient geometry

Cycle due 05:29 UTC, 13 September 2026; work and controls executed from 05:30. Previous cycle 02:29. This is an exact compression-object calculation, not a newly identified circuit.

**Outcome:** folding both QK numerators through their shared query/source produces almost no extra shared-rank gain over the value-only optimum. At rank64, squared-error gains are about 0.002%. A separate generalized-eigenvalue audit explains why: this joint metric is within about 1.05% Frobenius error of a scaled value metric. Its uniform norm-equivalence bound limits metric-only reoptimization gains to 4.75–4.86% for the fixed producer. Do not infer that a different arithmetic graph or jointly changed QK factors cannot compress.

## Object, dimensions and interface

At one fixed query/source position pair, consider the numerator

$$
p(q,s)=(q^\top A s)(q^\top B s)Fs,
\quad q\in\mathbb R^{1152},\quad s\in\mathbb R^{2304},
\quad F\in\mathbb R^{128\times2304}.
$$

A and B are the head17.2 folded QK matrices, with native BF16-rounded rotary tables and score scaling; their last 1152 source columns are zero. F concatenates the current and first-layer value maps with the actual signed mixture. Both bilinear factors read the same q and s. The coefficient tensor is symmetric in its two query slots and separately in its three source slots. This is query degree2, source degree3, total degree5.

The contraction graph contains two bilinear forms, their scalar product, and one vector-valued linear form. Native matrix factors are retained. Invertible internal changes of head coordinates do not identify semantic units; swapping the two scalar factors or reciprocal rescaling leaves this numerator unchanged. Polynomial coefficient coordinates are fixed in the native input spaces.

For the previously constructed MLP17 mixed tensor, flatten output token and residual-input indices into a matrix M of shape 13824-by-128 (or 6912-by-128 for six spelling contrasts). Composing Mp gives query2/source3 coefficients with a further independent residual-input index. This independence is a declared relaxation: that residual and the query can be coupled in the actual model.

QK norms, residual RMS, multiple source positions, retained finite-edit trajectory dependence, the quadratic-in-head-write term and final readout are outside this numerator. The model is not globally a polynomial. No input closure, OOD prediction or selective-removal claim follows from this norm calculation.

## Literature mapping and a usable exact contraction

The polynomial coefficient metric corresponds to the apolar inner product with the bidegree normalization 2!3!, because the query and source groups are symmetrized separately. The factorial-weighted monomial definition and its relation to the homogeneous Bombieri normalization are described in [Aldaz, Bravo and Render, Section 2](https://arxiv.org/html/2403.10584v1). This specifies a metric, not a factor-recovery theorem. Full symmetry across query and source variables would be a different object.

[Isserlis's moment formula](https://doi.org/10.1093/biomet/12.1-2.134) evaluates Gaussian products through covariance pairings. Here it is an exact algebraic contraction device. Gaussian L2 is **not** the coefficient metric: the query trace must be removed. No sampled Gaussian observations or text are fitted.

For fixed q, let a=Aᵀq and b=Bᵀq. A head output has source coefficient tensor sym(a⊗b⊗f_i). Summing its six source permutations gives

$$
K_{ij}(q)=\frac16\left[
(\|a\|^2\|b\|^2+(a^\top b)^2)(f_i^\top f_j)
+\|a\|^2(f_i^\top b)(f_j^\top b)
+\|b\|^2(f_i^\top a)(f_j^\top a)
+(a^\top b)((f_i^\top a)(f_j^\top b)+(f_i^\top b)(f_j^\top a))\right].
$$

Let \(G_{\rm Gauss}=E_{q\sim N(0,I)}K(q)\). The exact fourth-moment rule reduces every term to matrix products and traces. If \(Q=\operatorname{sym}(A^\top B)\), tracing the two query slots yields source polynomial \((s^\top Qs)Fs\), whose coefficient Gram is

$$
G_{\rm trace}=\frac{\|Q\|_F^2FF^\top+2FQ^2F^\top}{3}.
$$

For symmetric query matrices, the fourth moment has two cross-pairings plus one trace pairing. Therefore the desired 128-by-128 coefficient Gram is exactly

$$
G=\frac{G_{\rm Gauss}-G_{\rm trace}}{2}.
$$

[Implemented contraction](joint_qk_source_gram_v1.py). It stores query/source matrices and a small output Gram, avoiding an array scaling as 128×1152²×2304³. The current dense contraction costs matrix products of order n_q²n_s+n_qn_s², plus output contractions of order m n_s²+m n_q² and smaller terms; working matrix storage is quadratic in query/source widths. Native Q/K low-rank factors permit further optimization if this kernel becomes a bottleneck. The measured two-position experiment costs only 1.64 CPU seconds, so another compiler is not justified now.

Given G=HHᵀ, the exact rank-r matrix approximation to MH solves the declared shared-head-mode restriction. SVD is globally optimal within that matrix-rank class; no restart uncertainty applies to this comparison. It does not solve arbitrary CP/LL1 topology selection or joint optimization of A, B and F.

## Executed checks and native result

[Explicit dense control](JOINT_QK_SOURCE_GRAM_V1_CONTROL.json) constructs all 2!3!=12 permutations on three tiny independent fixtures. Gram disagreement is at most 4.15e-16; diagonal function evaluation agrees to 3.75e-14. Omitting the trace correction gives 23.9–50.0% Gram error on those fixtures. This is an executed discriminator against confusing Gaussian moments with coefficient Frobenius norm.

[Native result](HEAD17_JOINT_QK_PULLBACK_V1_RESULT.json) uses query/source positions (1,0) and (8,0), actual rounded RoPE and both value streams. Compare the value-only optimum and the joint-QK optimum in the **same joint metric**:

| Query/source positions | Output target | Rank64 value-only fit | Rank64 joint-QK fit | Squared-error gain |
|---|---|---:|---:|---:|
| 1 / 0 | Twelve tokens | 39.30384% | 39.30343% | 0.00210% |
| 1 / 0 | Six contrasts | 38.91499% | 38.91462% | 0.00192% |
| 8 / 0 | Twelve tokens | 39.30073% | 39.30035% | 0.00192% |
| 8 / 0 | Six contrasts | 38.91210% | 38.91176% | 0.00175% |

All require 128 shared directions for 2% error in this restriction. The query-trace correction is 0.664–0.700% of the Gaussian Gram in Frobenius norm on these native weights; small here does not excuse omitting it generally.

The strongest explanation for the weak gain is near-equivalence of the induced metrics. [Executed audit](JOINT_QK_METRIC_EQUIVALENCE_V1_RESULT.json) whitens G by FFᵀ. Generalized-eigenvalue ratios are 1.04978 and 1.05099. For any matrix error E,

$$
\lambda_{\min}\|EF\|_F^2\leq\|EH\|_F^2
\leq\lambda_{\max}\|EF\|_F^2.
$$

If a representation family has an attained exact value-metric optimum, reoptimizing solely under G can reduce its squared G error by at most \(1-\lambda_{\min}/\lambda_{\max}\): 4.74% and 4.85% here. This is a uniform fixed-producer bound, not just an optimizer failure. It does not constrain changing the producer or exploiting a different arithmetic representation.

## Competing approaches and next action

The existing shared-cubic source tools already use permutation inner products and query quadratic factors; reuse those for a factorized full-polynomial fit. The older Segre/minor decomposition addresses multi-source quadratic attention interactions and should not be substituted for this one-source degree5 object. Neither prior route was a proved whole-model circuit reconstruction.

Dense TT/Tucker compression of the huge expanded array is unnecessary for the tested mode because the 128-by-128 Gram solves it exactly. A sparse graph over the original Q/K/value readers instead changes a different object and may still be informative. Polynomial factorization can recover the already supplied product of two bilinear forms and a value form, but merely recovering native factors gives no new compression. Hankel/weighted-automaton realization would require a finite-state input closure and string-level data that this local numerator does not supply.

Do not queue another identical shared-rank sweep on this metric. The next higher-information structural comparison would change the joint QK/source factor graph or restrict the actual retained interaction's producer domain, while keeping all new dependencies priced. The new exact Gram can support such a comparison but does not by itself implement its cross-kernel or graph optimizer.

Meanwhile the independently registered full-U shared/private multistart fit is live. Its completed configuration artifacts should receive the already implemented per-token and compiled-function audits as they land. This is the immediate executed continuation alongside the exact kernel, native measurement and metric-equivalence discriminator. The program's four behavioral properties remain unproved.

Next mathematical review is due 08:29 UTC, at the first safe boundary.
