# Sparse interactions in one shared basis: a negative result and a metric correction

22 September 2026, 03:07 UTC.

**Allowing more pairwise interactions improves this particular quadratic decomposition, but it still misses most of the changing signal.** Its apparently better Gaussian function error is largely explained by capturing the mean. This is a completed weights-only baseline, not a failure of unrestricted Tucker or HT.

## Which part of the model?

This test returns to the **last bilinear MLP alone**, block 17 in zero-based indexing. Its input is a full 1,152-dimensional vector. We fold its output projection into the same 16 fixed output readers used elsewhere in this study:

$$
F_v(x)=x^\top T_vx,\qquad
T_v=\operatorname{Sym}\!\left(L_{17}^\top\operatorname{diag}(c_v)R_{17}\right).
$$

Here, each row $c_v$ comes from contracting the last MLP output projection with one fixed output reader. The 16 coordinates describe selected output effects; they are not identified semantic features. Biases and normalization stay outside this quadratic target. This is separate from our two-MLP quartic target and its reported 7–8% text-state errors.

The native implementation is an exact baseline with 4,608 bilinear products. We test whether a shared orthogonal input basis plus selected products can represent this same map more economically.

## What was fitted?

We normalize each matrix $T_v$ by its Frobenius norm and diagonalize the sum of their matrix squares. This produces one weight-derived orthogonal basis $Q$. Then

$$
s=Q^\top x,\qquad F_v(x)=s^\top (Q^\top T_vQ)s.
$$

We retain either all squares, all squares plus greedily selected disjoint pairs, or the highest-energy shared pairs. A pair $s_i s_j$ is computed once even when multiple outputs use it. Off-diagonal coefficients include both symmetric tensor entries. Top-pair selection is optimal for the specified coefficient metric **within this fixed basis**; neither the basis nor the greedy disjoint pairing is globally optimized.

| Representation | Distinct products | Equal-output coefficient error | Uncentered Gaussian error | Centered Gaussian error |
| --- | ---: | ---: | ---: | ---: |
| squares | 1,152 | 96.07% | 23.76% | 92.03% |
| paired_blocks | 1,728 | 94.56% | 23.09% | 89.41% |
| top_1152 | 1,152 | 87.43% | 34.95% | 81.21% |
| top_2304 | 2,304 | 85.81% | 29.51% | 78.98% |
| top_4608 | 4,608 | 83.90% | 26.12% | 76.41% |
| Native bilinear implementation | 4,608 | 0% | 0% | 0% |

The candidate coefficient arrays contain 1.35–1.40 million floats, versus 10.69 million for the native selected-output implementation. Pair-index storage is additional and not included in those float counts. Storage savings alone do not make the high reconstruction errors acceptable.

## Why the error columns differ so much

For a symmetric matrix $T$ and an isotropic standard Gaussian input,

$$
\mathbb E[x^\top Tx]=\operatorname{tr}(T),\qquad
\mathbb E[(x^\top Tx)^2]=2\|T\|_F^2+\operatorname{tr}(T)^2.
$$

After subtracting each function's own mean, its energy is just $2\|T\|_F^2$. Thus centered Gaussian relative error equals the naturally weighted coefficient error. The equal-output column instead rescales each output to equal coefficient norm before combining errors; it is a different weighting.

For this target, **93.33% of uncentered isotropic Gaussian output energy is the constant mean**. A predictor that always emits the 16 means, ignoring its input, already achieves **25.82%** uncentered relative error. It has **100%** error on the centered varying signal. This constant predictor is a diagnostic comparator, not a homogeneous quadratic replacement or an acceptable circuit.

The paired-block candidate gets 23.09% uncentered error but **89.41% centered error**. Its apparent performance is therefore mostly mean prediction. The top-4,608-pair candidate reduces centered error to 76.41%, while uncentered error is 26.12% because it also loses some mean. Ranking candidates solely by uncentered error would obscure this tradeoff.

```mermaid
flowchart LR
  A[Native last MLP and 16 fixed readers] --> B[16 symmetric quadratic matrices]
  B --> C[One shared orthogonal basis]
  C --> D[Keep selected pair products]
  D --> E[Mean reconstruction]
  D --> F[Changing signal reconstruction]
  E --> G[Report separately]
  F --> G
```

## What this changes

This construction does not supply a usable simpler program. It also does not establish an expressivity or optimization limit for Tucker, nonorthogonal dictionaries, low-rank quadratic slices, or arithmetic DAGs. The preceding commutator audit concerned the still narrower common-orthogonal-squares hypothesis; neither test rules out more general circuits.

For subsequent Gaussian comparisons, mean-only and centered-error baselines should accompany total error. Actual text-state, finite-response, and native intervention errors remain separate requirements. The ongoing learned-quartic-residual experiment targets small output coordinates 4–15; the separate removal-geometry experiment targets coordinate 1. Neither result is supplied by this single-layer baseline.

## Implementation and checks

Five toy structures were exercised: shared diagonal, paired blocks, sparse, dense, and signed shared forms. Planted diagonal/block cases reconstruct to numerical precision. The native folded quadratic agrees with its original bilinear factors to relative error $2.3\times10^{-15}$. An independent three-point-per-dimension Gauss–Hermite integration checks the mean/variance identities on five additional quadratic families; discrepancies are below $7\times10^{-16}$. All computations use CPU float64 with two threads. Native audit runtime is recorded in the result JSON (approximately seven seconds).

[Plan](../../direct_tensor_match/ORTHOGONAL_INTERACTION_PLAN_V1.md) · [Results](../../direct_tensor_match/ORTHOGONAL_INTERACTION_V1.json) · [Implementation](../../direct_tensor_match/audit_orthogonal_interactions.py) · [Independent metric controls](../../direct_tensor_match/QUADRATIC_METRIC_CONTROLS_V1.json) · [Earlier commutator restriction](../../direct_tensor_match/JOINT_SQUARE_BASIS_INTERPRETATION_V1.md).
