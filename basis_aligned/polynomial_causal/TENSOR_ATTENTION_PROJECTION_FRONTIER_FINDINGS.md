# Tensor-preserving attention projection frontier

Date: 2026-08-28

Status: discovery-only executable subsystem result. This is not yet a compressed
whole-model ship because the MLPs remain native.

## Result

The best tested attention program preserves the exact bilin18 squared-attention
operator but replaces the four routing projections with one shared rank-384 input code
and four typed decoders. Value and output projections remain dense:

$$
h_t = E x_t, \qquad
(q_t,k_t,q'_t,k'_t)=(D_qh_t,D_kh_t,D_{q'}h_t,D_{k'}h_t).
$$

The program then executes the unchanged per-head RMSNorm, RoPE, product of two QK
contractions, causal mask, cross-layer first-value bus, value contraction, and output
projection. It has total token support, no table, no output hook, and zero native
attention calls.

| arm | held-out recovery | replication recovery | stored values | dense storage | multiply-adds |
|---|---:|---:|---:|---:|---:|
| dense identity | 100.00% | 100.00% | 143.328M | 100.00% | 100.00% |
| routing-384 | 97.07% | 97.09% | 111.478M | 77.78% | 80.00% |
| value-384 | 94.97% | 95.13% | 135.366M | 94.44% | 95.00% |
| joint-384 | 93.06% | 93.35% | 103.515M | 72.22% | 75.00% |
| joint-512 | 97.25% | 97.46% | 130.057M | 90.74% | 91.67% |
| **shared-QK-384** | **99.46%** | **99.43%** | **87.590M** | **61.11%** | **65.00%** |

Recovery is

$$
R_A=\frac{\operatorname{CE}_{\rm constant}-\operatorname{CE}_A}
{\operatorname{CE}_{\rm constant}-\operatorname{CE}_{\rm native}}.
$$

On skip-7000, native CE is 3.29205, the constant-attention CE is 6.84909,
and shared-QK CE is 3.31114: only +0.01908 nat harm. On the disjoint skip-11000
replication, the corresponding values are 3.09711, 6.85233, and 3.11836:
+0.02125 nat harm. Scoring covers 27,974 and 27,497 positions respectively.

## Composition result

Joint-384 harm is +0.24678 nat. Routing-384 and value-384 harms sum to +0.28334
nat, giving the preregistered interaction margin

$$
0.24678-(0.10428+0.17906)=-0.03656\ \text{nat}.
$$

Thus routing and value compression compose under the exact tensor contraction; their
joint harm is slightly smaller than additive and safely below the +0.10-nat threshold.
All five compressed arms meet the frozen executable-compression gate on both role
labels: at least 90% recovery, fewer complete bits and multiply-adds than dense
attention, total support, and zero native attention calls.

## What the rank-384 code means

This is a site-specific continuous *routing-relevant state code*. At each layer it is
the 384-dimensional linear coordinate system of the current normalized residual state
from which all four objects that determine attention routing can be decoded. It is not
yet a semantic labeling of the coordinates, and it is not the lexical code found in
MLP tables. Operationally it says that “where to read” is governed by one common state
interface even though bilin18 stores four separate dense matrices.

The encoder is found by a simultaneous, activation-weighted factorization. If $A$ is
the deployed-state covariance and $C_j$ are the four registered ridge coefficients,
the fitted encoder $E$ and decoders $D_j$ minimize

$$
\sum_{j\in\{q,k,q',k'\}}
\left\|A^{1/2}(C_j-ED_j)\right\|_F^2.
$$

This is solved by horizontally concatenating the whitened $C_j$, taking its leading
left singular space, and unwhitening the encoder. It is gauge-invariant at the program
level under $E\mapsto EG$ and $D_j\mapsto G^{-1}D_j$; a later canonicalization can fix
that internal basis without changing the executable function.

## Controls and scope

- Dense bank CE equals native CE exactly on both corpus roles.
- Every bank executes all 18 sites in order with exact block and first-value-bus
  transaction closure.
- All native attention objects are poisoned during program evaluation; literal native
  calls are zero.
- Complete receipts include projections, lambdas, rotary constants, bits, and measured
  multiply-add counts. Token-table values are zero.
- Fit uses 480 spent rows; the coverage mask uses 96 spent rows; evaluation uses two
  disjoint 192-row roles. No new final or promotion role was opened.
- Runtime was 452.5 seconds after joint-batch amortization.

This result compresses only the attention subsystem. Native MLPs still execute, so it
does not change the strict 0% admitted recovery of the fully compiled +0.8976 ship-gap
denominator. It does establish the first high-fidelity, compressed, total-support,
zero-native-call component class suitable for that eventual full program.

## Most important remaining falsifier

The shared arm uses the optimal weighted simultaneous estimator, whereas routing-384
replays the historical independent per-map ridge-plus-SVD estimator. Therefore the
result proves that the shared program class works, but does not by itself prove that the
shared constraint causes its large advantage over independent factors. The next cheap
matched control is four independently activation-weighted rank-384 factorizations,
compiled bottom-up on the same roles. Comparing it with shared-QK-384 separates:

1. benefit from the improved weighted objective;
2. benefit from forcing one common routing interface;
3. benefit from the lower $5Dr$ rather than $8Dr$ price.

After that control, the highest-value whole-model move is the factorial cross between
this attention program and a total-support rank-8--64 lexical/correction program.
The earlier hybrid oracle's $-2.17559$-nat interaction forbids adding their isolated
recoveries.

## Artifacts

- `tensor_attention_projection_frontier_results.json`: create-only numerical receipt
- `tensor_attention_projection_frontier.py`: source-closed compiler/evaluator
- `tensor_preserving_attention.py`: executable tensor operator and program bank
- `TENSOR_PRESERVING_ATTENTION_PREREGISTRATION.md`: frozen decisions
