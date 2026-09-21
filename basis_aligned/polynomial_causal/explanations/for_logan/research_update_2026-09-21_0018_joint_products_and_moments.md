# Research update — 2026-09-21 00:18 UTC

Joint product decomposition reduces the number of products, but it has not yet preserved the confirmed program's accuracy or identified stable constituent computations. An exact joint-input moment improves held-out swap fidelity compared with separable covariance weighting. The smaller eight-product candidate still fails the registered preservation threshold and is not promoted.

## The two tensors and the optimization

The target here is the confirmed four-feature, 16-product program, expressed as four bilinear matrices in compact 16-dimensional input spaces. It is an approximation to the broader native target; this experiment does not silently replace that distinction. We fit a joint CP representation

$$
\widehat T_{gij}=\sum_{k=1}^{r}W_{gk}A_{ik}B_{jk}.
$$

Each product can contribute to all four output features. This allows product reuse across outputs, whereas the baseline assigns four separate products to each feature. Ranks 4, 8, 12 and 16 were fitted with three starts and two Adam learning rates, using 1,200 steps per run.

Five planted structures tested the fitting code and optimizer choice: orthogonal components, dense rank three, shared output direction, signed cancellation, and dense rank one. Adam and Muon received two rates, two starts and 600 steps per case. Mean best-case error was approximately 6.4e-13 for Adam and 0.00295 for Muon. This selects Adam for this experiment; it is not a general optimizer comparison, and different schedules may change the result.

| Products | Best separable weighted error to confirmed tensor | Native calibration scalar error |
|---:|---:|---:|
| 4 | 5.05% | 27.3% |
| 8 | 1.96% | 12.7% |
| 12 | 1.03% | 10.1% |
| 16 | 0.87% | 8.8% |

The confirmed program's native calibration error is 7.18%. An exact rank-16 representation is known and independently replays at floating-point precision. Therefore the random rank-16 fits stopping above zero demonstrate incomplete optimization, not an expressivity lower bound. Low weighted coefficient error is also not equivalent to low centered native variation error.

## What the joint moment adds

In the compact input coordinates, define the lifted feature and its empirical second moment:

$$
\phi=n\otimes m,\qquad M=\mathbb E[\phi\phi^\top].
$$

For flattened tensor error D, exact paired-input mean squared error is

$$
\mathcal E=\operatorname{tr}(D M D^\top).
$$

Here M is 256×256 and includes input dependence and nonzero means. It is the second moment of lifted inputs, rather than their centered covariance alone. Separate marginal moments produce a different matrix. The tensor target remains weight-derived; original calibration inputs supply M. No held examples are fitted.

The implementation's loss and gradients match direct paired evaluations within 1.5e-16 and 2.1e-15. Six warm-start Adam fits of the eight-product candidate reduce calibration error to the original native scalar target from 12.7% to 8.5%, while worsening its separable weighted coefficient error from 1.96% to 2.90%. These metrics favor different approximations.

| Joint same-token native effect error | Separable eight-product fit | Joint-moment eight-product fit | Confirmed 16-product baseline |
|---|---:|---:|---:|
| FineWeb | 10.84% | 9.09% | 6.45% |
| Code | 12.73% | 10.15% | 6.22% |

The registered 20% squared-error improvement over the equally sized separable model passes in both domains. Preservation within 1.25 times the confirmed baseline error fails. This is a useful data-informed metric improvement, not promotion of the eight-product candidate. The tested eight-product graph stores 18,432 input coefficients and 32 output-mixing coefficients, plus the unchanged four means and output interface.

A successor CPU audit finds a relative matrix discrepancy of 0.899 between the joint and independent-marginal moments in the same coordinates. For the separable candidate's residual, paired error energy is 1.79 times the independent-marginal estimate; for the joint-refitted residual it is 0.179 times that estimate. The fit explicitly moves error into directions less important under the paired calibration distribution. Held native swaps provide the separate transfer evidence.

```mermaid
flowchart LR
    A[Confirmed bilinear tensors] --> B[Joint product factorization]
    B --> C[Separable weighted matching]
    C --> D[Exact paired-input moment refit]
    D --> E[Held native swaps improve]
    E --> F[Preservation threshold still fails]
```

## Why product identity remains unresolved

Within one baseline output feature, replacing A by A H and B by B H for an orthogonal four-by-four H preserves the sum of products exactly. An executed control preserves each scalar within 2.4e-15 while changing matched product-activation correlations to as low as 0.098. Individual products therefore are not uniquely identified by that feature's function.

The joint rank-eight fits also have weak component agreement across starts: mean matched signed rank-one tensor cosine is about 0.45–0.46. Similar function errors do not establish stable components. The joint-moment refit was not shown to resolve this ambiguity.

The retained baselines remain the independently confirmed 16-product program and the smaller shared-input graph with limited validation. Stable identification and semantic selectivity are still open. Further work should distinguish basis-invariant reusable computations from arbitrary product coordinates before claiming circuits have fallen out of the factorization.

Receipts under `direct_tensor_match`: `JOINT_PRODUCT_TOYS_V1.json`, `MIDPOINT_PRODUCT_GAUGE_V1.json`, `MIDPOINT_JOINT_PRODUCTS_V1.json`, `MIDPOINT_JOINT_MOMENT_REFIT_V1.json`, `MIDPOINT_CP8_SWAP_V1.json`, and `MIDPOINT_JOINT_METRIC_AUDIT_V1.json`.
