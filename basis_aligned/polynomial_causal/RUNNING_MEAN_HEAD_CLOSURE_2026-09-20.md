# Head 1.8: closing the diagonal dependency

A parameter-free negative prefix mean can replace the entire routing pattern of head1.8 with low language-model loss damage on these opened panels. It nevertheless misses38–42% of native head output, and complete head removal is also fairly cheap. This is a compact replacement candidate, not an identified faithful semantic circuit.

## Extracted computation

For the mixed value stream v_t in R^128, the zero-diagonal candidate is y_0=0 and y_t=-(sum_(s<t) v_s)/t. It uses one128-value prefix accumulator, O(T*128) arithmetic, no Q/K, token tables or fitted routing parameters. A fitted variant adds g*v_t with g=0.010474354, estimated on8192calibration positions.

The earlier v627 program replaced only offdiagonal attention and kept the native diagonal. This test removes that dependency. Values still come from the model; output projection and upstream normalization/source generation are not included in the routing price. The entire model is not extracted. The experiment still computes native Q/K for its instrumentation; the standalone operator does not. No end-to-end speedup is claimed.

## Native intervention results

Each cell uses16documents at the stated length. CE added is nats/token relative to the native model on the same tokens. Output error is L2 difference divided by native head-output L2 norm, measured before the output projection. Calibration rows skip80; all evaluation panels have been opened previously.

| Panel | Length | Replacement | CE added | Head-output error |
| --- | ---: | --- | ---: | ---: |
| fineweb_n192_skip7000.pt | 128 | mean_native_diag | 0.006059 | 0.413 |
| fineweb_n192_skip7000.pt | 128 | mean_zero_diag | 0.005738 | 0.404 |
| fineweb_n192_skip7000.pt | 128 | mean_fit_diag | 0.005686 | 0.402 |
| fineweb_n192_skip7000.pt | 128 | zero_head | 0.013749 | 1.000 |
| fineweb_n192_skip7000.pt | 128 | flip_mean | 0.344739 | 2.013 |
| fineweb_n192_skip7000.pt | 512 | mean_native_diag | 0.005019 | 0.384 |
| fineweb_n192_skip7000.pt | 512 | mean_zero_diag | 0.004878 | 0.381 |
| fineweb_n192_skip7000.pt | 512 | mean_fit_diag | 0.004895 | 0.380 |
| fineweb_n192_skip7000.pt | 512 | zero_head | 0.016765 | 1.000 |
| fineweb_n192_skip7000.pt | 512 | flip_mean | 0.751097 | 1.968 |
| fineweb_n192_skip11000.pt | 128 | mean_native_diag | 0.005633 | 0.429 |
| fineweb_n192_skip11000.pt | 128 | mean_zero_diag | 0.005536 | 0.420 |
| fineweb_n192_skip11000.pt | 128 | mean_fit_diag | 0.005558 | 0.418 |
| fineweb_n192_skip11000.pt | 128 | zero_head | 0.009439 | 1.000 |
| fineweb_n192_skip11000.pt | 128 | flip_mean | 0.312263 | 2.027 |
| fineweb_n192_skip11000.pt | 512 | mean_native_diag | 0.004666 | 0.400 |
| fineweb_n192_skip11000.pt | 512 | mean_zero_diag | 0.004831 | 0.398 |
| fineweb_n192_skip11000.pt | 512 | mean_fit_diag | 0.004885 | 0.397 |
| fineweb_n192_skip11000.pt | 512 | zero_head | 0.015107 | 1.000 |
| fineweb_n192_skip11000.pt | 512 | flip_mean | 0.936694 | 1.968 |

## Registered verdicts and red-team

| Claim | Evidence | Status |
| --- | --- | --- |
| Native-diagonal mean reference adds <=.015 nats | edit, opened panels | passes |
| Fitted scalar closes diagonal with <=.015 nats | edit, opened panels | passes |
| Closed program predicts head output within10% | fold prediction, opened panels | fails |
| Zero removal and sign flip both cost >=.02 nats more than program | edit, opened panels | fails: zero removal too cheap |

Zero-diagonal and fitted-diagonal CE are nearly identical. Thus the positive result does not require the fitted scalar. The negative output-fidelity verdict already occurs with native diagonal, so diagonal simplification does not explain it. Sign reversal costs0.31–0.94nats, but deleting the head costs only0.009–0.017nats. A destructive opposite-sign edit alone does not prove semantic selectivity, indispensability or faithful mechanistic recovery.

49native forwards,1calibration scalar,0model updates. Standalone tests compare against an independent dense pattern and sequential recurrence for lengths1/2/17, positive/negative/zero mass and scalar/tensor diagonals. Two tests pass including a second fold through value/output matrices.

## Further folding and next evidence

With normalized source ports n1,n0 and mixed values v=(1-mix)V1*n1+mix*V0*n0, linearity allows averaging either before or after value/output projection. `execute_projected` implements the thin-factor version using V1,V0 and O, while its test independently forms composed residual-to-residual maps. At native dimensions, those three thin matrices cost442368values plus the mixing scalar; this excludes both source generators. This identity is CPU-tested; native standalone extraction of these particular weights is not yet verified.

Next tests must explain which part of the head output matters to downstream computation, evaluate untouched text families and counterfactual preservation controls, and test the same operator in other heads without choosing parameters on their test data. Ordinary low CE, this algebraic reuse, and the existing multi-head replacement composition are not substitutes for those requirements.

[Native receipt](../bilinear_quotient/circuits/followups/mean_head_closure_v636_result.json) · [standalone operator](running_mean_head.py)
