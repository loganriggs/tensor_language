# Learned output blocks: converged fits, limited gain and a contrast tradeoff

13 September2026, setting2 weights-only compression. This fits twelve individual token numerator readers, preserving the information needed for later separate token softcaps. Final normalization/softcap execution itself is not included in this coefficient fit.

## The fitted family and optimization

Let \(T\in\mathbb R^{12\times1152\times128}\) be the mixed residual/head-write tensor. We fit

$$
\widehat T_o=\sum_{j=1}^{12}Q_{oj}A_jB_j,
\qquad Q^TQ=I,\quad A_j\in\mathbb R^{1152\times32},
\quad B_j\in\mathbb R^{32\times128}.
$$

For each \(Q\), the best blocks are rank32 SVDs of \((Q^TT)_j\). This variable projection eliminates all block-reader optimization; the nonlinear search moves only the output basis. It is an orthogonal-output LL1 restriction, not general overlapping output blocks.

The objective is squared coefficient Frobenius error divided by \(\|T\|_F^2\). A small skew-matrix exponential parametrizes updates to \(Q\). Gradients use the residual of the exact block fit, followed by the derivative of that small exponential. Actual-weight directional finite-difference checks agree within3.66e-8 relative. We separately measure the intrinsic gradient on the orthogonal group, preventing a small chart gradient from being mistaken for stationarity.

Ten starts (identity, output spectral basis, eight random bases) receive15 iterations each. The best three receive longer refinements with chart recentering available. All three reach intrinsic gradient norms3.61e-8–5.23e-8; these are converged local fits. The seven other starts were screened, not run to convergence. The managed GPU run took140.70seconds.

## Registered results

- A objective replay: pass, GPU/CPU normalized-loss difference2.89e-15.
- B at least two promoted fits stationary: pass, all three.
- C at least10% relative squared-loss improvement over the best initial basis: **fail**, improvement0.8456%.

Best relative coefficient error is68.2924%, compared with68.5830% for the best initial spectral basis. The mixed representation stores491,664 scalars versus1,769,472 in the dense mixed tensor; other response terms and normalization/background machinery remain outside that local price. At this error it is not adopted as a faithful compressed circuit.

Two refined starts produce nearly identical functions. The best spectral-start basin differs from them with function cosine about0.92986, despite squared losses differing by only5.18e-7. Low loss differences and local convergence do not establish a unique factorization or stable circuit identification.

## Executed red team: fitting common token content instead of their difference

Transform each token pair into orthonormal sum/difference coordinates:

$$
T_{+,j}=(T_{UK,j}+T_{US,j})/\sqrt2,\qquad
T_{-,j}=(T_{UK,j}-T_{US,j})/\sqrt2.
$$

The pair sums carry85.22% of the target squared coefficient norm; differences carry14.78%. Squared residual norms partition exactly under the same transformation. No fitting weights were changed for this audit.

| Representation | Total error | Pair-sum error | Spelling-difference error |
|---|---:|---:|---:|
| Separate token blocks | 68.69% | 66.39% | 80.66% |
| Spectral output basis | 68.58% | 68.53% | 68.90% |
| Best learned basis | 68.29% | 65.80% | 81.19% |

The learned total objective improves while the spelling-difference objective regresses relative to the spectral basis. This is a measured tradeoff, not a claim of task-specific causal damage: these remain coefficient errors before native normalization and softcaps. A future balanced mean/difference objective would be a separately registered objective, not a repair of this result. It can be defined from weights without training on text.

## Counter-review: what can stronger optimization still change?

For any LL1 decomposition, the input unfolding rank cannot exceed the sum of its matrix-block ranks. Twelve rank32 blocks therefore have input rank at most384, even without the orthogonal-output restriction. The actual target unfolding gives a numerically evaluated coefficient-error lower bound of40.19% at that budget. Thus a substantially better basin may exist between40% and68%; three stationary starts do not prove a global optimum. But no optimizer can achieve2% coefficient error in this rank-budget family: that tolerance requires total block rank at least1105 for this target.

This is not an arithmetic-DAG lower bound: full-rank maps can still have simple sparse computations. It also does not constrain the same computation on the actual coupled attention-producer inputs. Do not spend more identical rank32 output-rotation restarts hoping for high fidelity. Change the graph/factor assumptions, the retained producer scope, or explicitly compare a different output objective. The third full-vocabulary setting remains part of the campaign and must not be forgotten.

Primary receipts: `HEAD17_OUTPUT_BLOCK_FIT_V1_RESULT.json`, `HEAD17_OUTPUT_BLOCK_FIT_V1_PROGRAM.pt`, `HEAD17_OUTPUT_MEAN_CONTRAST_V1_FIT_AUDIT.json`, `HEAD17_OUTPUT_BLOCK_RANK_BOUND_V1.json`. The uncompressed retained predictor remains the behavioral reference.
