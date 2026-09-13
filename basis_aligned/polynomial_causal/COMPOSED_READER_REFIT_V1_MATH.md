# Moving the product readers after folding the private correction

13 September 2026. The fixed-native-product lower bound does not apply when
the two input readers of each product move. This tests that larger class at
4492 products, the largest count compatible with 10% local scalar savings.

Each candidate atom is $S_k=(l_kr_k^T+r_kl_k^T)/2$, with unit-norm reader rows.
The output weights are solved exactly for each reader setting. If $G$ is the
candidate atom Gram and $B$ its inner products with each target output tensor,
the optimal output writer matrix $W$ satisfies $GW=B$. The reduced objective is

$$
\mathcal L=\frac{\|T\|_F^2-\operatorname{tr}(W^TB)}{\|T\|_F^2}.
$$

For differentiation, hold the solved writers fixed and differentiate
$\|T\|^2+\operatorname{tr}(W^TGW)-2\operatorname{tr}(W^TB)$.
The normal equations cancel the writer derivative. This avoids differentiating
through a large linear solve while retaining the exact reduced gradient.
No ridge changes the objective; a dependent candidate dictionary fails the
Cholesky solve explicitly. Row normalization removes trivial scaling freedom,
though permutations and swapping the two readers remain symmetries.

The kernel agrees with a small explicit symmetric tensor to $1.11\times10^{-16}$
in normalized loss. Directional finite-difference gradient error is
$4.88\times10^{-10}$. A full planted dictionary reconstructs within numerical
precision. These validate the objective and derivative, not global optimization.

The actual composed target starts from 4492 norm-ranked native products and
performs three backtracked gradient updates. Output weights are refit at every
trial. Squared coefficient loss falls from 0.0145334 to 0.0137649, a 5.29%
improvement. Relative error moves from approximately 12.06% to 11.73%.
Reader gradient RMS falls from $1.08\times10^{-5}$ to $3.91\times10^{-6}$ but
does not establish stationarity. The run takes 76.54 CPU seconds. The 1%
improvement predicate passes; the 10% error predicate fails.

This is evidence that changing readers improves over the fixed dictionary,
not a converged answer. The next optimization should use quasi-Newton curvature
and a bounded GPU cost check, keeping exact writer solves and monitoring reader
scales and meaningful stationarity. Multiple starts become useful after that
cost and convergence check; repeating many copies of three crude steps is not
a robust comparison. No data fitting, native replacement or adopted parameter
reduction is claimed. The full goal remains open.

Kernel/control: `symmetric_product_varpro_v1.py` and
`SYMMETRIC_PRODUCT_VARPRO_V1_CONTROL.json`. Actual fit:
`composed_reader_descent_v1.py`, `COMPOSED_READER_DESCENT_V1_RESULT.json`.
Keep subsequent fitting results in this primary note.

## Managed GPU curvature fit

`ops/run_composed_reader_lbfgs_v1.py` uses the same 4492-product initialization
and exact FP64 objective with L-BFGS, eight curvature-history pairs, strong-Wolfe
line search and a 40-iteration/80-evaluation setting under a 900-second alarm.
The initial GPU loss must replay the CPU value within $10^{-8}$ relative error.
Stationarity is measured again after converting readers to unit coordinates:
gradient RMS at most $10^{-7}$ is a separate predicate from reaching the iteration
budget or the 10% coefficient-error target. Reader norm ranges and evaluation
costs are recorded to expose scale-gauge stopping or poor GPU return on cost.

The fit saves a frozen FP32 reader/writer program, and recomputes its exact
coefficient loss in FP64 to check the storage conversion within $10^{-6}$
absolute normalized loss. This is an implementation tolerance, not a native
effect-preservation bar. The first run is a cost/convergence pilot, not a
multiple-start recovery claim. Binding and terminal receipt are
`COMPOSED_READER_LBFGS_V1_BINDING.json` and
`COMPOSED_READER_LBFGS_V1_RESULT.json`; inspect the latter and live runner state
before launching any continuation.

The first GPU fit completed in 37.95 seconds, 36 L-BFGS iterations and 49
objective/gradient evaluations. Typical evaluation cost is 0.71 seconds.
Relative error reaches 11.6786%; the unit-coordinate gradient RMS is
$8.55\times10^{-10}$, passing the declared local stationarity criterion.
Reader norms remain between approximately 1 and 1.0284. The CPU initial loss
replays to $1.19\times10^{-16}$ relative difference, and FP32 storage changes
normalized loss by roughly $5\times10^{-16}$. Peak allocated GPU memory is
4.43 GB. This is a locally stationary miss of the 10% fidelity target, not a
global absence result. The frozen program is 62.10 MB including serialization.

The measured cost justifies a ten-start countercheck through the same managed
GPU lane: composed/upstream norm-ranked supports, each crossed with reader
perturbation scales 0, 0.01, 0.03, 0.1 and 0.3 using fixed distinct seeds.
Each gets 60 iterations/100 evaluations and reports its final unit-coordinate
gradient separately. These are structured and perturbed starts, not arbitrary
global initialization coverage. Total alarm budget is 1800 seconds. Only the
best frozen program is retained; no fit uses native text or behavioral scores.
Follow `COMPOSED_READER_MULTISTART_V1_RESULT.json` and its progress receipt
before deciding whether any apparent negative is robust to these starts.

## Joint-graph pricing audit while the ten starts run

The standalone price cannot be used for the current complete interaction
executor. `composed_joint_response_v1.branch` still needs the full pristine
MLP9 output $m_0$ through both response basis vectors, and
`response_attention_projection_v2` reads it for changed Q/K/V projections
and response normalization. Thus those consumers prevent deleting the original
MLP9 producer when replacing only $J_{10}B_9$.

The native producer stores 15,925,248 scalars. Once its output is available,
the exact compiled downstream consumer adds only 1,327,104 scalars. Their
combined subgraph costs 17,252,352. The independent learned factorization
costs 15,524,352 scalars: a 10.016% standalone reduction, but adding it beside
the still-required producer costs 31,449,600—an **82.29% increase** for this
joint subgraph. Common context, biases and other model components are excluded
from both counts. Learned input readers cannot be counted as shared native
readers merely because they started there.

This resolves the earlier conditional pricing caveat: even a fidelity pass
would not justify adopting this candidate as joint-circuit compression. Let
the bounded ten-start comparison finish as evidence about the representation,
but do not escalate it on standalone savings. An adopted change must either
replace the common producer with all its consumers preserved, or improve the
incremental consumer while reusing $m_0$. The existing exact shared-tail
implementation already follows the latter dependency structure. Receipt:
`COMPOSED_READER_JOINT_PRICE_V1_AUDIT.json`.

All ten starts completed in 442.19 seconds. Every arm passes the declared
unit-coordinate gradient criterion (maximum $1.87\times10^{-9}$). The five
composed-support starts return 11.6786% error; the five upstream-support starts
return 12.0344%. Several upstream arms also reach their 60-iteration setting,
so preserve that fact alongside their independently small gradients. No arm
meets 10% fidelity. Different initial supports reach different locally stationary
values; this is robustness to the tested perturbations, not a global optimality
certificate. The best saved FP32 program replays coefficient loss numerically.
The cost and shared-dependency findings do not justify further escalation of
this particular standalone fit as a joint-circuit compression candidate.
