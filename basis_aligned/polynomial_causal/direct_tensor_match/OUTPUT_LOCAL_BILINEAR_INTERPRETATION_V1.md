# Why independent output decompositions lose useful sharing

22 September 2026, 03:10 UTC.

Allowing each output its own input features makes the last-MLP quadratic approximation much better than using the previously tested shared orthogonal basis. However, it remains worse than the exact native computation at the same total product count. **We should retain cross-output sharing when searching for a simpler circuit.** This is a restriction-specific result, not a failure of tensor decomposition in general.

The target is the same 16 selected output projections of the native last MLP described in the [fixed-basis report](../explanations/for_logan/research_update_2026-09-22_0307_sparse_quadratic_baseline.md). It is quadratic in all 1,152 last-MLP input coordinates. It is not the two-layer quartic approximation, a full-vocabulary decomposition, or a text-state evaluation.

## A baseline with a solved optimization problem

For each output, write its symmetric quadratic matrix as

$$
T_v=\sum_i \lambda_{vi}q_{vi}q_{vi}^{\top}.
$$

A real product of two linear forms contributes a symmetric matrix with at most one positive and one negative eigenvalue. A sum of $k$ such products therefore has at most $k$ eigenvalues of either sign. Conversely, retain the $k$ largest positive eigenvalues and the $k$ largest-magnitude negative eigenvalues, and pair opposite signs:

$$
(\sqrt{\lambda}\,q^\top x+\sqrt{\mu}\,r^\top x)
(\sqrt{\lambda}\,q^\top x-\sqrt{\mu}\,r^\top x)
=\lambda(q^\top x)^2-\mu(r^\top x)^2.
$$

Unpaired terms remain signed squares. The eigenvalue tail gives the best Frobenius approximation subject to this positive/negative inertia restriction. Thus there is no gradient optimizer or initialization uncertainty in this particular baseline. It does not solve the shared multi-output circuit problem.

Centered isotropic Gaussian error equals coefficient Frobenius error. This measures reconstruction of varying outputs after centering each function by its own mean. Uncentered errors are also recorded in the JSON; they do not replace this measure.

| Products per output | Total products | Pooled centered error | Largest per-output centered error |
| ---: | ---: | ---: | ---: |
| 8 | 128 | 86.12% | 92.28% |
| 16 | 256 | 82.01% | 88.34% |
| 32 | 512 | 75.78% | 81.96% |
| 64 | 1,024 | 65.88% | 71.38% |
| 128 | 2,048 | 50.43% | 54.54% |
| 288 | 4,608 | 24.16% | 25.21% |
| 512 | 8,192 | 4.47% | 5.06% |
| 1152 | 9,831 | 0.00% | 0.00% |

The final row retains the complete floating-point spectrum. Its 9,831 products are this construction's count, not an interval-certified minimum for exact symbolic weights. At 288 products per output, the registered pooled-error prediction of below 20% **fails** (24.16%); the prediction that every output is below 30% **passes** (maximum 25.21%).

## Was uniform allocation the problem?

Partly. A descriptive follow-up optimizes the allocation of a fixed total product budget. Each next product retains one positive and one negative eigenmode where available. Its energy gain decreases with the local product count, so sorting these marginal gains gives the exact optimal allocation for the chosen additive coefficient metric. Ten exhaustive small allocation controls verify this implementation.

At a total budget of 4,608 products:

| Allocation | Natural pooled centered error | Equal-output centered error | Worst output |
| --- | ---: | ---: | ---: |
| Uniform: 288 per output | 24.16% | 24.07% | 25.21% |
| Optimized for natural output energy | 17.44% | 29.02% | 42.83% |
| Optimized after giving outputs equal coefficient energy | 24.70% | 24.04% | 25.13% |
| Native shared-channel computation | 0% | 0% | 0% |

Natural weighting assigns between 174 and 500 products per output; equal weighting assigns 280–303. The pooled improvement therefore sacrifices some smaller outputs, echoing the issue found in text-state residual analysis. This follow-up does not retroactively change the failed uniform-allocation prediction.

The adaptive implementations store roughly 10.61 million reader coefficients before signs and destination metadata, versus 10.69 million floats in the native selected-output implementation. They save essentially no coefficient storage at this budget. For a signed square, the repeated reader is stored once. The original native program reuses each channel's product across output directions, whereas these independent dictionaries do not.

## Research consequence

Two distinct restrictions have now been exposed. The fixed shared orthogonal basis is a poor basis for sparse interactions. Independently optimized output dictionaries avoid that basis restriction but discard valuable cross-output reuse. Neither is a suitable replacement for the native map. A joint learned dictionary with output-specific low-rank slices and shared products remains a materially different hypothesis.

These are coefficient/function reconstruction results. No semantic labeling, OOD prediction, selective removal, or component composition follows. The learned quartic residual and native-removal geometry jobs remain separate managed-queue experiments.

Five planted matrix families pass explicit signed-product replay and eigen-tail checks. Native folding agrees with the original factors to $1.93\times10^{-15}$ relative error; all 16 truncated factor reconstructions agree within $3.41\times10^{-15}$. CPU float64, two threads; complete native spectral audit took approximately four seconds.

[Preregistered restriction and predictions](OUTPUT_LOCAL_BILINEAR_PLAN_V1.md) · [Results, allocations and every output error](OUTPUT_LOCAL_BILINEAR_V1.json) · [Executable audit](audit_output_local_bilinear.py) · [Exhaustive allocation controls](OUTPUT_LOCAL_ALLOCATION_CONTROLS_V1.json).
