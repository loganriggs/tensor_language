# Better algebraic fitting helps; freezing the discovered input space is too restrictive

21 September 2026, 12:56 UTC. **We improved the cheap graph substantially without adding any operations, but it still misses fidelity. A new rank bound shows why further fitting inside the same input subspaces cannot meet the coefficient target.** The next native experiment allows the private input directions to move while keeping the reusable shared dictionaries and graph cost fixed.

This is still the selected six quadratic reads feeding three components described in the [overall trajectory](research_update_2026-09-21_0104_full_coverage_and_shared_baselines.md). It is not a faithful decomposition of the entire folded section or a standalone circuit.

**What changed in the graph fit**

The [previous conversion](research_update_2026-09-21_1237_stage_one_gain_and_graph_conversion_gap.md) fixed the private quadratic forms and fitted a shared–private coupling. We now alternate exact updates of both. Each update minimizes its part of the coefficient objective; five independent dense-system controls and fifteen perturbed-pair execution controls pass. Joint global optimality is not guaranteed.

All four native candidates were fitted with 200 alternating updates and then compiled. The primary keeps exactly the same **1,056 products, 1,058,124 floating coefficients and 1,047,648 source multiplications**—21.26% less source arithmetic than the original pair baseline.

| Primary covariance fit | Earlier conversion | Alternating fit | Required limit |
|---|---:|---:|---:|
| Covariance coefficient error | 11.15% | **7.994%** | 7.734% |
| Native-isotropic coefficient error | 63.40% | **59.73%** | 60.79% |
| Component 1 error | 2.76% | **2.23%** | 2.06% |
| Component 2 error | 2.92% | **2.36%** | 2.25% |
| Component 3 error | 14.21% | **10.93%** | 9.69% |

The fit improves, and every compiler/execution check passes this time. Arithmetic passes; component and coefficient fidelity still fail. The native-isotropic requirement passes on its own. These component values are diagnostics on the previously examined 448 states.

The primary used all 200 updates. Over its last 20 updates, the three within-span objectives improved only about 0.19–0.24%. That does not prove convergence to a global optimum. The rank calculation below gives a stronger reason not to keep extending this particular fixed-span fit.

**A bound on the actual private-branch structure**

For one pair of quadratic reads, split the input into shared coordinates and their orthogonal complement:

$$
Q_o=\begin{bmatrix}A_o&B_o\\B_o^\top&C_o\end{bmatrix},\qquad o\in\{1,2\}.
$$

The shared branch can represent any $A_o$. Both reads must obtain their remaining terms through the same 224 private features. Consequently, their reconstructed matrix

$$
K=\begin{bmatrix}
\sqrt2 B_1&\sqrt2 B_2\\
C_1&C_2
\end{bmatrix}
$$

has rank at most 224. The factor $\sqrt2$ accounts for the two symmetric cross blocks in each coefficient matrix. The singular-value tail beyond rank 224 is therefore a lower bound on reconstruction error. This allows arbitrary symmetric cores and is more permissive than the final sparse product graph.

| Where private directions may lie | Covariance-error lower bound |
|---|---:|
| The previously learned 352-dimensional pair input spans | **7.798%** |
| Anywhere in the original 1,152-dimensional input, with shared spaces fixed | **7.174%** |
| Required error | **At most 7.734%** |

The first bound excludes meeting the requirement while keeping those pair spans frozen. The second does not exclude success, but does not prove an attainable solution. These are floating-point spectral bounds, supported by fifteen planted rank controls, rather than interval certificates. They do not constrain moving the shared spaces, adding explicit shared–private products, or using a different arithmetic DAG.

This refines the two-stage lesson: **a decomposition can propose useful features without providing the final input subspaces. The graph stage may need to move directions as well as simplify interactions.**

**The obvious spectral construction did not solve it**

We tried selecting new private directions directly from the full-input rank relaxation, then applying the same 200 alternating updates. This retained cost and all acceptance criteria, but the primary covariance error worsened to **11.65%**, with third-component error **14.55%**. All four compiled candidates failed fidelity.

There is no contradiction with the 7.174% bound. The relaxed rank approximation ignores the symmetry requirements on the reconstructed private quadratic forms. A good relaxed approximation need not produce a good feasible graph. Planted cases reconstruct exactly, so this is a retained native miss rather than evidence that the relaxation is an exact algorithm for our model.

**The next fit releases private directions gradually**

We implemented a parameterization that moves the private input subspaces while keeping the shared dictionaries fixed. At each step it solves for the best symmetric quadratic cores exactly. The remaining optimization variables are the private directions and their coupling to the shared features.

The loss gradient uses the optimum-core property: once the cores minimize their conditional objective, the gradient through their optimizer is unnecessary. Five checks compare this gradient with full tensor reconstruction, finite differences and an independent differentiable dense-system solve. This also avoids differentiating an eigendecomposition through repeated eigenvalues.

| Optimizer, rate 0.03 and 1,800 steps | Planted targets recovered within 1%, using the better of two starts |
|---|---:|
| Adam | 2/5 |
| Muon | **4/5** |

Muon also wins the registered aggregate-error criterion. The remaining toy miss is retained: its best error is 1.97%. This meets the existing four-of-five recovery gate, not universal or reliable recovery of every target.

Both full-native-shape preflights pass, including export back into actual product instructions and physical storage checks. Four managed native fits are now registered and running: covariance-shaped and native-isotropic objectives, each from the stronger alternating-fit solution and from a random initialization. The graph cost, primary selection rule, component limits, derivative limits and both coefficient limits remain unchanged. No completed outcome is available at this report revision.

**Evidence and scope**

- [Alternating-fit results](../../direct_tensor_match/ALTERNATING_COMPLETION_V1.json), [perturbed-pair execution controls](../../direct_tensor_match/REFINED_COMPLETION_PREFLIGHT_V1.json).
- [Private-branch rank bounds](../../direct_tensor_match/PRIVATE_BRANCH_RANK_BOUND_V1.json), [planted rank controls](../../direct_tensor_match/PRIVATE_BRANCH_RANK_PREFLIGHT_V1.json).
- [Spectral-direction failure](../../direct_tensor_match/SPECTRAL_PRIVATE_COMPLETION_V1.json).
- [New loss/gradient controls](../../direct_tensor_match/FREE_PRIVATE_VARPRO_PREFLIGHT_V1.json), [Adam/Muon comparison](../../direct_tensor_match/FREE_PRIVATE_TOY_V1.json), [full-size preflights](../../direct_tensor_match/FREE_PRIVATE_NATIVE_PREFLIGHT_V1.json), [native preregistration](../../direct_tensor_match/FREE_PRIVATE_NATIVE_PLAN_V1.json).

The alternating and spectral studies took 54.18 and 57.07 seconds on CPU. The optimizer comparison took 45.78 seconds. No new text calibration or model forward pass was used. Native intermediate states and explicit normalization remain part of the conditional interface. OOD prediction, selective manipulation, stable feature identity and full extraction remain outstanding; no circuit is adopted here.
