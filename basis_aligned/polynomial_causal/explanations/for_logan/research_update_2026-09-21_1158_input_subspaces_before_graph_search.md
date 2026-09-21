# The input subspaces must improve before graph rewiring can suffice

21 September 2026, 11:58 UTC. **Changing products inside the latest graph's fixed input subspaces cannot meet its original coefficient-accuracy requirement.** Even an unrestricted dense quadratic core leaves 8.08% covariance-shaped error, above the 7.73% limit. This narrows the next useful experiment: the shared input directions must also be allowed to move.

This is a diagnostic for the smaller six-read target described in the [overall review](research_update_2026-09-21_0104_full_coverage_and_shared_baselines.md), not a result for the whole folded model. It adds no fresh behavioral or causal evidence.

**What the diagnostic removes**

For each component pair, the graph's shared and private projections span a 352-dimensional subspace of the 1,152-dimensional input. Let $U$ be an orthonormal basis for that subspace, and $Q$ one target quadratic form in the chosen fitting coordinates. The best unrestricted quadratic form within the subspace is

$$
\widehat Q=U(U^\top Q U)U^\top.
$$

It minimizes coefficient Frobenius error among all forms $UKU^\top$. This gives the graph every possible quadratic interaction in its current input span, removing its sparse product restrictions. The remaining error cannot be repaired by changing product wiring while keeping that span fixed.

| Covariance-shaped fit | Coefficient error |
|---|---:|
| Latest trained product graph | 8.224% |
| Best dense cores in its same pair subspaces | 8.078% |
| Original required limit | **7.734%** |

The other restart gives essentially the same result: 8.086% with unrestricted cores. The coefficient-optimal primary also gives third-component error 12.63%, compared with 13.53% for the product graph. That component value is a diagnostic, **not a lower bound on all possible component errors**: optimizing a different loss could change it.

**Does replacing the private subspace solve it?**

We retained the learned 128-dimensional common subspace and selected new 224-dimensional private subspaces from the target weights. One construction uses each pair's mode Gram matrix, the sum of the squares of its two symmetric target forms. A second emphasizes interactions with the common subspace. Both are heuristics, not global optima.

| Dense-core support construction | Covariance coefficient error |
|---|---:|
| Current learned shared/private spans | **8.078%** |
| Same shared span, mode-Gram private spans | 8.182% |
| Same shared span, cross-emphasized private spans | 8.187% |
| Weight-derived common span, mode-Gram private spans | 8.970% |
| Independent 352-dimensional pair spans, without a sharing constraint | **7.587%** |

The independent pair construction shows that 352 dimensions per pair are not themselves too few for this coefficient threshold. The particular common-subspace constraint or how we optimize it matters. It does not prove that a better common subspace exists at the same cost, nor that independent pair programs are cheaper.

The new common span was selected from the sum of the three pair-normalized target mode Grams. It performed worse than the learned common span. These simple weight-derived replacements do not resolve the problem.

**How this connects to the toy recovery experiments**

The mixed shared/private product topology is exactly capable of representing its five planted targets, but unrestricted random fitting recovered only three within 1%. Longer Adam/Muon fits and L-BFGS polishing retained that miss. Restricting directions to subspaces inferred from the target matrices recovered four of five, meeting the existing toy gate.

That is evidence for decomposition-guided fitting on these toys. It does not validate a native fit automatically: the toy targets have exact low-rank supports, whereas the native target forms have full support. Moreover, the fixed approximate supports tested here are insufficient for the registered native coefficient threshold.

**What changes next**

The planned fixed-span cross-product fit should not be launched as a solution to the original fidelity target. A useful next diagnostic is to optimize the common and private subspaces with unrestricted cores, separating the difficulty of choosing input directions from the difficulty of representing their interactions cheaply. If that relaxed problem can pass, its learned subspaces can propose a new graph. If it cannot, merely changing the sparse core is unlikely to help at this layout and budget. A failed optimization would still not be a global impossibility proof.

This remains stage-one guidance for a possible stage-two graph. The eventual program must pay for its dense linear maps and cores, preserve individual components, survive fresh interventions and expose reusable, stable computations. None of those requirements is waived by this projection test.

**Evidence and execution details**

- [Learned-span projection results](../../direct_tensor_match/JOINT_READER_SPAN_V1.json) and [implementation](../../direct_tensor_match/audit_joint_reader_span.py).
- [Private-support alternatives](../../direct_tensor_match/SHARED_PRIVATE_SUPPORT_V1.json) and [weight-derived common-support alternatives](../../direct_tensor_match/WEIGHT_DERIVED_COMMON_SUPPORT_V1.json): CPU, float64, two threads, approximately 5 and 6 seconds. Projection residuals are orthogonal to represented cores to numerical precision.
- [Support-guided toy results](../../direct_tensor_match/SUPPORT_GUIDED_TOY_V1.json): five unchanged targets, two random coefficient starts, 3,600 Muon steps at initial rate 0.03; four of five recovered within 1%, 62.81 seconds. The remaining target's best error is 2.687%.

The native diagnostics use saved weights and the previously examined 448 states where component values are reported. No new model forward pass, OOD validation, circuit adoption or runtime speedup is claimed.
