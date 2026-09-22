# Giving smaller features more weight did not recover them

22 September 2026, 02:11 UTC. The matched balanced-versus-uniform optimization completed in **2,407 seconds**. Integrity checks passed, but both registered scientific criteria failed. The additional emphasis on smaller outputs gave only a modest improvement in their errors while damaging important response predictions.

## What was fitted?

We are still simplifying the pure quartic path from MLP16's contribution through both inputs of MLP17, measured in the same 16 fixed output directions. Other residual, attention, bias and cross terms are excluded. These 16 directions are an output basis, not 16 examples or identified concepts.

The candidate computes 144 quadratic intermediate features, each a sum of four products of learned linear input directions. It then computes 512 selected products of those quadratic features, and combines them into 16 outputs. This is a shared arithmetic graph with **1,088 variable products, 1,353,728 floating coefficients and 1,024 connection indices**. The graph connections were fixed; the input directions and output coefficients could change.

For each of two learned starting programs, we ran two continuations with the same reset Muon optimizer and 11 updates. One kept uniform output weighting. The other used inverse calibration-output mean-square values, capped at a 1,000-fold weight ratio. Thus smaller outputs received more emphasis, though they were not perfectly equalized. Both arms optimized the same form of exact weight-based coefficient/Gaussian objective, with mean/covariance and the output weights informed by calibration data. Neither selected checkpoints using evaluation errors.

The question was whether poor small-feature fidelity was largely a capacity-allocation problem: could the same graph learn better small-feature computations if the objective cared more about them? Comparing against an equal-budget uniform continuation prevents attributing ordinary extra training to balancing.

## Registered results on the original 2,048-state panel

“Small-feature RMS” means the root mean square of twelve separate relative errors for coordinates 4–15. It differs from pooling their energy, which would again favor the larger coordinates.

| Seed and arm | Pooled value error | Small-feature RMS | Root1 matched-response error |
| --- | ---: | ---: | ---: |
| 1101, uniform | 10.14% | 87.74% | 24.28% |
| 1101, balanced | 12.58% | 84.62% | 38.57% |
| 1102, uniform | 10.37% | 88.96% | 26.31% |
| 1102, balanced | 11.88% | 84.06% | 37.74% |

The registered small-feature criterion required at least a 15% relative reduction in both seeds. The observed reductions were **3.55% and 5.51%**, so it failed. The retention criterion permitted at most a 10% relative increase in the root1 response and sensitivity errors, and at most a 25% increase in pooled value error. Response errors rose **58.9% and 43.4%**, so retention failed. The other retention subconditions passed, but that does not rescue the conjunction. Absolute errors remain far above a 10% component-fidelity target.

Root1 here is one of the fixed output coordinates. The response metric compares finite changes between existing same-token states. It is not a selective semantic intervention. Normalization and final logit operations remain separate in actual model intervention tests.

## Does the larger panel change that conclusion?

No. A subsequent diagnostic evaluated all four frozen programs on the already-opened 16,384 states from 256 documents. The 2,494 response pairs match current token and position across documents and cover all 16 outputs.

| Seed and arm | Pooled values | Small-feature RMS | Pooled matched responses | Small-feature response RMS |
| --- | ---: | ---: | ---: | ---: |
| 1101, uniform | 11.37% | 96.05% | 19.68% | 93.50% |
| 1101, balanced | 13.93% | 94.20% | 23.05% | 90.37% |
| 1102, uniform | 11.77% | 99.53% | 20.90% | 93.96% |
| 1102, balanced | 13.42% | 93.99% | 23.65% | 91.82% |

Smaller outputs improve slightly, but remain poorly reconstructed. Aggregate values and finite responses worsen. This extra panel is an opened diagnostic, not fresh confirmation; the original registered verdict is kept separate.

## What the negative result tells us

Output weighting changes which features the graph learns, but this short matched experiment did not recover accurate reusable components. The result is consistent with an objective tradeoff, limited graph capacity or support, and optimization limitations. It does not prove that output balancing can never work or that every shared DAG fails. Only two warm starts, one cap and one continuation budget were tested.

Uniform continuation itself improves the earlier shared programs' aggregate values, showing that some previous weakness was unfinished optimization. Even so, these shared graphs remain less accurate than the CP parents and conditional programs under the current comparisons. They have different arithmetic/storage prices, so this is not a matched-capacity proof of one architecture's superiority.

The next structural question is whether smaller outputs need distinct computations that the fixed shared dictionary fails to supply. Repeating the same balancing sweep is lower priority than changing that hypothesis or evaluating the already-queued conditional programs inside native interventions. There is no basis to promote a semantic circuit from this result.

## Validation and reproducibility

Initial physical predictions, normal-equation residuals and float32 exports passed the registered integrity checks. The CPU follow-up matched the GPU's original-panel errors within 1e-5 and verified every artifact hash. Its scorer passed exact-match and known multiplicative-error controls. All four arms are retained, including failures. The larger-panel follow-up made no coefficient updates or candidate selections.

[Registered protocol](../../direct_tensor_match/BALANCED_SHARED_FEATURES_PLAN_V1.md) · [Terminal results and all optimization histories](../../direct_tensor_match/BALANCED_SHARED_FEATURES_NATIVE_V1.json) · [Larger-panel diagnostics](../../direct_tensor_match/BALANCED_SHARED_FOLLOWUP_V1.json) · [Follow-up scorer](../../direct_tensor_match/audit_balanced_shared_followup.py).

## 02:16 follow-up: give smaller outputs separate corrections

A different structural test kept the CP parents fixed and added a separate quadratic correction to each of outputs4–15. These corrections come from exact Gaussian moments of the native-minus-parent weights, rather than fitting text labels. Outputs0–3 remain exactly unchanged at this polynomial interface.

The primary version adds192 square operations and235,404 stored coefficients. It improves small-feature values by about7–8%, but their finite responses by only4–5% on the original panel, failing the registered15% improvement requirement. On the larger opened panel, small-feature errors remain roughly53–56%. Even retaining the entire quadratic correction leaves about40–43% error there and has a much larger implementation cost.

So output-local corrections avoid the dominant-output damage seen with balancing, but a few low-degree corrections do not recover the missing computation. This narrows the next structural hypothesis toward richer products or better response-focused discovery. [Derivation, all ranks, prices and controls](../../direct_tensor_match/OUTPUT_SPECIFIC_CORRECTION_INTERPRETATION_V1.md).

## 02:22 follow-up: combine the existing quartic products

The two existing CP fits contain512quartic product features each. Combining all1024products and refitting their output coefficients directly against native Gaussian weight contractions gives only about **2% improvement** in small-output values and responses over the better single-bank scores. Both single-bank controls were refitted under the same objective. The union fails the registered15% improvement screen while doubling variable products to3,072.

This is stronger than the earlier failed averaging test: the union could independently recombine every product, but still had limited useful complementarity. It motivates learning new residual products instead of another combination of the current dictionaries. [Method, costs and complete comparison](../../direct_tensor_match/UNION_CP_DICTIONARY_INTERPRETATION_V1.md).
