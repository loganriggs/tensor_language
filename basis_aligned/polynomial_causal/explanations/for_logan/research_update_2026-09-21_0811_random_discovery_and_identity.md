# Random starts recover the function, but not the same product features

21 September 2026,08:11 UTC. Follow-up to the [response-metric study](research_update_2026-09-21_0754_response_metrics.md).

**Randomly initialized product directions learn useful approximations to the folded source tensor at the same program cost. The result does not require inheriting the previous decomposition. However, those fits still disagree sharply about their internal products, and do not repair the derivative-fidelity gap.**

This is evidence for direct constrained optimization from weights. It is not yet evidence that identifiable semantic circuits automatically emerge.

## Direct random-student experiment

The target comprises four quadratic source reads feeding two selected components. A third component keeps its existing private branch and remains evaluated. The student has256shared mixed products,

$$
\widehat q_o(z)=\sum_{r=1}^{256}w_{or}(l_r^\top z)(r_r^\top z)
+\text{fixed-centered-affine corrections},
$$

with input directions initialized independently at random. The centered linear and mean contributions of the original reads are preserved through the corrections. Output weights are solved analytically as directions are optimized. Including the private third branch, the program retains512source products and897,804floating coefficients.

Eight Adam fits compared coefficient-only and coefficient-plus-source-gradient objectives, learning rates0.02/0.05, and seeds816/1816. Every random arm received2,000cosine-decayed steps. The objectives use the previously frozen covariance geometry; this is weights-first fitting with data-informed metrics, not a wholly data-free experiment. Native component values were evaluated but did not choose the winning fit.

| Fit | Coefficient error | Component1/2/3 value errors | Component1/2 derivative errors |
|---|---:|---|---|
| Previous inherited parent |8.71%|2.65%,2.48%,11.94%|20.09%,17.93%|
| Selected random coefficient-only fit |8.78%|2.56%,2.55%,11.94%|20.22%,17.95%|
| Selected random derivative-weighted fit |9.23%|2.09%,2.81%,11.94%|18.50%,17.57%|

The four coefficient-only fits cluster at8.779–8.787%coefficient error. Their closeness supports repeatable functional approximation under this parameterization. It does not prove a global optimum. The inherited parent had substantial prior optimization, so its comparison is not matched for total lifetime compute.

The selected random programs pass the existing scalar limits: every component error at most15%and at most1.10times the separate baseline. The primary derivative-weighted fit fails the required10%derivative improvement for both components. The third component's known fresh-panel failure remains unresolved.

All errors above use previously opened states. There was no new fresh native-model confirmation in this experiment.

## The internal dictionaries still do not agree

After allowing product permutation and scale/sign gauges, the selected random coefficient-only fit has median atom cosine **0.054** with the parent; no matched atom exceeds0.90cosine. Yet the two fitted coefficient tensors differ by only **4.49%**. Allowing arbitrary linear combinations does not recover the whole dictionary either: median principal cosine is0.119.

The derivative-weighted random fit also has poor atom alignment, although it changes the objective as well as the initialization. The coefficient-only comparison is the cleaner identity control.

The evidence now extends beyond small perturbations of one start. It supports stable approximation quality with unstable product identities. Because the fitted tensors are approximate and differ, it does **not** prove that the exact original tensor has multiple identical minimum-cost decompositions.

## Two explanations tested and limited

First, downstream factors correlate with source-gradient errors. For component1, treating those factors as independent understates squared error by20–27%; component2is within6%. A coupled readout solve included the actual training-state factors and their cross terms while keeping product directions fixed. It passed explicit design-matrix checks but changed derivative errors negligibly. This correction alone does not solve the problem.

Second, similar output directions might allow product features to rotate without changing the function much. For exactly equal output weights, rotating a pair of left factors and the corresponding right factors by the same orthogonal matrix preserves their summed contribution. A planted exact control passes.

On the trained fit, pairing nearby output directions and rotating each pair45degrees changes atom identities substantially, but changes the joint function by **13.51%**, failing the2%limit. Random pairing is worse at75.87%. Despite a median output cosine of0.9993, the forced matching contains poor tail pairs; two of those account for most individual-pair damage. The median concealed a consequential tail.

Thus this particular finite rewrite does not explain the observed instability while preserving the function closely. Other pairings or more general transformations remain possible. Neither output similarity nor a well-conditioned product Gram matrix establishes feature identifiability.

## Methods, scope and receipts

Random fitting took200.30seconds on the managed GPU, with no native-model forwards. FP64, disabled TF32, implicit/dense objective replay, and the existing planted loss/recovery controls apply. The shared executor gained configurable filenames and random seeds; the new plan binds its hash, and historical defaults remain available. The separate CPU controls use1536training sites and448opened evaluation sites from historical chunks, which lack verified document independence.

Conditional derivatives hold the later native input h fixed; they are not full upstream causal derivatives. The model's explicit normalization and final softcap have not been removed. Cost counts exclude generating native inputs. Random-start success does not close extraction, OOD behavior, selective manipulation or semantic reuse.

- [Random-start plan](../../direct_tensor_match/SOURCE_RANDOM_START_PLAN_V1.json), [native-size preflight](../../direct_tensor_match/SOURCE_RANDOM_START_PREFLIGHT_V1.json), and [all eight fit results](../../direct_tensor_match/SOURCE_RANDOM_START_FIT_V1.json).
- [Independent-start identity audit](../../direct_tensor_match/SOURCE_RANDOM_IDENTITY_V1.json).
- [Downstream dependence](../../direct_tensor_match/DOWNSTREAM_WEIGHT_DEPENDENCE_V1.json) and [coupled fixed-readout control](../../direct_tensor_match/DOWNSTREAM_READOUT_V1.json).
- [Finite output-sharing rewrite control](../../direct_tensor_match/OUTPUT_PARALLEL_REWRITES_V1.json).
