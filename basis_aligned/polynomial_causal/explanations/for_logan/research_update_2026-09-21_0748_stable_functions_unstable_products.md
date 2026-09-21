# Smaller programs, but unstable internal products

21 September 2026,07:48 UTC. Follow-up to the [overall review](research_update_2026-09-21_0104_full_coverage_and_shared_baselines.md).

**The two-stage graph edit and refit now saves products at matched coefficient storage. Fresh comparisons support that saving, but one accuracy requirement still fails. Separately, the learned product nodes are not stable enough to call recovered circuit units.**

## What the graph refit achieved

Three selected components each require two quadratic input reads from the preceding MLP. The baseline computes each pair separately. The edited graph shares256mixed products across the first two components and keeps256private products for the third. After jointly refitting input directions and analytically solving readout coefficients:

| Program | Source products | Stored floating coefficients | Opened-state component errors |
|---|---:|---:|---|
| Separate baseline |768|897,804|3.06%,2.76%,11.94%|
| Shared graph |**512**|**897,804**|**2.65%,2.48%,11.94%**|

These are errors in scalar feature values relative to the target's variation, on previously opened states. They are not token error rates. The first two branches improve; the third is unchanged.

On a fresh panel of32FineWeb documents and16code files, all72relative comparisons passed: graph error stayed within1.10times baseline error. But the third component's removal effect at FineWeb continuation sites had16.73%error, above the15%cap. Increasing its private capacity in both programs produced576vs832products and matched971,660floating coefficients. A second fresh panel again passed all72relative comparisons but retained the same type of absolute failure at16.01%.

The second failure's recipient-bootstrap95%interval is11.15–24.92%, with donors fixed. The graph and baseline have identical errors for this private branch. Uncertainty does not turn the failed point threshold into a pass. These panels use different capacities, so they are not identical replications.

## Is the decomposition recovering the same internal features?

We compared the four refits after allowing product permutation, scaling and sign changes. A product's joint contribution includes both its quadratic input form and its four output weights:

$$
A_r=w_r\otimes\operatorname{sym}(l_r r_r^\top),
\qquad \widehat T=\sum_r A_r.
$$

Individual products match poorly: median absolute cosine is0.047–0.065. Total fitted functions differ by only4.2–4.4%. This is not explained by straightforward numerical instability: product Gram condition numbers are24–26 and FP32-versus-FP64 scalar discrepancies are below0.00033%.

A stronger test allows arbitrary linear recombinations of products. If the same full dictionary were merely rotated, its span would remain the same. Instead, only3–4of256principal directions match above cosine0.90. The full-span stability hypothesis fails too. A planted invertible-change-of-basis control passes; a rotated-input negative control fails as intended.

## A stable aggregate does not establish simple internal units

Projecting onto directions shared across the fitted spans retains three directions and99.994%of the fitted tensor's energy. Executing that projected function, with the original centered mean and linear terms preserved, gives component errors2.86%,2.48%,11.94% on the opened states. Dense source replay agrees to about2e-15.

But a critical control changes the interpretation: **the leading consensus direction alone carries99.987%of the function's energy, and all256shared products still have nonzero coefficients.** We have mostly rediscovered the entire fitted function as one stable direction. This does not expose three scalar features or create a cheaper circuit.

The arithmetic saving remains valid. The claim that particular internal products are meaningful, reproducible units does not yet have support. Stable downstream-defined observables may be useful units, but their semantic interpretation, selective manipulation and upstream dependencies remain to be established.

## Additional check: value accuracy does not preserve input sensitivity

An analytic derivative screen at fixed later input h finds18–20%error relative to the original components, despite2–3%value errors. All four fits fail the registered15%derivative threshold; their derivatives differ from one another by less than9%. Central differences agree with the analytic code within8e-10. Restricting perturbations to the tangent plane of the normalized earlier input does not remove the miss: winner errors are20.52%and18.61%.

This is a conditional local sensitivity test on448opened states, not a full upstream intervention or a proven comparison against the separate baseline's derivative quality. It identifies a missing response-fidelity requirement for subsequent fitting. [Sensitivity code and results](../../direct_tensor_match/PROFILED_SENSITIVITY_V1.json).

## Methods and receipts

The refit used Adam with learning rates0.005/0.02, pruned and1%-perturbed starts,4,000cosine-decayed steps, and analytic ridge readouts. The weight objective selected the winner. No outcome-based selection among fresh panels was used. Native programs consume earlier normalized input z and later residual input h; normalizations and the final softcap remain explicit. Product counts exclude producing these native inputs.

The fresh panels use identified documents/files and balanced same-current-token donors. “Natural” means removal in the recipient state; “hybrid” first substitutes the earlier source contribution from a donor; “change” subtracts those removal effects. Individual and combined components are scored by domain and token cohort. Historical fitting caches lack equivalent document provenance.

- [Refit and dense audit](../../direct_tensor_match/PROFILED_PARTIAL_GRAPH_AUDIT_V1.json).
- [Fresh panel one](../../direct_tensor_match/PARTIAL_GRAPH_FRESH_NATIVE_V1.json), [panel two](../../direct_tensor_match/PARTIAL_GRAPH_FRESH_NATIVE_V2.json), [panel-two bootstrap](../../direct_tensor_match/PARTIAL_GRAPH_FRESH_AUDIT_V2.json).
- [Atom stability](../../direct_tensor_match/PROFILED_PRODUCT_STABILITY_V1.json), [conditioning and precision](../../direct_tensor_match/PROFILED_CONDITIONING_AUDIT_V1.json).
- [Subspace comparison](../../direct_tensor_match/PROFILED_SUBSPACE_AUDIT_V1.json), [consensus](../../direct_tensor_match/PROFILED_CONSENSUS_V1.json), [executable consensus check](../../direct_tensor_match/PROFILED_CONSENSUS_EXECUTION_V1.json).
- [Mathematical review and primary literature](../../THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-21_0744.md).

Stability analyses are CPU-only in FP64 and compare nearby starts/two rates, not independent discovery datasets. They provide evidence about implementation stability, not a theorem that no identifiable decomposition exists.
