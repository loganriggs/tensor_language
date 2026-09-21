**A local graph now passes reconstruction checks at a lower cost**

21 September2026,14:51UTC. The two-stage direction has reached a useful local milestone: a compiled shared graph now meets all of our existing coefficient, component-value and source-derivative requirements on the examined states, while costing less than the original pair-program baseline. Fresh behavioral evaluation has not happened yet.

This is still the smaller diagnostic problem—six quadratic reads feeding three selected components—not the full folded MLP or a token-to-output circuit. The full Tucker/HT-to-general-DAG objective remains open.

**What changed**

The fitted shared graph was cheap but inaccurate. Adding correction products based only on large coefficient errors helped too little. We then refitted their amplitudes against the worst normalized fidelity error. Convex conditional solves showed that several fixed direction sets could not pass their requirements; simply optimizing their amplitudes longer was insufficient.

A graph edit driven by downstream fidelity replaced one correction direction and reduced component-three error from9.83% to9.08%, at unchanged cost. A random replacement did not produce that improvement. A subsequent edit helped component one but worsened the overall objective, so it was rejected. Smaller rotations gave only a tiny gain.

Each correction product can contribute to both source reads. Enabling those two coefficients uses the same products and the same already-charged readout storage. We alternate coefficient fits with one read held fixed: each conditional fit is convex, although the joint problem is not.

**The cost tradeoff**

The original20%source-saving requirement remains a failed experiment. We separately tested larger correction dictionaries without weakening any reconstruction limits:

| Added correction products | Source multiplications | Saving versus original | Reconstruction outcome |
|---|---:|---:|---|
|14|1,063,818|20.05%|Fails fidelity|
|32|1,084,608|18.49%|Passes values and derivatives; narrowly fails coefficient limit|
|64|1,121,568|15.71%|Passes all registered reconstruction limits|

The64-product correction sits on top of the existing1,056-product graph, for1,120 nonlinear products total. Dense projections dominate the quoted arithmetic. These are source-operation counts, not measured whole-model speedups.

The passing candidate has covariance-shaped coefficient error7.628% and original-coordinate error57.737%. Its three component errors are1.727%,2.174% and8.346%. Passing means satisfying the previously defined relative tolerances; the large original-coordinate error remains explicit.

**What has been checked, and what has not**

The exported artifact reproduces the recorded scores. Actual graph autodiff matches the dense source derivatives to below3e-14 relative error. FP32 component drift is below8.4e-6. Physical storage is1,131,980 floating-point coefficients.

We also built two independent pair-sharing baselines, one per fitting geometry, each fitting within both the candidate's storage and source-arithmetic budgets. On examined states, the candidate meets every110%relative component and derivative comparison against both. These are weight-derived baselines; they have not received the candidate's same adaptive functional-fitting effort, so this is not yet an optimizer-matched comparison.

The candidate and baselines are frozen before preparing a seventh panel of32new FineWeb documents and16new code files, excluding the six earlier panels. That evaluation will test natural, donor-hybrid and change effects, individually and in combination. No results from that panel are available at this report's timestamp.

Stable semantic identity, selective manipulation, extraction from tokens and reusable composition remain unproven. The current result establishes an executable local approximation worthy of independent validation, not an identified circuit.

[Fit result](../../direct_tensor_match/TWO_READ_CORRECTION_FRONTIER64_V1.json), [independent export audit and baseline comparisons](../../direct_tensor_match/FRONTIER64_EXPORT_AUDIT_V1.json), [pre-evaluation freeze](../../direct_tensor_match/FRONTIER_FRESH_FREEZE_V1.json), [overall two-stage explanation](research_update_2026-09-21_0104_full_coverage_and_shared_baselines.md).

**Fresh evaluation follow-up,15:00UTC.** The corrected managed run completes in7.70seconds. All72absolute cells pass, but five relative comparisons fail: three code/spaced-word/natural cells against the covariance baseline, and two FineWeb/continuation cells against the isotropic baseline. Ratios are1.125–1.148 versus the1.10limit. Allfive paired-document95%bootstrap intervals cross1.10; this uncertainty does not change the registered failures. We retain the frozen graph and preregister one independent replication, with separate and pooled reporting and no candidate refitting. The initial attempt failed due to index tensors being cast to floating point during GPU transfer; the corrected transfer preserves integer indices, and the failure receipt remains. [Fresh results](../../direct_tensor_match/FRONTIER_FRESH_NATIVE_V1.json), [independent audit](../../direct_tensor_match/FRONTIER_FRESH_AUDIT_V1.json), [replication preregistration](../../direct_tensor_match/FRONTIER_REPLICATION_PREREG_V1.json).
