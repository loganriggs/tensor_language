# Practical GPU cost of the complete interaction generator

Frozen13September2026. The matched FP64 CPU benchmark favors shared generation on repeated interventions. Its practical competitor is nativeFP32 execution, not FP64 direct algebra. This test keeps the validated mixed-precision generator unchanged.

Fixed historical prefixes0 and96. For each,1/4/12pairs from a deterministic amplitude schedule; three changed branches per pair. Four implementations: native serial, native batched, shared serial, shared batched. Batching stacks all changed branches across pairs with first-layer values expanded; sequence positions and causal semantics remain unchanged. Return an identical stacked state tensor in every implementation.

A: relative post10 state difference from native serial <=1e-5 for every implementation/workload.
B: shared batched at least10%faster than native batched at12pairs on both prefixes.
C: shared batched no slower than native batched at1pair on both prefixes.

Seven measured repetitions rotate method order, with CUDA synchronization before and after each call. Shared global/context preparation is included; common pristine states, static weight loading/conversion and suffix are excluded. No peak-memory or whole-model claim. Native branches use FP32 model operations; shared generates the preMLP10 input inFP64, rounds toFP32 and executes nativeMLP10, matching the validated formulation.

Null: higher precision, attention algebra or preparation cost defeats CPU-observed savings on this GPU. A miss does not negate the response identity or behavioral evidence; it would reject this implementation as a practical speed improvement. Conversely, a pass does not establish freshOOD prediction or whole-model compression.

Managed lane1 only;180second cap. Source/dependency binding in COMPOSED_JOINT_GPU_COST_V1_BINDING.json. Existing full-panel native fidelity is the positive reference; state replay also gates timing here. No fit, optimization restarts or precision changes in this experiment.
