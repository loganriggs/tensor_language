# Does the complete shared interaction generator save computation?

13September2026. Yes on the tested repeated-intervention CPU workload, including preparation, although batching the direct reference substantially narrows the advantage. This is not a GPU or whole-model speed result.

The previous native validation established fidelity of the original interaction when generating child, remainder and joint branches. Here the target is all three post-MLP10 branch states at each requested amplitude pair. Every method returns the same set of states; prefix and suffix generation are outside this conditional operator.

## Matched implementations

The direct version recomputes MLP9, attention10 and MLP10 at each changed input. The response-only version replaces MLP9 recomputation with the exact fixed-writer response but projects each changed attention input directly. The shared version also prepares common response/attention projections once, then combines them for each amplitude. Both latter versions retain dense MLP10 and the full attention calculation. The prepared context is rebuilt inside each timed workload; its global-cache construction is charged too.

The first comparison used sequential changed branches. A second executed discriminator batches all three amplitudes per pair, across all requested pairs, in both direct and shared versions. Sequence length and causal mask remain unchanged; the intervention index becomes a batch index. First-layer values are expanded consistently. All versions retain the same output list. Seven timing repeats rotate method order. Branch replay passes the registered1e-10relative bar in both experiments.

Actual model weights, one synthetic17-position context, FP64 algebra and two CPU threads were used. Each pair requires changes at a, b and a+b. Pristine inputs are supplied equally. The fixed-writer response has no fitted approximation.

## Results including preparation

| Amplitude pairs | Direct serial | Shared serial | Direct batched | Shared batched | Matched batched speed ratio |
|---|---:|---:|---:|---:|---:|
|1|66.99ms|45.09ms|47.03ms|43.92ms|1.07×|
|4|267.76ms|139.07ms|144.18ms|94.89ms|1.52×|
|12|802.05ms|387.90ms|420.85ms|257.77ms|1.63×|

The first unbatched study gave approximately1.48–2.08×speedup over direct recomputation. Its response-only version was faster than shared preparation for one pair,40.75ms versus44.39ms. At twelve pairs the shared version beat response-only by1.27×. Thus neither all of the gain nor all of the cost is attributable to attention projection sharing.

The batching discriminator confirms that serial small matrix multiplications inflated the apparent advantage. It does not eliminate the repeated-intervention benefit. The roughly7%single-pair gain is small and should not be treated as a robust hardware-independent performance claim. Both registered twelve-pair speed bars hold in their respective experiments.

## Literal local weight inventory

The conditional direct operator uses39,815,424weight scalars. Replacing the MLP9 response computation with its fixed mixed map reduces this to25,217,280, a36.7%reduction in this local inventory. Common attention10/MLP10weights account for23,889,024scalars. Direct-specific weights account for15,926,400; the folded program needs1,328,256.

This is **not** a36.7%model reduction. The native MLP9weights are still required to generate the pristine context and to handle arbitrary other directions; storing its additional fixed mixed map can increase the complete model's storage. Prepared projection/state banks also use memory beyond this constant-weight inventory. Earlier context-memory measurements remain separate; these runs do not measure peak memory. A standalone conditional interface can be smaller while its integration into the full model is larger.

Graphically, the simplification replaces repeated MLP9 product computations with one common context preparation and amplitude-dependent scalar combinations. Attention projection preparation serves multiple branch consumers. This is demonstrated computational reuse within one interaction family, not identification of a shared semantic circuit across unrelated behaviors.

Next performance discriminator: compare the validated mixed-precision generator with native FP32 execution on the managed GPU, including preparation. The present FP64CPU result cannot establish that practical GPU comparison. Continue preserving the full original-interaction target during further compression.

[Initial comparison code](composed_joint_cost_v1.py) · [Initial receipt](COMPOSED_JOINT_COST_V1_RESULT.json) · [Batched countercheck](composed_joint_cost_batched_v1.py) · [Batched receipt](COMPOSED_JOINT_COST_BATCHED_V1_RESULT.json) · [Native fidelity evidence](COMPOSED_JOINT_RESPONSE_V1_MATH.md).
