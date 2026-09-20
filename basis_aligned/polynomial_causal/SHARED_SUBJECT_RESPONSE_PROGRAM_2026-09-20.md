# Shared subject-number response program

v654 is the first reduced model in this lineage to pass the registered conditional
effect gate in every evaluation cell. Calibration now contains256 prompts: the
original128 subject-final noun/template combinations and corresponding earlier-
subject prompts. Test nouns remain disjoint; the48-row test panel is already opened.
Native calibration accuracy is90.6–100% across the four batches. Width8, exact
joint contractions, RMS geometry and the5%/10% gates are unchanged.

| Evaluation cell | Error versus dense conditional chain | Error versus full native effect |
| --- | --- | --- |
| Preposed beside, singular | .0194 | .0500 |
| Preposed beside, plural | .0120 | .0343 |
| Preposed beyond, singular | .0206 | .0392 |
| Preposed beyond, plural | .0249 | .0473 |
| Postposed beside, singular | .0216 | .1022 |
| Postposed beside, plural | .0292 | .1125 |
| Postposed beyond, singular | .0134 | .0883 |
| Postposed beyond, plural | .0155 | .0884 |

Local contraction replay is2.34e-11 maximum absolute error. Conditional prediction
passes; full-native prediction fails in two cells. This distinguishes a successful
compression of a conditional surrogate from discovery of the complete requested
circuit. Frozen attention responses remain a material source of error. The same
48 examples cannot now establish fresh generalization, and broader calibration
incurs additional data and native execution cost.

## Shared products and portable execution

[shared_response_runtime.py](shared_response_runtime.py) compiles each symmetric
8x8 input core into36 unordered products and shares them across eight outputs.
Across six stages, it uses216 unique quadratic products and1728 signed quadratic
coefficients. The cores are dense in that pair basis; there is no claim that core
sparsity has been learned. Runtime storage is2582 floating values including carry,
norm geometry, residual scales and the projected final readout, excluding metadata.

Context generation and the producer are separate charged dependencies. The
unpacked producer representation has522,246 values, and a consumer invocation
requires505 per-example context/initial-coordinate values plus answer orientation.
The native context generator is still needed for new prompts. Counts for the
runtime alone cannot be compared to the whole native model.

The v655 export is complete: packed native execution matches the prior joint
executor within 4.44e-14. A separate CPU-only consumer, with no checkpoint or
model object, replays 304 saved calibration/evaluation rows within 3.55e-15.
The artifact is 5,529,935 bytes and contains producer tensors, runtime tensors,
and saved prepared contexts. It does not establish token-input extraction:
new prompts still require native context generation. The physical producer
contains 522,240 values and runtime 2,582; the logical producer count of 522,246
includes six residual scales held in the runtime. Do not add logical and physical
counts as though they described disjoint storage.

See [CPU receipt](SHARED_RESPONSE_EXPORT_CPU_2026-09-20.json) and
[portable artifact](../bilinear_quotient/circuits/followups/subject_response_v655_program.pt).

The next scientific issue is the omitted attention response, followed by genuinely
new validation and selective manipulation. Do not fit a global gain to conceal the
remaining position-dependent discrepancy. The full goal remains unfinished.

Primary evidence: [v654 receipt](../bilinear_quotient/circuits/followups/subject_joint_chain_v654_result.json).

## Context-generation correction

v659 adds six previously omitted output encoders (55,296 values), keeping the old response tensors bitwise unchanged. v655 remains a valid prepared-context consumer artifact but is not sufficient to generate new contexts with the stated native factors alone. See [composition and corrected accounting](COMPOSED_SUBJECT_RESPONSE_TRANSFER_2026-09-20.md).
