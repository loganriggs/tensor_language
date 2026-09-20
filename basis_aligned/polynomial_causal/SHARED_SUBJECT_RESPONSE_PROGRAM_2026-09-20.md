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

An independent CPU test verifies serialization without a model object and equality
to full symmetric-core execution at five edit amplitudes. v655 is registered to
export the unchanged native-derived program, producer tensors and prepared contexts,
and to check packed execution against the existing joint-tensor executor at1e-8.
[check_shared_response_export.py](check_shared_response_export.py) is ready to
replay the exported contexts in a separate CPU-only consumer process. The export
job remains queued; artifact creation and native/CPU replay are not yet claimed.

The next scientific issue is the omitted attention response, followed by genuinely
new validation and selective manipulation. Do not fit a global gain to conceal the
remaining position-dependent discrepancy. The full goal remains unfinished.

Primary evidence: [v654 receipt](../bilinear_quotient/circuits/followups/subject_joint_chain_v654_result.json).
