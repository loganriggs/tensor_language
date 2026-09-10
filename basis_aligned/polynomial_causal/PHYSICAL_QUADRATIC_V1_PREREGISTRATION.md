# Physical last-layer replacement screen, 10 September 21:33 UTC

Frozen input functions transfer across the two corpus panels, but reconstruction
before finalRMS/tanh does not establish model prediction preservation. Replace
the actual MLP17 module with three frozen compiled implementations: shared64/
product128 with old writers, the same readers with Pile-training-refitted writers,
and freeproduct128 with Pile-training-refitted writers. Retain the native bias.
Shared implementations actually reuse64input projections; no expansion into256
independent dense input readers. All earlier modules and full unembedding stay.

Use all224 Pile validation documents, all32 registered sampled positions/document.
Targets are the following tokens at those positions. This panel has been used
for local reconstruction comparisons, so this is an exploratory validation
screen, not untouched test/OOD certification. Test split stays unopened. No new
parameter fitting or scale tuning. Existing MLP17 low-rank causal failures remain
valid and motivate this check rather than an adoption claim.

For each document record native CE, candidate true-token CE minus native CE
(positive is damage), mean absolute token CE change, native-to-candidate full-
vocabulary KL, and agreement with the native top1 token. Mean and paired document
bootstrap95% intervals use existing helper,4000draws seed91162133. No individual
token or document filtering after seeing effects. KL includes all50304 logits.

Registered predictions:

- A instrument: exactly226bodyforwards904sequences (four arms x224 documents,
  batch4, plus two4-sequence controls); no-op and native-factor replacement CE
  max-absolute differences<=2e-5 nats and relative-logit-L2<=1e-5 on first4
  validation documents; finite metrics; compiled CPU FP64/FP32 controls held.
- B shared-refit preserves prediction: mean absolute token CE change<=0.05nats
  and native top1 agreement>=0.95. Both required, unchanged if failed.
- C writer recalibration helps behavior: mean full-vocabulary KL for shared-refit
  <=90% of shared-original KL. A failure can coexist with a reconstruction gain.

No-op wrapper and direct native-factor module controls physically execute through
the same replacement path. The latter uses all native4608products and is not a
compressed candidate. Every candidate physically replaces the module; original
MLP need not execute for its forward pass. Retained native weights for the rest
of the model, source fit/capture costs and the screen's in-memory baseline copy
remain charged. Candidate numbers include native1152bias. No claim of independent
text-to-circuit input producers, selective removal or composition from this run.

Managed lane1 only,900s alarm. Output: per-document JSON metrics and frozen module
artifact/control hashes. Any A failure invalidates B/C interpretation. If B
fails, preserve the failure, compare signed CE/KL/position behavior and locate
which residual/norm/readout component causes it before rejecting all structure.
