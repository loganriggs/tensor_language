# Frozen QK source features on FineWeb — 11 September 2026

Purpose: validate a frozen weight-derived source intervention and its conditional
prediction on natural model states. No fitting, frame adjustment, task-specific
selection or million-token expansion. This is not semantic circuit identification.

Data: first32rows of .rowcache/fineweb_n192_skip7000.pt, first128input tokens,
next-token target at128. The rowcache/census generators use HuggingFaceFW/fineweb.
These are historically opened cached rows, not fresh/OOD or verified document
holdouts. File hash and selected row hashes are frozen before GPU execution.
The generic bilin18_eval_tokens_large.pt is Pile and is excluded.

Features: nine fixed rank17 source frames from POSITION_SHARED_QK_SOURCE_V1_RESULT.
Random control: seed1321, rank17 Gaussian orthonormal frame inside each head's
joint key-weight span. Fixed before observing data outputs; no control selection.

Intervention: attention17, source position32, all nine heads. Each head reads
s-E_hE_h^T s only for its two key maps and current value map. Its queries and the
base-value stream are unchanged. Head key RMS is recomputed; the upstream source
RMS is not recomputed because this is an internal read-edge intervention.
All other source positions and all other layers' reads are untouched.

Endpoint: attention17 output at query127 and the next-token logits/CE there.
For each batch, predict the full residual change from UNEDITED ports before any
physical edited forward. Independently perform the physical key/value edit.
Then inject only the predicted residual change at attention17 query127 and
recompute the native remaining MLP/unembedding. There is no subsequent attention,
so changes at other query positions cannot affect this endpoint.

Arms: baseline, zero-hook control, physical learned, predicted learned, physical
random, predicted random. Batch8,4batches:24model-body forwards,192sequence
evaluations,4096distinct input token positions,24576processed positions. Full
native model retained; no adoption saving. Save baseline ports and endpoint
logits in shared memory for audits, not for fitting.

Predictions:

- A: exact row/count/scope checks; zero-hook endpoint logits within1e-6 of baseline;
  predicted versus physical attention-change relative L2<=1e-3 for every batch
  and both frames; all finite. Hook checks must confirm only source32 key/value
  reads change. Existing vector algebra control must pass.
- B: predicted versus physical logit-change relative L2<=.01 for every batch and
  both frames.
- C: per-row CE prediction mean absolute error<=1e-3 for both frames.

Before native execution, the runner schema required three prediction keys. The
original joint B check was split into B (logits) and C (CE), with the same bars.

Report signed CE changes, KL from baseline, and projected source-state energy
for learned/random frames descriptively. No predicted semantic advantage or
selectivity threshold; one arbitrary source position and32rows cannot identify
a linguistic circuit. A CE increase is damage, not success.

If numerical prediction misses, preserve it and audit rounding/cancellation,
hook semantics and endpoint isolation using saved baseline/edited receipts.
Do not adapt source features to repair it. This is in-distribution validation
only; Pile stays separately labelled OOD for a later frozen validation.
