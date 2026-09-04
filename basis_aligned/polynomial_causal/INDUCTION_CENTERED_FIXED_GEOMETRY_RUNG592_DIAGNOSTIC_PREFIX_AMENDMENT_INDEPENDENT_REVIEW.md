# Independent review: R592 diagnostic-prefix amendment

Date: 2026-09-04 UTC

Reviewed commit: `3be7c21c3886502ea989efdaeba5c137aef45d8e`

Reviewed amendment SHA-256:
`f153fa3df6d7d00e951d2e7d2f0a270e6383f9133d0d34049a9eee57640b2c62`

Verdict: **BLOCKED before implementation**

The per-call prefix design fixes the previous mid-chunk ambiguity. A completed
directed call now has its own directory at its real batch size, so failures
after native, replay, score, payload, or joint all have one literal manifest
prefix without padding a four-arm rectangular tensor. One narrow byte-level
ambiguity remains: simultaneous nonfinite values in two or more raw arrays
cannot be represented by the specified mask filename.

This was an exact-byte, model-free, outcome-blind review. I read the amendment,
its parent specification, and the blocking review, but did not inspect an R592
implementation or outcome, load a model, use CUDA/GPU or a queue, or alter any
reviewed authority.

## Exact authorities

The reviewed commit contains the amendment bytes above and correctly pins:

| authority | SHA-256 |
|---|---|
| parent R592 preregistration amendment | `5e9fe2bcf41b88c199b5dfab2ba3ec7d0fa8f4b4b2952173c1984391e4d53094` |
| blocking independent review | `21bdc310b4798d3ae6d47fc2ed7dfee969afd871bc90db381db634e2c4cae2f5` |

The working-tree amendment is byte-identical to the blob at the reviewed
commit. The scope explicitly leaves all rows, splits, sites, roles, arms,
interventions, thresholds, bootstrap identities, complete-result evidence,
and the 639/322/961 forward price unchanged.

## What now passes

### Exact completed-call prefixes

The failure-producing completed call is the final call in the published
prefix, and no later call executes. The directory name binds its zero-based
manifest index and call ID. `call_prefix.jsonl`, the receipt, and the directory
census all have to agree with an exact frozen manifest prefix. This gives
unique unpadded encodings for failures after each of the five directed calls:

```text
native
native, replay
native, replay, score
native, replay, score, payload
native, replay, score, payload, joint
```

Missing or extra directories and a call ID taken from beyond the prefix are
rejectable directly. Each call uses its actual batch size. In particular, the
last SELECT directed chunk is represented as 16 rows throughout; it cannot be
silently extended to 32.

### Failure choice and unsafe calls

The eight predicates have a fixed total precedence. Predicates not yet
evaluable are skipped, and the first newly failing predicate is recorded. A
forward that raises, is killed, or cannot materialize its required observations
is not a completed call and is an unpublishable hard abort. Temporary bytes are
not public evidence, and receipt-last publication ensures that a crash before
the receipt creates no recognized diagnostic.

### Diagnostic and scientific namespaces

The invalid artifact has distinct evidence, diagnostic, and receipt names. It
cannot contain `split_scores`, bootstrap intervals, a scientific terminal, or a
held/null claim, and normal R592 paths must remain absent. Conversely, a valid
complete phase publishes none of the per-call diagnostic namespace. A partial
diagnostic therefore cannot be promoted to a normal result.

### One-array nonfinite semantics

For one affected raw float array, the rule is auditable: retain exact NaN/Inf
bytes, retain a same-shape boolean mask, hash both, compare the mask to
`~isfinite(raw)`, and compute no scientific aggregate. The planted correct mask
passes; wrong content or dtype fails.

## Sole blocker: mask paths collide for multiple affected arrays

The amendment says the final failing-call directory stores a packed boolean
file named exactly `nonfinite_mask.npy` **for each affected raw array**. All raw
arrays otherwise live in that one flat call directory. If two arrays are
affected, both masks therefore require the same path.

This is not a remote corner case. A nonfinite cached or planned value in a
directed arm can appear in `planned_hook_deltas.npy`, propagate to
`hook_deltas.npy`, and then propagate to `logits.npy` in the same completed
call. Those arrays have different shapes, so one shared mask cannot represent
them. The prose supplies neither array-specific filenames nor a raw-array to
mask map. It also calls the file “packed” while requiring a same-shape boolean
`.npy`; bit-packed and ordinary NumPy boolean arrays have different byte and
shape contracts.

Consequently two incompatible producers satisfy part of the prose:

1. retain only one `nonfinite_mask.npy`, losing the other affected arrays; or
2. invent mask paths or a container schema not frozen by the amendment.

The auditor cannot decide which bytes are authoritative. This violates the
amendment's exact evidence and independent reconstruction requirement.

## Required narrow correction

Prospectively freeze an injective mask representation. One sufficient rule is:

```text
nonfinite_masks/<raw-array-filename>.npy
```

with each mask an ordinary C-contiguous NumPy `bool` array having exactly the
same logical shape as its raw array. The call record should bind an explicit
sorted map from each affected raw filename to its mask filename, shape, byte
length, and SHA-256. Require exactly one mask for every and only every raw float
array containing a nonfinite value, and require every mask to equal
`~isfinite(raw)` elementwise. If actual bit packing is intended instead, freeze
the packing axis, bit order, padding-bit value, logical shape, and byte formula.

No row, call, threshold, factor formula, scientific gate, bootstrap, or claim
needs to change.

## Adversarial packet

`test_induction_centered_fixed_geometry_rung592_diagnostic_prefix_amendment_review.py`
reports `15 passed, 1 strict xfailed`.

The passing fixtures bind the exact commit blob and authority hashes; exercise
failures after all five directed calls; reject missing, extra, and non-prefix
directories; enforce the unpadded batch-16 tail; accept a correct one-array
nonfinite mask and reject wrong masks; enforce predicate precedence; model
hard-abort/no-publication and receipt-last recognition; and reject mixing the
diagnostic with a scientific result. The strict expected failure plants three
simultaneously affected arrays and demonstrates the mask-path collision.

## Decision

**BLOCK exact commit `3be7c21c3886502ea989efdaeba5c137aef45d8e`.**
The per-call prefix, call shapes, predicate order, hard-abort boundary, atomic
publication, and scientific separation should be retained. Only the
array-specific nonfinite-mask serialization needs a prospective exact-byte
repair and another independent review.
