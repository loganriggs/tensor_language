# Rung 592 prospective streaming-storage and capacity amendment

**Frozen:** 2026-09-04T02:09:50Z, before any repaired R592 implementation, model call, or outcome

**Status:** prospective narrow specification correction; implementation and execution remain blocked pending this
amendment's committed bytes and a later different-agent exact-byte review of the repaired implementation

## Authority and scope

The independent exact-byte review at commit
`7c7aa4b401c4ae6556dab9da00d30efec37314ea` blocks R592 candidate
`3f44c224ee0144a2a58da0487ffc863bfa75e7d7` only because its retained call tree and complete FIT tree require
10,677,399,552 data bytes at the same time, exceeding measured free space. The review and test hashes are:

- review: `e88ea815b154d922df44143d549c735068d6947e729d668b4849cfbd23e4f444`;
- model-free review tests: `ec1759555f8abf80cde08a93fe01c9e97fe32b6effc467085c75d06a551c6899`.

The original preregistration and its executable, diagnostic-prefix, nonfinite-mask, and 50,304-logit topology
amendments remain authoritative. This amendment supersedes only physical evidence staging, invalid-prefix addressing,
and the free-space preflight. It changes no scientific quantity, counterfactual, model call, numerical operation,
threshold, split decision, or claim.

## Canonical append-only phase arrays

For each opened phase, the implementation must create the final-shape canonical NumPy arrays on the same filesystem as
the eventual public evidence directory before that phase's first model call. Their shapes, dtypes, row order, and
values are exactly those already frozen in R592. An array is append-only in semantic row order: only the next registered
axis-0 slice may be written, and no verified slice may later be changed.

Each completed endpoint call is processed as follows:

1. retain one temporary raw call directory;
2. validate every mandatory response, predicate, token hash, dtype, shape, and finite requirement;
3. copy the call's rows into the next endpoint slice of every applicable canonical array;
4. flush and explicitly fsync every changed canonical file;
5. compute a SHA-256 hash over each written canonical slice's dtype, shape, axis-0 bounds, and contiguous C-order payload
   bytes, append those descriptors to the ordered call-prefix ledger, and fsync the ledger; and
6. delete the raw call directory only after all preceding checks and durable writes succeed.

Directed calls retain at most one current registered chunk: native, replay, score, payload, and joint, in their frozen
order. A failure stops immediately, so a later arm is never called. Only after all five calls in a chunk are valid may
the implementation compute and append that chunk's native arrays, actual hook changes, four replay-relative full-logit
differences, and scientific primitive records. It must flush and fsync every changed canonical file, hash every
canonical slice, append the five ordered prefix records, fsync the ledger, and only then delete the five raw call
directories. No raw bytes from a verified earlier call or chunk remain concurrently with a later chunk.

The endpoint and directed row offsets must exactly reach the frozen phase counts. Finalization must retain the existing
complete finite scans, reconstruction gates, whole-file byte lengths and SHA-256 hashes, scoring, bootstrap draws,
FIT-first rule, and receipt-last publication.

## Invalid-prefix representation

This amendment prospectively supersedes the old rule that every prior valid call remains as a raw call directory. On an
invalid completed call:

- all earlier verified endpoint calls and complete directed chunks are represented by the canonical arrays plus the
  ordered prefix ledger's content-bound slice descriptors;
- the current endpoint call, or every completed call in the current partial directed chunk including the failing call,
  remains in exact raw form;
- nonfinite mask-index and mask bytes remain one-to-one with the failing raw arrays and are content-bound by the invalid
  receipt;
- each prefix record states whether its evidence is `canonical_slices` or `raw_current_chunk`, and the ordered union must
  equal the literal frozen call-manifest prefix with no gap, duplicate, or reorder; and
- the invalid receipt binds every retained evidence file, every slice descriptor, the written row bounds, and the
  ordered call-prefix ledger. Unwritten canonical tails are not observations and may not enter a diagnostic predicate.

An auditor must be able to recompute every predicate applicable to the executed prefix from the canonical slices and
current raw chunk. No scientific score or scientific terminal is computed for an invalid prefix. An incomplete call
still hard-aborts without any public namespace.

## Exact streaming price and capacity gates

The complete FIT plus SELECT canonical array data occupy exactly 7,798,325,760 bytes. The largest retained 32-row
directed chunk occupies 41,671,168 data bytes. Therefore the exact registered maximum streaming data peak is

`7,798,325,760 + 41,671,168 = 7,839,996,928` bytes.

This price excludes NumPy headers, JSON/JSONL, receipts, directories, and filesystem metadata, as did the prior raw-array
subtotals. The implementation must report all three exact constants in the model-free dry run.

A conservative capacity threshold of **9,000,000,000 available bytes** is frozen. Available bytes are exactly
`os.statvfs(path).f_bavail * os.statvfs(path).f_frsize`, measured on the actual staging/publication filesystem.

1. The managed adapter and producer each check the threshold before model construction. Failure hard-aborts with zero
   model calls and no normal or invalid public namespace.
2. The producer checks the same threshold again after FIT has been finalized and immediately before SELECT is opened.
   Failure hard-aborts before any SELECT call and publishes no R592 namespace.
3. The checks are lower bounds, not reservations. A later I/O error remains a hard abort; it cannot become a scientific
   null or diagnostic terminal.

## Preserved protocol

R592 still uses all 50,304 logits, physical width 30, the four machine arms `replay`, `score`, `payload`, and `joint`,
the exact centered additions, the same active controls and bootstrap identities, 639 FIT calls, conditionally 322 SELECT
calls, 961 maximum forwards, zero backwards, and zero weight updates. Masks, hashes, immediate predicate precedence,
same-filesystem staging, atomic evidence/result/receipt ordering, FINAL/OOD closure, and the partial-output-factor claim
are unchanged.

The repaired implementation must pass model-free tests for insufficient space at both boundaries, mid-stream invalid
publication, incomplete-call recovery, deletion only after copied/fsynced/slice-hashed evidence, and numerical equality
to the legacy full-tree materialization on synthetic evidence. Execution remains prohibited until a fresh independent
exact-byte review approves the repaired candidate.
