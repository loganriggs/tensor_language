# Independent pre-execution review: R592 streaming candidate

**Reviewed:** 2026-09-04 UTC, before any R592 model execution or outcome

**Candidate commit:** `521e4c38ca55b9ede6f51cb5408aa1fdbb4486d2`

**Review mode:** immutable Git blobs and model-free NumPy fixtures only

**Verdict:** **BLOCKED BEFORE EXECUTION — streaming semantics pass, but the two capacity gates make SELECT unreachable on the current disk**

## Exact candidate

| artifact | SHA-256 |
|---|---|
| producer | `741d7a1481e79a726d3a2edb8bb5274a5d262ce0a93803d438c5762911809efb` |
| model runtime | `09309b1299b85f2c57689913547fef01f2a9e7b538b2768ac62ff3e48e0f039c` |
| managed adapter | `420ab088c8b361f7645ceb77f158a3e187277c2a9c849f830fb16fcdf85a6654` |
| owner test | `59764d300fdbe3f2024ee40b32b23fb2bcc56ccd79b48e7b1abbe5c0083eb2fc` |
| fake-runtime test | `52d3d22e7d1eeaaa31bed66a01d28aef296974bff94e96ab7707af6fa4219e85` |
| repaired-implementation test | `dceb2416d20e7e795f8d3d0dd59bac18c123e3ed7705d3660fe6187abfc73844` |
| streaming-storage test | `2f36d595bfe7efa8f8825e9829912e1f6c70ff0d4c0d1f69fe054aebc48a7fda` |
| managed-adapter test | `c6c4e6bb8e9b23a63b1352064f670429fba8227c92260bb638004edadeb22478` |
| dry run | `937d5d9682ea89ca7e4feda3e646937dee83d56f31b18b8dffd4f04b26b4a1eb` |
| streaming amendment | `2df290b9670adfb8541d675e51fc607f856f7f70c083248fdba14ab8cf90df07` |

The dry run recomputes exactly from the committed producer when the immutable blob is given its registered logical
path. Both authoritative static gates and both advisory preflights pass on the committed producer and adapter.

## What the repair gets right

The repaired implementation preserves the earlier scientific computation and closes the original storage-layout
blocker:

1. Canonical FIT and SELECT arrays have the frozen schemas and exact authority order. Endpoint calls append 1,728/864
   rows; native directed calls append 3,744/1,872 rows. Every directed chunk is the same registered native, replay,
   score, payload, joint sequence on identical tokens. The model-call census remains 639 FIT plus conditional 322
   SELECT = 961 maximum; there are zero backwards and zero weight updates.
2. Each slice is copied, flushed, file-fsynced, read back, and hashed over its filename, dtype, shape, axis bounds, and
   C-order payload before its ledger record is fsynced. Endpoint raw bytes are deleted only after that record. All five
   raw directories in a directed chunk survive until all canonical slices and all five ordered ledger records are
   durable.
3. An invalid partial chunk retains the exact current raw calls and nonfinite masks. Earlier complete chunks are
   represented by canonical slice descriptors. `call_prefix.jsonl`, the canonical ledger, all complete files, all
   current raw files, the mask index, and every mask are covered by the invalid receipt's byte lengths and hashes.
   An incomplete call remains a hard abort with no publication.
4. Finalization requires exact endpoint and directed offsets and the full call-ledger count, refuses surviving raw call
   directories, scans every floating canonical file for finiteness, recomputes the native-write identities, fsyncs all
   files, and records whole-file byte lengths and SHA-256 hashes. Unwritten tails cannot become normal evidence.
5. The runtime is unchanged from the previously reviewed semantic repair. Full 50,304-logit output, physical width 30,
   literal-zero replay, centered score/payload/joint additions, the same-layer L8 transaction, full nine-head native
   reconstruction, FIT-first opening, bootstrap identities, thresholds, and FINAL/OOD closure remain intact.
6. The adapter pins the producer, runtime, tests, dry run, authority lineage, previous block review, and streaming
   amendment. It embeds verified producer bytes for dispatch; the producer independently verifies and immutably loads
   its executable dependency closure. Normal and invalid publication remain evidence first, result second, receipt
   last.

The independently reconstructed data sizes are:

| concurrently relevant data | bytes |
|---|---:|
| complete FIT canonical arrays | 5,198,883,840 |
| complete SELECT canonical arrays | 2,599,441,920 |
| largest current 32-row five-call chunk | 41,671,168 |
| registered streaming data peak | 7,839,996,928 |

These exact data-only values are correct. NumPy headers, JSON/JSONL, directory metadata, and receipts remain outside the
count, as stated in the amendment.

## Blocking capacity contradiction

The adapter and producer accept execution when the filesystem has exactly 9,000,000,000 bytes available before model
construction. If FIT passes its scientific gates, however, the producer then retains 5,198,883,840 bytes of complete
FIT canonical arrays and requires **another** 9,000,000,000 bytes to remain available before opening SELECT. With no
other disk changes, an initially accepted 9 GB filesystem has only

$$
9{,}000{,}000{,}000-5{,}198{,}883{,}840=3{,}801{,}116{,}160
$$

bytes available at the second gate, so the held/SELECT path must hard-abort. Keeping the second gate literally at 9 GB
therefore requires at least

$$
5{,}198{,}883{,}840+9{,}000{,}000{,}000=14{,}198{,}883{,}840
$$

initially available, before headers and metadata. During this review, `statvfs` was about 9.38 GB before the run. Thus
the first preflight could authorize 639 model calls, while a successful FIT phase would deterministically be discarded
before SELECT. This is safe failure behavior but not an executable prospective test of the registered held terminal.

This contradiction is in the frozen storage amendment as well as the implementation: the amendment simultaneously
describes 9 GB as the conservative capacity threshold for the 7.84 GB total peak and demands that the same 9 GB remain
free after FIT. The implementation follows the latter sentence exactly, so changing code alone would violate the
current amendment.

## Required repair

Either:

1. provide and verify at least 14,198,883,840 available bytes plus a documented metadata margin before dispatch while
   leaving both frozen 9 GB gates unchanged; or
2. freeze a narrow prospective capacity amendment defining a phase-relative pre-SELECT threshold. Preserving the same
   1,160,003,072-byte margin over the exact total streaming peak gives
   `2,599,441,920 + 41,671,168 + 1,160,003,072 = 3,801,116,160` available bytes before SELECT. The adapter's initial
   threshold remains 9 GB.

No scientific formula, row, call, threshold, or evidence byte layout needs to change. Candidate `521e4c38c` must not be
enqueued on the current disk under its advertised 9 GB preflight.

## Tests and scope

The new independent suite binds immutable Git blobs, re-derives schemas and prices, checks authority row order and
five-call chunks, inspects durability/deletion order, checks invalid-prefix and whole-file binding, recomputes the
frozen dry run, and plants the two-gate capacity failure. No model, Torch, checkpoint, CUDA/GPU, queue, R592 outcome,
FINAL, or OOD artifact was opened.
