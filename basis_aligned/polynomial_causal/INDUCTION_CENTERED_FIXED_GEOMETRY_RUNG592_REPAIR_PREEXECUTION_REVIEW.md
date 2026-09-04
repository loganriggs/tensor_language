# Independent pre-execution review: repaired R592 implementation

**Reviewed:** 2026-09-04 UTC, before any R592 model execution or outcome

**Candidate commit:** `3f44c224ee0144a2a58da0487ffc863bfa75e7d7`

**Review mode:** immutable Git blobs and model-free fixtures only

**Verdict:** **BLOCKED BEFORE EXECUTION — semantic repairs pass, peak disk and preflight do not**

## Exact repaired candidate

| artifact | SHA-256 |
|---|---|
| producer | `9d75aaa291af61321cee29410b4ecfa772425e3dd2298e15440fb3a5843e799b` |
| model runtime | `09309b1299b85f2c57689913547fef01f2a9e7b538b2768ac62ff3e48e0f039c` |
| managed adapter | `64cda676fa0ba05c80af3986b5595659aa25937b26ed587034de929de97604dd` |
| owner test | `59764d300fdbe3f2024ee40b32b23fb2bcc56ccd79b48e7b1abbe5c0083eb2fc` |
| fake-runtime test | `52d3d22e7d1eeaaa31bed66a01d28aef296974bff94e96ab7707af6fa4219e85` |
| repair test | `691eb9786f344f1851447776ce0a2f5d324c60f9efbb0c780731c489e5e3c7dd` |
| dry run | `152c0cc38c671e7a1b96e199a76ebed607e058427b68be9cd9a53611d83c614e` |

The adapter pins these bytes plus the original preregistration, all four prospective amendments, all independent
reviews, and the prior implementation-block review/test. Its real branch embeds the verified producer bytes before
dispatch. Both producer and adapter pass the authoritative repository gate and advisory preflight.

## The seven prior blockers

All seven strict attacks from the first implementation review are closed in the exact candidate:

1. **Output topology:** every logit and full-vocabulary difference uses the checkpoint's unsliced 50,304 coordinates.
   FIT/SELECT difference arrays are `[3744,4,50304]` and `[1872,4,50304]`; corrected data-byte prices are
   3,013,410,816, 1,506,705,408, 4,520,116,224 combined, and 5,141,200,896 for the registered principal subtotal.
2. **Static gates:** both managed entry points pass `ops/gate.py`; registered `pred_a`, `pred_b`, and `pred_c` keys are
   present, and exception/local aliases no longer collide.
3. **Missing observations:** an absent mandatory array is rejected as an incomplete-call hard abort before any call
   evidence or public namespace is written.
4. **Invalid receipt depth:** the invalid receipt now hashes and records the byte length of every evidence file,
   including `nonfinite_mask_index.json` and every individual mask. A one-byte index mutation changes the anchored
   digest.
5. **Native-write gate:** the runtime independently recomputes the complete nine-head attention contraction from the
   observed state and saves both that vector and the untouched native attention write. It no longer copies the selected
   head vector into both fields. Endpoint and directed-native evidence both preserve `[N,4,1152]` raw arrays, and
   `instrument_gates.json` is derived from them.
6. **Memmaps:** both large arrays have exact final-offset checks, complete finite scans, explicit file fsync, shapes,
   byte lengths, and receipt hashes.
7. **Entry topology:** the 50,304-wide destination now exactly matches the pinned facade output, so the old first-call
   host shape failure is gone.

The independent fixtures also reconfirm exact physical width 30, byte-identical five-call batches, literal-zero replay,
the four frozen centered changes, component and hybrid cache/live checks, one same-state L8H3/H4 transaction, FIT-first
SELECT opening, FINAL/OOD closure, and exactly one facade forward per manifest call. The registered maximum remains 639
FIT plus 322 SELECT = 961 forwards, zero backwards, and zero updates.

## New blocking finding: exact implementation exceeds available disk before it can publish

The candidate retains every raw completed-call directory until it has constructed a second complete rectangular phase
tree. Only then does it delete `evidence/calls`. This creates a peak almost twice the final FIT evidence size.

Using only the declared raw array shapes and excluding NumPy headers, JSON, receipts, and filesystem metadata:

| FIT object concurrently present | exact data bytes | GiB |
|---|---:|---:|
| endpoint call directories | 602,989,056 | 0.562 |
| directed-native call directories | 1,306,446,336 | 1.217 |
| each directed intervention arm | 892,270,080 | 0.831 |
| all retained FIT call directories | 5,478,515,712 | 5.102 |
| complete FIT rectangular evidence | 5,198,883,840 | 4.842 |
| **FIT construction peak** | **10,677,399,552** | **9.944** |

SELECT has a 2,739,257,856-byte call tree and a 2,599,441,920-byte complete tree. With complete FIT evidence retained,
its corresponding raw-data peak is 10,537,583,616 bytes. FIT is therefore the larger peak.

After deleting my verified completed `/tmp/r592_review_81cwo1` snapshot, `statvfs` reported 9,592,934,400 free bytes.
That is 1,084,465,152 bytes below the FIT data-only minimum before headers and metadata. The separate stale R590 temp
directory was not mine and was not touched; it had already disappeared when free space was rechecked. The exact R592
adapter contains no `statvfs`/`disk_usage` capacity gate, so it can load the checkpoint and spend model calls before a
deterministic ENOSPC hard abort.

This is an execution blocker, not a scientific null and not a defect in the centered counterfactual.

## Minimal prospective storage amendment

The smallest robust repair is to keep the frozen call order and numerical operations but make the canonical complete
arrays the primary append-only evidence store.

1. Preallocate the FIT canonical `.npy` arrays on the same filesystem before the first model call. Record exact shapes,
   row ranges, and expected final offsets from the frozen manifest.
2. For each endpoint call, validate the complete response, copy its rows into the canonical endpoint arrays, fsync the
   written files at a bounded checkpoint, record per-array slice hashes in the call-prefix ledger, and remove that
   verified temporary call directory.
3. For each directed chunk, retain at most its five raw calls. Preserve immediate stopping after native, replay, score,
   payload, or joint. Once the chunk is valid and complete, write the native arrays, four actual hook-delta slices, four
   replay-relative logit-difference slices, and scientific primitive records in the exact current order and arithmetic;
   hash the canonical slices, then remove the raw chunk.
4. Prospectively supersede only the invalid-prefix physical layout: prior valid calls are represented by receipt-bound
   canonical-array slice descriptors, while the currently failing partial chunk retains its exact per-call raw arrays
   and nonfinite masks. The auditor must reconstruct the same ordered call prefix and every applicable predicate from
   the canonical slices plus final raw chunk. No scientific score may be computed for an invalid terminal.
5. After the final phase offset, run the same finite scans, fsyncs, evidence hashes, scoring, FIT-first gate, and
   receipt-last rename. The call manifest, token hashes, authority hashes, centered values, row order, float32
   differences, float64 aggregates, bootstrap draws, thresholds, and claim boundary remain byte-for-byte or
   value-for-value unchanged as applicable.

With this layout, complete FIT plus SELECT raw data is 7,798,325,760 bytes. The largest 32-row directed chunk adds only
41,671,168 bytes, for a data-only streaming peak of 7,839,996,928 bytes. A conservative frozen preflight should require
at least **9,000,000,000 free bytes** on the exact staging/publication filesystem both before model construction and
again before SELECT opens. It must hard-abort with zero model calls and no public namespace if the bound is not met.
The bound leaves over 1.16 GB for NumPy headers, JSON/JSONL, receipts, directories, and contemporaneous filesystem
variation; the implementation should recompute a still more conservative value if measured metadata upper bounds are
larger.

This amendment changes evidence staging and invalid-prefix addressing only. It must not change any scientific array,
statistic, predicate, failure precedence, terminal, model-call price, or claim.

## Tests and decision

Commands executed model-free:

```text
candidate owner/fake/repair/adapter/topology suites: 40 passed
independent exact-byte packet: 11 passed, 1 xfailed
producer gate: PASS
adapter gate: PASS
producer preflight: no findings
adapter preflight: no findings
adapter dry run: 0 forwards, 0 backwards, 0 updates; sealed splits closed
```

The one strict expected failure is the missing capacity check. No model, Torch, checkpoint, CUDA/GPU, queue, SELECT,
FINAL, OOD, bootstrap, or R592 outcome was opened. Candidate `3f44c224e` is **blocked before execution** until a
prospective storage amendment, implementation repair, and fresh different-agent exact-byte review establish adequate
peak capacity.
