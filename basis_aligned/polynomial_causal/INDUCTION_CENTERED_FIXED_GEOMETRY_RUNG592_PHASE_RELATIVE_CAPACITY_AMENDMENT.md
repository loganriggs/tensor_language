# Rung 592 prospective phase-relative capacity amendment

**Frozen:** 2026-09-04T02:40:46Z, before any corrected implementation, model call, or outcome

**Status:** prospective narrow correction to the pre-SELECT free-space threshold; implementation and execution remain
blocked pending a later fresh different-agent exact-byte review

## Authority and contradiction repaired

The independent streaming-candidate review at commit
`a3492a2edec8c5d5d49d6b5cfe48e8bbdfb477bf` binds candidate `521e4c38ca55b9ede6f51cb5408aa1fdbb4486d2`
and finds its scientific and streaming state machines sound, but blocks its repeated 9,000,000,000-byte capacity gate.
The review and model-free test hashes are:

- review: `8a22980fb766b8b51cac81acb69ad8e84cd886dae053613591acabc415c6f225`;
- test: `7c84f858625b92af4b7242b168cf7d321d8dcc7ae82a5988bfcb9372d099514b`.

The original streaming amendment correctly requires 9,000,000,000 available bytes before model construction. It also
requires the same absolute amount again after the complete FIT canonical arrays already occupy 5,198,883,840 bytes.
That second requirement makes the advertised 9 GB initial preflight unable to reach SELECT. This amendment supersedes
only that post-FIT threshold.

## Exact boundary-specific thresholds

The initial producer and managed-adapter threshold remains exactly **9,000,000,000 available bytes** on the actual
staging/publication filesystem before model construction.

The exact maximum streaming data peak is 7,839,996,928 bytes, so the initial threshold contains the safety margin

`9,000,000,000 - 7,839,996,928 = 1,160,003,072` bytes.

Immediately before SELECT, FIT's 5,198,883,840 canonical data bytes already exist. The remaining registered SELECT
storage requirement is:

- complete SELECT canonical arrays: 2,599,441,920 bytes;
- largest simultaneously retained five-call chunk: 41,671,168 bytes;
- subtotal: `2,599,441,920 + 41,671,168 = 2,641,113,088` bytes.

Preserving the identical 1,160,003,072-byte safety margin therefore fixes the pre-SELECT threshold at exactly

`2,641,113,088 + 1,160,003,072 = 3,801,116,160` available bytes.

At the limiting initial capacity, the two gates compose exactly:

`9,000,000,000 - 5,198,883,840 = 3,801,116,160` bytes.

Thus an otherwise unchanged filesystem that passes the initial gate at 9 GB also passes the SELECT gate after writing
exactly the registered FIT data. A filesystem one byte below either boundary must fail that boundary.

Available bytes remain defined as `os.statvfs(path).f_bavail * os.statvfs(path).f_frsize`. Both comparisons are inclusive:
the boundary passes when available bytes equal the threshold and fails only when available bytes are smaller.

## Failure and evidence semantics

An initial-capacity failure occurs before model construction and therefore has zero model calls. A pre-SELECT capacity
failure occurs after FIT but before the first SELECT call. Both are dependency/preflight hard aborts: neither may create
or rename a normal or invalid R592 public namespace, and neither may be scored as a scientific null. Temporary staged
FIT bytes are removed on the pre-SELECT failure through the already-frozen hard-abort cleanup path.

The dry run and managed preflight must report both named thresholds and the exact derivation terms so an auditor does not
mistake the boundary-relative SELECT requirement for a second full-run reservation.

## Everything else remains frozen

No evidence shape, dtype, row order, slice hash, mask, receipt, call, counterfactual, threshold, bootstrap draw, scientific
terminal, or claim changes. R592 retains all 50,304 logits, width 30, 639 FIT calls, conditionally 322 SELECT calls, 961
maximum forwards, zero backwards, zero weight updates, receipt-last publication, and closed FINAL/OOD splits.

The repaired candidate must add model-free boundary tests at equality and one byte below, demonstrate that an initial
9.5-billion-byte filesystem remains above the SELECT threshold after exact FIT data, and prove capacity failures cannot
publish a namespace or cross the relevant model-call boundary. It remains prohibited from model, Torch, checkpoint,
CUDA/GPU, queue, and outcome access until fresh independent approval.
