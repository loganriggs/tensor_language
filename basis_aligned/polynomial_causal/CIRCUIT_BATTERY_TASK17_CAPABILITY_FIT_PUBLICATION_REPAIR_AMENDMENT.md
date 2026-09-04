# Task 17 FIT capability publication-repair amendment

**Frozen prospectively:** 2026-09-04 05:22 UTC. **Status:** CPU implementation repair only; model execution remains
blocked. This is a versioned successor to the execution amendment with SHA-256
`f90b0b91ee5256ed6d5962300cf8a82666efc304edbc5d273d043b623388e7e4`. It changes no authority row, call, metric,
prediction, threshold, scientific decision, phase order, checkpoint, runtime requirement, or literal price.

## Veto being repaired

Independent pre-execution review found that the first publication implementation did not prove its stated
create-only guarantee:

1. `Path.exists()` returns false for a dangling symlink, so a preexisting filesystem entry could be misclassified as
   an unused result, evidence, or receipt namespace.
2. The generic package publisher checks destinations and later calls `os.replace()`. A filesystem entry created in
   that interval can be overwritten by `os.replace()`.

This is an instrument-level veto. The original producer and adapter remain unexecuted, and there are no task-17 model
outcomes to retract. The compiler contract remains approved and unchanged.

## Repaired preflight

All three final paths are checked with `os.lstat`, not `Path.exists`. Therefore a regular file, directory, FIFO,
device, live symlink, or dangling symlink all count as occupied. This check runs before runtime validation, checkpoint
loading, or model calls. It is an early diagnostic only: correctness does not rely on the check remaining true.

## Atomic no-overwrite publication

The repaired producer uses the Linux operation

$$
\operatorname{renameat2}(\mathrm{AT\_FDCWD}, s,
                         \mathrm{AT\_FDCWD}, d,
                         \mathrm{RENAME\_NOREPLACE})
$$

for each final install. `RENAME_NOREPLACE` makes destination nonexistence and the rename one atomic filesystem
operation. It works for both regular files and the evidence directory and returns `EEXIST`/`ENOTEMPTY` instead of
replacing any existing entry, including a dangling symlink. If the primitive is unavailable, unsupported, or crosses
filesystems, execution fails closed; there is no fallback to a check-then-replace operation.

Publication order remains:

1. the complete evidence directory;
2. the mutually bound result file; and
3. the receipt file last.

Thus presence of the receipt continues to identify a complete package. After every successful rename, the producer
checks that the destination has the same device/inode/mode/size identity as the staged source and fsyncs its parent
directory.

## Race-safe rollback and retry

On a failure after one or more installs, rollback considers only destinations successfully installed by this
invocation. Before moving one back to the stage, it verifies the exact saved inode identity and uses
`RENAME_NOREPLACE` in reverse. It never moves a destination it did not install, never overwrites an existing staged
source, and never moves an externally substituted inode into the trusted stage.

For an ordinary process failure, all installed entries return to the recognized complete stage and the same stage can
be retried. If another actor removes an installed inode and substitutes a different entry before rollback, rollback
fails closed and preserves the external entry. That exceptional state is deliberately not called retryable or
complete; it requires audit rather than guessing ownership.

## Frozen repair bytes and tests

- Repaired producer SHA-256:
  `3dcf04c0f776c056f3701967a666025ed8b63cab4d7e60a868fd766b00ac98ea`.
- The generic artifact package is deliberately unchanged at SHA-256:
  `6c8f81f16e3465b33c27abacd1114bd8ae7ce2fffa358c2a665f906a49f011cc`.
- Frozen compiler-contract SHA-256 remains:
  `526f292338abb5583942f95241be6aa2485db8421270e395bb9fa64bb34751c9`.
- Frozen call-manifest SHA-256 remains:
  `0edd2541dcddb0d3442b05e6df3f65971a9d973281a676fc9117338435567bdf`.
- Raw numeric evidence remains exactly 1,536 bytes from 8 calls and 192 row-side evaluations.

Targeted tests place dangling symlinks at each final destination, create each destination after preflight and
immediately before its atomic install, crash after each of evidence/result/receipt, retry every normally rolled-back
stage, and replace an installed inode before rollback. Every attack must leave external bytes untouched. The suite also
calls the real Linux primitive on a file, a directory, and a dangling-symlink collision.

## Authorization remains closed

The repaired adapter must freeze this amendment and the repaired producer hash, regenerate its model-free dry-run
receipt, and keep `EXECUTION_AUTHORIZED=False`. A fresh different-agent review of the repaired producer and adapter,
followed by a later authorization amendment that binds that review, remains necessary before managed enqueue. This
repair grants no model, checkpoint, GPU, queue, outcome, localization, SELECT, TEST, or OOD authority.
