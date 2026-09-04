# R591 managed-adapter handoff

Date: 2026-09-03 UTC

Status: **REPAIRED CANDIDATE READY FOR DIFFERENT-AGENT REVIEW.** This addendum
supersedes the queue-boundary portion of builder commit `34f8c76b0` and the
blocked exact packet `1396747c0`. It does not change the R591 rows, schedules,
measurements, tolerance, or price and is not an approval or enqueue request.

## Exact candidate

| Artifact | SHA-256 |
|---|---|
| diagnostic producer | `fb8239ded4f3e99510f37ea72337c2d69e4640f7a2556748c9062aa82b2751bc` |
| producer owner test | `8a24a9903d10ada8a4048c7adcb33cb4ef3e8aeef11d6f9718f8e50e57b6212c` |
| deterministic dry run | `8a6331fb1a4d3800abff5ab6b7e291105872b06b41a43b003436312b6e50dc5d` |
| amended preregistration | `2dd8f918f767a6e5d91af357cfaa14770b79334ebac837d1bf52e8046ce190a5` |
| builder handoff | `202f1268e583a82f6cca385f4223b6edf4e8f8bbaee2c1cc975b09e51cd95f12` |
| managed adapter | `b0a0654c4b6fd28a9dfbfb947969049c203ef346cc580f87f5406701ac876d20` |
| managed-adapter test | `338dd545838e75ae8de4a8bd6405f4bac601fe2ad8a81f594bab8104151de0ed` |

The adapter also directly pins the exact R585 producer/test/dryrun, facade,
induction helper, manifest, dependency lock, and phase-specific-support v5
method bytes already pinned by the producer. It therefore fails before dispatch
if either the candidate or a transitive method dependency changes.

The repaired producer also pins shared handoff v6, the direct R578 rows, R585
amendment, canonical/factor helpers, and `jacclust/tt_model.py`. It snapshots
executable bytes before importing them. Its model-free authority path replaces
R585's outcome-parsing verifier with directly checked semantic inputs and an
immutable manifest/row snapshot, so it does not read R586/R587 artifacts.

## Managed behavior

With `BQLIB_DRYRUN=1`, the adapter executes an in-memory snapshot of the
hash-checked producer, runs only its CPU authority/schedule/hash validation, and requires exact equality
with the committed dry-run JSON. With `BQLIB_DRYRUN` absent, it can call
`os.execv` only with an isolated Python `-c` launcher containing the exact
base64-encoded producer bytes that were hashed immediately before command
construction. The logical `__file__` and `sys.argv[0]` remain the canonical
producer path, but Python never reopens that mutable pathname.

The adapter accepts no arguments and rejects every other value of
`BQLIB_DRYRUN`. Before either branch it refuses any occupied conventional R585
or R591 result, receipt, or evidence namespace. It has no model import or model
call of its own.

Verification after repair is 19 producer tests and 11 adapter tests, with the
shared v5/v6 method tests run separately. The producer and adapter both pass
the static gate and preflight; both managed dry runs are model-free. No model,
CUDA, GPU, queue, or outcome was accessed.
