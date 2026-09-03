# R591 managed-adapter handoff

Date: 2026-09-03 UTC

Status: **READY FOR DIFFERENT-AGENT REVIEW.** This addendum supersedes the
queue-boundary portion of builder commit `34f8c76b0`; it does not change the
R591 diagnostic computation and is not an approval or enqueue request.

## Exact candidate

| Artifact | SHA-256 |
|---|---|
| diagnostic producer | `b2b266529f0f842211fea46856064133df5e3f4a8a7758c9095e7d29a94b6c49` |
| producer owner test | `e756ba3d17d3ebee2f81e97e573dd216090555de1fd3f1cfc926268f902d9ce7` |
| deterministic dry run | `161193de5d90da69aafcd681e375993fa91d32e99100f0ed02fb586d5a629d8b` |
| amended preregistration | `e72cb386d65c68f55b767c8141c3c4d774b3c8ad9387ac7f8ad43bebef118593` |
| builder handoff | `61f8fb407dc026a7a2b126f2dce02b60266d040ffcce7159c5dc6a0d2517cc4f` |
| managed adapter | `5fe0a0d3bb4c149881a1d6d76f5adf7e661df35af39cc37e1cd9893b93cc33cd` |
| managed-adapter test | `b20ea468089c90629191f71c6e5f97d4caec180fce64bf0d1ce17f3f9565d7b6` |

The adapter also directly pins the exact R585 producer/test/dryrun, facade,
induction helper, manifest, dependency lock, and phase-specific-support v5
method bytes already pinned by the producer. It therefore fails before dispatch
if either the candidate or a transitive method dependency changes.

## Managed behavior

With `BQLIB_DRYRUN=1`, the adapter imports the hash-checked producer, executes
only its CPU authority/schedule/hash validation, and requires exact equality
with the committed dry-run JSON. With `BQLIB_DRYRUN` absent, it can call
`os.execv` only as

```text
<current Python executable> <exact R591 producer path>
```

The adapter accepts no arguments and rejects every other value of
`BQLIB_DRYRUN`. Before either branch it refuses any occupied conventional R585
or R591 result, receipt, or evidence namespace. It has no model import or model
call of its own.

Verification after adding the adapter is 28 passed: 15 producer tests, 10
adapter tests, and 3 shared v5 method tests. The producer and adapter both pass
the static gate and preflight; both managed dry runs are model-free. No model,
CUDA, GPU, queue, or outcome was accessed.
