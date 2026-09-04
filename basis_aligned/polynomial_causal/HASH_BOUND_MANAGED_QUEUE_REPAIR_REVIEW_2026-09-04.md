# Independent review: hash-bound lane-1 queue preflight repair

**Reviewed:** 2026-09-04 05:39 UTC.
**Target:** commit `7d675812f05ae1cf351b48924603c8e5c2ef0e35`.
**Verdict:** **VETO; snapshot bytes remain mutable before dry-run execution.**

This successor fixes the first `3e71f2b4b` VETO: it verifies the original candidate against `EXPECTED_SHA256` before
executing candidate code. It nevertheless reintroduces the same class of flaw through a writable snapshot path. The
helper passes that snapshot to `gate.py` and later reopens it with `python3 "$check_path"`, without rehashing or safely
capturing the snapshot for same-process execution. A gate or concurrent same-user process can replace the snapshot and
cause unreviewed bytes to execute during preflight. The new infrastructure must not be activated or used for task-17.

## Exact reviewed objects

| Object | SHA-256 |
|---|---|
| repaired `ops/enqueue.sh` | `34d5bf07948a88387154f09b9a003f38678592cb1598c6deec78f65be17f6455` |
| unchanged `ops/bqrunner.sh` | `a8b9aae2be074dea1a9f261a329a663ee16e2b45d7c5c8e262d2b5ea3cb40a1e` |
| repaired tests | `918c18fcf20a7fad755f7ca46721ac745bc69471b649805bd24ec7cc888dc6a9` |
| `LESSONS.md` at target | `73ea68d99f669f95abea3ebe749e33f4743471d093522db86db4535da142abb5` |
| `AGENT_BOARD.md` at target | `66f4612f9619d97907f0513cc561abc724de2544a2ddebe500a47573fc28cee0` |

The eight owner tests pass and both shell scripts pass `bash -n`. I reviewed Git blobs because the shared working tree
may move during the next repair. I did not touch a real queue, restart a runner, or access model/GPU/checkpoint/outcome
state.

## What the repair fixed

Lane 1 now safely opens the original with `O_NOFOLLOW`, requires a regular file, compares descriptor identity before
and after reading, computes its digest, and checks `EXPECTED_SHA256` before parsing or executing candidate bytes. It
writes those captured bytes to a same-directory `mkstemp` snapshot. Syntax, gate, and dry-run checks point at the
snapshot. After preflight it safely rehashes the original path and refuses if it differs from the first capture. A
late edit of the original therefore cannot execute during dry run and cannot enter the queue.

The new owner regression tests correctly show that a mismatched original cannot create a marker before refusal and
that changing the original during gate leaves dry-run on the reviewed snapshot, followed by a final hash refusal.
Lane-2 still writes its legacy bare-path record. Runner-side exact hash-record parsing, safe target capture, digest
verification before compile, captured-byte execution, no reopen, legacy compatibility, imports, exit status, and
logging remain as described in the first review.

## Remaining blocking exploit

The snapshot is a regular mode-0600 file, but all collaborating agents run as the same user. Its random name is not an
immutability boundary. The exact order is:

1. create a snapshot containing the reviewed payload;
2. pass its pathname to `gate.py`;
3. reopen that pathname with `python3 "$check_path"`;
4. never compare the reopened snapshot with the reviewed digest.

I used an exact temporary copy of the successor. The candidate original remained benign and retained its reviewed
hash. The fixture gate overwrote only its `sys.argv[1]` snapshot with a valid script that writes a marker, then returned
success. The result was:

```text
returncode 0
snapshot_mutation_executed True
queue_record '<benign-reviewed-sha256>\t<benign-original-path>\n'
```

Thus unreviewed snapshot bytes executed and enqueue reported success. The final original-path rehash cannot detect
this because the original was never changed. A concurrent process can perform the same replacement between gate and
dry run.

There is also a semantic mismatch: `python3 "$check_path"` exposes the random snapshot as `__file__` and
`sys.argv[0]`, while the runner later exposes the original path. A script that derives its namespace or imports from
its filename can therefore pass a different preflight computation from the eventual managed run.

## Required successor

The next version must retain the early original capture and expected-digest check, but the target dry run must be
performed by a trusted loader that:

1. safely captures the snapshot (or retains the original captured payload in the same trusted process);
2. requires its digest to equal the already reviewed original digest;
3. only then compiles and executes those captured bytes without reopening any candidate pathname; and
4. sets `__file__`, `sys.argv[0]`, and `sys.path[0]` to the original target semantics.

It must add the exploit above as a regression: mutation of the snapshot by gate or a concurrent process must either
leave dry run on the original captured payload or fail before candidate execution, with no marker and no queue record.
It must preserve final original-path revalidation, exact hash-tab-path append, runner-side safe execution, legacy
compatibility, and lane-2 behavior. New hashes and another independent review are required before runner activation or
task-17 enqueue.
