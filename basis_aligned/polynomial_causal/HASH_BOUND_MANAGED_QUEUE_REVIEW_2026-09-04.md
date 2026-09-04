# Independent review: hash-bound lane-1 managed queue

**Reviewed:** 2026-09-04 05:36 UTC.
**Target:** commit `3e71f2b4b91d194cd0fda34967c6d530b9fd23eb`.
**Verdict:** **VETO for reviewed-job enqueue or runner activation.**

The runner-side captured-byte execution is sound, but `enqueue.sh` checks `EXPECTED_SHA256` only after it executes the
target's dry-run branch. A substituted, unreviewed script can therefore execute arbitrary top-level code during
enqueue and only then be refused for hash mismatch. The new queue protocol must not be relied on for task-17 or
activated by restarting the runner until this ordering flaw is repaired and independently reviewed.

## Exact reviewed objects

| Object | SHA-256 |
|---|---|
| `ops/enqueue.sh` | `9dc5df6067aac0ea9601eefa02c909262623fe34ba174aee088125d3101e7d37` |
| `ops/bqrunner.sh` | `a8b9aae2be074dea1a9f261a329a663ee16e2b45d7c5c8e262d2b5ea3cb40a1e` |
| queue tests | `79fe3f5bcd79dea46c4948e294443ede99dc741ac92b670c361217eadd9c18c7` |
| `LESSONS.md` at target | `0351d50fa4d2c7b7f8270d236df2b34195b70e0a239b0a377a748fc24b5b5cd2` |
| `AGENT_BOARD.md` at target | `0f096e8e8f0a06f8745d87d2e67b2a86910e0c9d52a5d6ad3b3733433ea4688c` |

Later commits do not alter the three code/test paths under review. I did not edit either real queue, restart either
runner, invoke the enqueue helper on the repository, or access a GPU/model/checkpoint/outcome.

## Runner-side behavior that passed

For a new record of the exact form

```text
<64 lowercase hex SHA-256><TAB><absolute path>
```

the lane-1 runner parses the digest and path separately, rejects malformed digest or relative-path records, safely
opens the target with `O_NOFOLLOW`, requires a regular file, reads from one descriptor, compares device/inode/size/
mtime/ctime before and after capture, and verifies the captured SHA-256. Only after all those checks does its trusted
inline Python program call `compile` and `exec` on the in-memory payload. It does not reopen the target.

I confirmed matching-script execution, changed-byte refusal, symlink refusal, directory refusal, lowercase/length
digest validation, and exact exit status propagation. A target that overwrites its own path during execution still
runs the already captured original bytes. A script importing a sibling module sees the target directory at
`sys.path[0]`, `__name__ == "__main__"`, the expected `__file__`, and `sys.argv == [path]`; an explicit
`SystemExit(7)` returns status 7. The surrounding `run_one` retains the existing per-script log and completion-ledger
status behavior.

Legacy bare absolute-path records still take the old direct `python "$path"` branch. This preserves compatibility but
does not add hash protection to already queued legacy work. New lane-1 records written by this commit are hash-tab-path
records. The lane-2 branch still writes a bare path to `queue2.txt`, requires the existing `# BQLANE: cpu` header, and
does not touch lane 1. I reproduced lane-1 deduplication against a hash-bound record and lane-2's unchanged bare-path
format in isolated temporary fixtures.

The six owner tests pass, and both shell files pass `bash -n`.

## Blocking enqueue-before-hash execution

The relevant lane-1 order in `enqueue.sh` is:

1. parse the target path with `ast.parse`;
2. run `test_fast.py`;
3. run `gate.py` on the target path;
4. execute `BQLIB_DRYRUN=1 BQLIB_NO_MODEL=1 python3 "$f"`;
5. only then safely capture the file, compute its digest, and compare `EXPECTED_SHA256`.

The environment variables are a convention, not a sandbox. An unreviewed replacement can ignore them and perform any
action available to the enqueueing user before the mismatch is detected.

I reproduced this against an exact temporary copy of the reviewed helper. I set `EXPECTED_SHA256` to a benign reviewed
payload, replaced the target with a different valid Python script that writes a marker, and invoked the helper on the
temporary project. The result was:

```text
returncode 1
mismatch_refused True
unreviewed_code_executed_before_refusal True
queue_empty True
```

Thus the helper correctly refused to append the changed digest, but it had already executed the unreviewed bytes. This
is precisely the enqueue-to-execution trust gap that the change claims to close, moved earlier into preflight rather
than eliminated.

There are related check/reopen windows between the initial symlink check, AST parse, gate, and dry-run path open. The
runner's later captured-byte behavior cannot undo a side effect that already happened during enqueue.

## Required repair

A successor must:

1. safely capture the target and, when `EXPECTED_SHA256` is supplied, verify that reviewed digest before any target
   byte is interpreted or executed;
2. run the dry run from those captured bytes, rather than reopening the path with `python3 "$f"`;
3. ensure the syntax and gate decisions apply to the same reviewed payload, or recapture and reverify the reviewed
   digest after non-executing path-based checks before queue append;
4. append exactly the verified digest and absolute path, preserving lane-1 deduplication and the legacy/lane-2
   compatibility boundary;
5. retain runner-side safe capture, digest verification before compile, captured-byte execution, exit/log behavior,
   and no target reopen; and
6. add a regression test in which a mismatched parseable target attempts a visible dry-run side effect and prove the
   side effect never occurs.

The repaired scripts and tests need new hashes and a fresh independent review. Until then, the task-17 authorization
adapter may be reviewed as source, but it must not be enqueued and this runner change must not be activated as its
trusted execution boundary.
