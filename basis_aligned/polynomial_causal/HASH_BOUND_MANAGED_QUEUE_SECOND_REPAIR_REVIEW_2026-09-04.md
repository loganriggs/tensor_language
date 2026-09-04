# Independent review: second hash-bound lane-1 queue repair

**Reviewed:** 2026-09-04 05:42 UTC.
**Target:** commit `afa628e118c4ca8a48c719328293dc2c25bb6399`.
**Verdict:** **APPROVE the hash-bound lane-1 queue protocol.**

This successor closes both prior infrastructure VETOs. A reviewed lane-1 target is now hash-checked before any
candidate execution, dry-run bytes are safely recaptured and reverified after the gate, and the trusted enqueue and
runner loaders both compile only their captured payloads without reopening the candidate path. This approval is for
the infrastructure bytes below; it is not itself a task-17 enqueue receipt or scientific authorization.

## Exact approved objects

| Object | SHA-256 |
|---|---|
| repaired `ops/enqueue.sh` | `35baab247d4d358dfaaa76e5862e5ce8fc53a17b181212b66af3212cb8c9649d` |
| unchanged `ops/bqrunner.sh` | `a8b9aae2be074dea1a9f261a329a663ee16e2b45d7c5c8e262d2b5ea3cb40a1e` |
| queue tests | `84bdcb61783cee8638d70cfe675ba31db4ea3665becef2ec6794a1e2b6546b6a` |
| `LESSONS.md` at target | `3530d06af3ff4932ff43da5156ff3ecc6f19e6d6f73777c910f8a02004f5342e` |
| `AGENT_BOARD.md` at target | `12f3da7a30852e17e2c340186dac4d8723e35fe8822b885447f4833ad9af3401` |

Later shared-work commits leave the four code/test/lesson paths byte-identical. The runner is unchanged from
`3e71f2b4b`; this review rechecks it as part of the end-to-end protocol.

## Enqueue-side chain

For lane 1, the helper:

1. rejects a non-file, symlink, non-absolute path, tab, or newline;
2. safely opens the original with `O_NOFOLLOW`, requires a regular file, compares device/inode/size/mtime/ctime before
   and after reading, and hashes those captured bytes;
3. compares that digest with `EXPECTED_SHA256`, when provided, before parsing or executing any candidate byte;
4. writes only that captured payload to a mode-0600 same-directory snapshot;
5. applies syntax and gate checks to the snapshot;
6. safely opens the snapshot with `O_NOFOLLOW`, repeats the regular-file and identity checks, requires its digest to
   equal the first reviewed digest, and then compiles and executes only the recaptured in-memory bytes for dry run;
7. restores the original target's `__file__`, `sys.argv[0]`, and `sys.path[0]` semantics for that captured execution;
8. safely rehashes the original path after all checks and refuses if it differs from the first capture; and
9. appends exactly `<reviewed sha256><TAB><absolute path>`.

The snapshot is removed by the exit trap. A late change after final revalidation cannot run different bytes: the
queued digest remains the first reviewed digest and the runner independently enforces it.

## Runner-side chain

The lane-1 runner parses a hash-bound record only when the prefix is exactly 64 lowercase hexadecimal characters and
the suffix is an absolute path. Its trusted inline Python loader safely captures that target through one no-follow
descriptor, verifies identity and the queued SHA-256 before `compile`, and executes the captured payload in the same
process. It sets ordinary script import and `__main__` state and never reopens the target.

Changed bytes, an in-place edit observed during capture, symlink substitution, non-regular targets, and malformed
records therefore fail before candidate execution. A target changed after capture still runs only the verified
payload. Exit status flows into the existing per-job log and completion ledger.

Legacy bare absolute-path queue records remain on the old direct-execution branch for already queued work. Lane 2
retains its prior bare-path `queue2.txt` format and `# BQLANE: cpu` requirement. The new protocol does not claim to
retrofit hash guarantees to either category.

## Adversarial evidence

The 10 owner tests pass in 1.01 seconds, and both shell scripts pass `bash -n`. They include matching execution,
changed target, symlink, no-reopen, expected-hash mismatch before side effect, original substitution during gate,
lane-2 compatibility, exact logical script semantics, and the exact writable-snapshot exploit from the previous VETO.

I independently rechecked:

- valid and malformed hash parsing, directory refusal, sibling imports, `__main__`, `__file__`, argv, import path, and
  explicit nonzero exit status;
- hash-bound record deduplication and isolated lane-2 bare-path output;
- original replacement before initial capture, during gate, during final revalidation, and after runner capture;
- snapshot replacement by a different regular file and by a symlink after gate; and
- that neither a changed original nor changed snapshot can write the planted marker before refusal.

The original `3e71f2b4b` exploit now fails before dry run because `EXPECTED_SHA256` is checked on the first capture. The
`7d675812f` exploit now fails inside the enqueue loader because the recaptured snapshot digest differs from the first
capture; its substituted bytes are never compiled. Preserving logical original-path semantics also removes the prior
snapshot-name mismatch between preflight and managed execution.

## Scope

This approves the exact infrastructure target for hash-bound lane-1 records. A reviewed experiment must supply its
independently approved digest as `EXPECTED_SHA256`; omitting it does not establish an external review binding even
though the queued bytes remain self-consistent. The task-17 adapter still requires its own final exact-source approval,
current digest recheck, an empty/safe queue check, and one explicit managed enqueue. I did not enqueue anything, edit a
real queue, restart a service, or access a model, checkpoint, GPU, or outcome while performing this review.
