# Ordinary reproducibility review: hash-bound lane-1 enqueue successor

**Reviewed:** 2026-09-04 05:46 UTC

**Exact target:** commit `afa628e118c4ca8a48c719328293dc2c25bb6399`

**Verdict:** **APPROVE**

This is a routine pre-run version/reproducibility review, not an adversarial exploit search. I inspected only the
checked-in enqueue, runner, lesson, and test bytes and ran their isolated CPU tests. I did not inspect or change the
live queue, run an enqueue helper, access a model/checkpoint/outcome/GPU, or control a service.

## Exact reviewed bytes

| Object at `afa628e11` | SHA-256 |
|---|---|
| `ops/enqueue.sh` | `35baab247d4d358dfaaa76e5862e5ce8fc53a17b181212b66af3212cb8c9649d` |
| `ops/bqrunner.sh` | `a8b9aae2be074dea1a9f261a329a663ee16e2b45d7c5c8e262d2b5ea3cb40a1e` |
| `ops/bqrunner2.sh` | `8248003bf891ffac2dc71b9384a7ca51a56a4cab6babdf5b7d40dd376a174d73` |
| `ops/test_hash_bound_managed_queue.py` | `84bdcb61783cee8638d70cfe675ba31db4ea3665becef2ec6794a1e2b6546b6a` |
| `LESSONS.md` | `3530d06af3ff4932ff43da5156ff3ecc6f19e6d6f73777c910f8a02004f5342e` |

The current worktree versions of all four code/test paths are byte-identical to the target commit. Both runner hashes
are also identical on the target commit and its parent; `afa628e11` changes the lane-1 enqueue preflight, not either
runner.

## Version-binding trace

For lane 1, `enqueue.sh` safely captures the original target bytes, computes `sha`, and checks that exact value against
the independently supplied `EXPECTED_SHA256` before executing candidate code. Syntax and the repository gate operate
on the private snapshot. Immediately before the model-free dry run, the new inline loader safely recaptures that
snapshot, recomputes `observed = sha256(payload)`, requires `observed == sha`, and compiles only `payload` in memory.
It restores the original target as `__file__`, `sys.argv[0]`, and `sys.path[0]`.

After preflight, enqueue safely rehashes the original target and again requires `current_sha == sha`. The sole lane-1
append is then literally `printf '%s\t%s\n' "$sha" "$f"`. Thus the digest proven around the bytes executed by
model-free preflight is the same shell variable stored in the queue record. The runner independently parses that
digest, safely captures the target, verifies the digest, and compiles only the captured payload. There is no second
unverified source read in either execution path.

## Compatibility checks

Legacy lane-1 path-only records remain recognized: a record without a tab leaves `expected_sha` empty and follows the
pre-existing `python "$path"` branch inside `run_one`. This preserves already queued records but intentionally gives
them no new external hash guarantee.

Lane 2 is unchanged. Its runner is byte-identical across the target commit, and enqueue still requires the literal
`# BQLANE: cpu` header, performs the original path-based dry run, and appends only the absolute path to `queue2.txt`.
The checked-in lane-2 compatibility test exercises that exact format.

## Reproduction

`pytest -q basis_aligned/bilinear_quotient/ops/test_hash_bound_managed_queue.py` passes **10/10** in **0.98 s**. The
suite covers exact matching execution, changed-byte and symlink refusal, captured-byte execution without path reopen,
reviewed-hash enforcement before candidate side effects, late original-path change refusal, snapshot re-verification,
original script context, and lane-2 path-only compatibility. `bash -n` separately passes for `enqueue.sh`,
`bqrunner.sh`, and `bqrunner2.sh`.

## Decision and scope

**APPROVE** exact commit `afa628e11` as the lane-1 infrastructure dependency for a separately approved hash-bound
experiment. This does not authorize any particular experiment or enqueue. A caller must still provide the exact
independently reviewed experiment digest via `EXPECTED_SHA256`; omitting it retains internal queue consistency but does
not prove correspondence to an external review.
