# Independent review: task-17 FIT capability producer and blocked adapter

**Reviewed:** 2026-09-04 05:28 UTC.
**Target:** commit `e4f35b255b25ca4c345cd38e927945941313d583`.
**Verdict:** **VETO against model execution, authorization, and enqueue.**

The capability computation itself is well specified and the checked-in adapter is safely blocked. However, the
producer does not satisfy its frozen create-only publication claim: it can overwrite both a dangling symlink that was
already present in a final namespace and a final entry created concurrently after its precheck. This contradicts the
execution amendment's literal statement that existing final paths are never replaced. The current
`EXECUTION_AUTHORIZED=False` must remain in force.

## Exact reviewed unit

This review is anchored to Git objects, not to a moving working tree. The reviewed hashes are:

| Object | SHA-256 |
|---|---|
| producer | `a46b64410d0090d2034523be5b1eee58250c876131d78f97b3262c25ca637750` |
| blocked adapter | `037be5d06a34972ba8f75c8ce536c32f8d61ac62ca457845efb6561e7eb18019` |
| producer tests | `b13020c31f0b770bda2099e1869f83aa1a9b686d9322867eaaf557f243cf6391` |
| adapter tests | `d8cb8e380d6ba498e54ba26078e4bcf6a434a7a1e43068a4939c0fbd95b96702` |
| execution amendment | `f90b0b91ee5256ed6d5962300cf8a82666efc304edbc5d273d043b623388e7e4` |
| saved model-free dry run | `5542ba2b7315f6d36a5c898c02f6abf0fbbba2b1602a9da31654ee3b1089daf6` |

At the start of review these files matched the target commit. After the VETO was reported, the owner began a
prospective repair in the shared working tree. That later uncommitted work is deliberately excluded from this verdict.
Hash-gate failures observed while that repair was live are evidence that the adapter rejects changed producer bytes,
not failures of the target Git blob.

## What independently passed

### Model-free closure and authorization boundary

The dry run reproduces byte-for-byte with SHA-256
`5542ba2b7315f6d36a5c898c02f6abf0fbbba2b1602a9da31654ee3b1089daf6` under Python hash seeds 0, 1, and 999. It
compiles contract `526f292338abb5583942f95241be6aa2485db8421270e395bb9fa64bb34751c9`, executes the synthetic full call prefix
`0edd2541dcddb0d3442b05e6df3f65971a9d973281a676fc9117338435567bdf`, and reports exactly 8 calls, 192 row-side
evaluations, 24 evidence files, and 1,536 raw numeric bytes. The passing fixture returns `ok`; the planted failing
fixture returns `hard_abort` with every projection value null. It imports no Torch, opens no model or checkpoint, makes
zero forwards/backwards/updates, and excludes all runtime-only model and canary-source roles.

The real branch tests `EXECUTION_AUTHORIZED` before `capture`, so the reviewed blocked adapter raises before reading
even a frozen artifact, importing a model dependency, touching CUDA, or opening a checkpoint. Invalid dry-run values
also fail before capture.

I planted both import-cache modules and competing modules earlier on `sys.path`. Loading from captured, hash-verified
bytes replaced them. I additionally checked every internal dependency identity, not only the producer's three explicit
assertions: package, experiment framework, integration contract, task adapter, managed entry, capability compiler, and
producer all point to the captured modules loaded in dependency order. Changed bytes and symlinked frozen source paths
fail closed. Absolute paths, parent traversal, and repository escapes are rejected by the managed-entry boundary.

### Calls, token positions, native computation, and price

I independently reconstructed every physical position in all eight requests. All 192 token matrices equal the exact
registered base or donor sequence in the row order of the call. Each target ID is the independently tokenized
side-specific answer. Each foil list is the nonempty union of the base and donor payload-token IDs with that side's
target removed; the target is absent from every foil set.

A two-block CPU toy model confirmed that `native_logits` is bit-exact to the model's native recurrence: embedding,
initial RMS normalization, sequential blocks carrying `first_value` and `x0`, final RMS normalization, unembedding,
and

$$
z=30\tanh(\mathrm{logits}/30).
$$

The evaluator reads only the final sequence position, gathers the registered target, takes the maximum over that row's
registered foils, and returns two contiguous finite `float32[24]` arrays. Eight such pairs give

$$
8\times2\times24\times4=1{,}536
$$

raw numeric bytes. The checkpoint revision, configuration hash, weight hash, weight byte count, exact runtime
versions, single-CUDA-device float32 placement, and both canary gates are all checked before checkpoint-backed
evaluation. Fake-module tests separately confirmed that a runtime-version mismatch and unavailable CUDA both abort.

### Contract mutation, coverage, stopping, and output surface

Joint mutations of call and metric row order, target/foil mutations, re-signed summaries, price changes, duplicated or
missing primitive rows, relabeled side/transform keys, extra primitive fields, malformed arrays, and nonfinite arrays
all fail closed. The capability decision revalidates the full compiled-contract, call-manifest, and metric-manifest
digests before interpreting primitive values. Exactly one primitive is required for every registered
`(call, row, side, transform)` key.

The framework checks evaluator and projector purity, detects primitive mutation, and recomputes the projection under
several row permutations. A failed capability predicate returns before calling the projector. Recursive planted keys
such as `attention_head`, `reader`, `writer`, `component`, `activation`, `localization`, and `selection` are refused;
a `hard_abort` with any non-null projection value is also refused. The source has no later-phase generator, SELECT,
TEST, OOD, localization, gradient, backward, optimizer, or model-update path.

Injected ordinary exceptions after each staged artifact and after each of the three publication moves roll back without
leaving a final namespace in the expected single-writer case. Receipt-last publication also prevents an incomplete
multi-file state from validating as complete. Those properties do not repair the create-only defect below.

## Blocking defect: publication is not create-only

Three checks use `Path.exists()`:

- `producer.require_unused_namespaces`;
- `circuit_artifact_package.stage_package`; and
- `circuit_artifact_package.publish_staged_package`.

`Path.exists()` is false for a dangling symlink. I planted a dangling symlink at the final result path. Producer
preflight accepted it, staging accepted it, and the subsequent `os.replace(source, destination)` silently replaced the
pre-existing symlink. The final package then validated successfully. Thus an occupied namespace can be destroyed while
the result falsely claims create-only publication.

There is a second, independent time-of-check/time-of-use failure. I staged a valid package, allowed the publication
precheck to observe three absent destinations, and injected an external result file immediately before the result
rename. `os.replace` silently overwrote that concurrently created file and publication completed. The same primitive
also permits rollback to move or replace an external inode if a destination is substituted after this invocation's
move. A precheck followed by `os.replace` cannot establish no-replace semantics.

These are exact CPU reproductions against the `e4f35b255` blobs:

```text
dangling_symlink_overwritten=YES
concurrent_destination_overwritten=YES
```

This violates the immutable amendment's requirements that all final entries be absent before the model boundary and
that existing final paths are never replaced. It is authorization-blocking even though no such entry currently exists.

## Queue compatibility and a second boundary to close prospectively

The exact absolute adapter path parses, passes `gate.py`, passes `ops/test_fast.py`, and its model-free preflight exits
successfully when the reviewed producer bytes are present. It is therefore syntactically compatible with
`ops/enqueue.sh`. I did not call the helper or edit either queue.

The current helper stores only an absolute path; `bqrunner.sh` later executes whatever bytes occupy that path. It does
not preserve or recheck the adapter digest established at enqueue time. The adapter hash printed by the dry run is also
informational, not compared with an external authority. This is not exploitable while the real branch remains blocked,
but an authorized revision must close the enqueue-to-execution path/hash gap or enforce an operational repository
write freeze for the whole interval. A hash-pinned launcher or queue record that verifies and executes the reviewed
adapter bytes is the stronger solution.

## Test accounting

Before the prospective repair changed the shared working tree, the focused producer/adapter and relevant compiler,
task, integration, package-framework, and result-contract tests passed. In a Git archive of the exact target, 84 tests
passed; five adapter tests then correctly rejected the now-different producer at the adapter's intentionally hard-coded
live repository root. Re-running the in-process adapter suite with only that root redirected to the exact archive passed
9/9; the excluded subprocess case had already reproduced successfully against the untouched live target. Independent
tests added exhaustive 192-position reconstruction, native-forward equivalence, complete internal module identity,
runtime/CUDA gates, recursive output attacks, jointly mutated manifests, coverage attacks, six injected crash points,
and both publication exploits above.

## Required repair and exact remaining authorization procedure

No `e4f35b255` real execution is authorized. A repair must:

1. use entry-aware `lstat`/`lexists` checks for every final and staging destination, including dangling symlinks;
2. install evidence, result, and receipt with a genuinely atomic no-replace primitive on the same filesystem, failing
   closed if that primitive is unavailable;
3. publish the receipt last and roll back only an entry whose inode/identity is proven to be the one installed by this
   invocation—never overwrite, remove, or capture a concurrent external entry;
4. test pre-existing regular files, directories, valid symlinks, and dangling symlinks at all three destinations;
5. inject a competing entry immediately before each install and substitute an external inode before each rollback;
6. retain the exact 8-call/192-evaluation/1,536-byte computation, all-null valid failure, phase closure, and zero
   backward/update guarantees; and
7. freeze the repaired producer, adapter, tests, dry-run artifact, and a new versioned amendment under new hashes. The
   existing amendment must remain immutable.

Then a different-agent review must approve those exact Git blobs and publish a review digest. A later authorization
amendment must bind that digest and the exact authorized adapter hash. The final adapter must remain blocked until that
amendment exists and must itself receive a last independent review after authorization is wired. Only after exact hash
and current-path equality are rechecked may the absolute authorized adapter be passed once through `ops/enqueue.sh`.
The enqueue-to-run interval must preserve the reviewed bytes, preferably through an executable hash-pinned launcher or
runner record. The managed queue receipt must then be checked. Until all of those steps pass, the only approved action
is the model-free dry run; no model, GPU, checkpoint, result namespace, or later phase may be opened.
