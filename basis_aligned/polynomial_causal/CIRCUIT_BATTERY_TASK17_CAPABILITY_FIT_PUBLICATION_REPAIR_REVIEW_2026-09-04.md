# Independent review: task-17 FIT capability publication repair

**Reviewed:** 2026-09-04 05:21 UTC.
**Target:** commit `af7393a38f724a6ce7ce10119f8b9852744c099b`.
**Verdict:** **APPROVE the repaired producer and blocked adapter as a prospective CPU-reviewed unit.**

This approval closes the create-only publication VETO on commit `e4f35b255`. It does not authorize a model call,
checkpoint read, GPU use, queue edit, enqueue, task-17 result, or any localization/later phase. The reviewed adapter
still has `EXECUTION_AUTHORIZED=False` and correctly aborts its real branch before artifact capture.

## Exact reviewed objects

I reviewed exact Git blobs and confirmed that later shared-work commits do not alter any reviewed path:

| Object | SHA-256 |
|---|---|
| repaired producer | `3dcf04c0f776c056f3701967a666025ed8b63cab4d7e60a868fd766b00ac98ea` |
| blocked adapter | `15d60e1760581228b69d214ffcebebf5231a15cd5a09d018bda4bd98bae69ca5` |
| producer tests | `d4eadf7a6615f5456f327d413d525f68a4628d07a09b1f52c2133055d87eaf00` |
| adapter tests | `06d274efe8f83ce207129b15c7c407d6d50c08ff61686d14ae85828ca63970d8` |
| saved dry run | `35dfa1edaa5af6c2352a30bbda7a89b30a612f6e292ed0585d8b3a5457b894e4` |
| original execution amendment | `f90b0b91ee5256ed6d5962300cf8a82666efc304edbc5d273d043b623388e7e4` |
| publication-repair amendment | `0c4a20b751cc05c5373b3a1d0eab95164ffc70e5dbe685cc12a9dbb341ff8301` |
| provenance correction | `14a982abbc79de99e970dea2d352952e22e70717e7e9f677ace23370f3e7685b` |
| unchanged generic package module | `6c8f81f16e3465b33c27abacd1114bd8ae7ce2fffa358c2a665f906a49f011cc` |

The scientific compiler, authority, manifests, thresholds, checkpoint, runtime, call count, evidence price, and
capability decision are unchanged from the independently approved compiler and the otherwise-sound portions of the
producer reviewed in `e4f35b255`.

## Provenance correction

The first repair commit is exactly `538cef96451b3e8f07758f20cca2be1b7bfdf561`. Its Git author and committer times
are both `2026-09-04T05:13:56+00:00`. Its embedded repair amendment incorrectly said `05:22 UTC`, a future-time
transcription error. The original amendment remains byte-identical at SHA-256 `0c4a20b...`; it was not silently
rewritten.

Commit `af7393a38f724a6ce7ce10119f8b9852744c099b`, authored and committed at
`2026-09-04T05:19:58+00:00`, adds the append-only correction `14a982ab...`. That correction identifies the exact repair
commit and its true timestamp, limits the correction to provenance, records that execution remained blocked, and is
itself captured by the adapter. Direct `git show` reconstruction confirms the original amendment bytes and both commit
timestamps. This is an adequate auditable correction because no task-17 model outcome, checkpoint access, enqueue, or
final namespace existed between the true freeze and the correction.

## Create-only publication repair

Final-namespace preflight now calls `os.lstat`. Every directory entry therefore counts as occupied: regular file,
directory, FIFO, device, live symlink, and dangling symlink. `run_science` performs this check before runtime probing,
canary reads, checkpoint loading, or any model call. I planted an occupied dangling receipt and instrumented the
runtime gate; the namespace error occurred first and the runtime hook was never called.

Correctness does not depend on that precheck. Each install uses the Linux primitive

$$
\operatorname{renameat2}(\mathrm{AT\_FDCWD},s,
                         \mathrm{AT\_FDCWD},d,
                         \mathrm{RENAME\_NOREPLACE}).
$$

Destination absence and rename are therefore one atomic operation. A pre-existing or newly raced entry produces
`EEXIST`/`ENOTEMPTY`; `ENOSYS`, `EINVAL`, and cross-filesystem `EXDEV` fail closed. There is no `os.replace` fallback.
The task-local publisher moves the complete evidence directory, then the mutually bound result, then the receipt last.
After every install it verifies the saved device/inode/mode/size identity and fsyncs the parent directory.

On an exception, rollback considers only entries recorded as successfully installed by this invocation. It first
requires the same saved inode identity and an absent staging source, then applies the same atomic no-replace primitive
in reverse. If an external actor substitutes the destination or creates the rollback source, rollback refuses that
entry, preserves the external bytes, reports incomplete safe rollback, and does not label the state complete or
retryable. Ordinary injected exceptions after evidence, result, or receipt restore all owned entries to a complete
stage, and an immediate retry publishes a package that passes full receipt/result/evidence validation.

## Independent adversarial evidence

In addition to the 11 owner tests, I ran a 24-case publication matrix:

- all three final roles occupied at preflight by each of file, directory, live symlink, dangling symlink, and FIFO
  (`3 x 5 = 15` refusals);
- a regular file or directory created immediately before each of evidence, result, and receipt install
  (`3 x 2 = 6` no-overwrite refusals); and
- forced `ENOSYS`, `EINVAL`, and `EXDEV` from the no-replace primitive (three fail-closed cases).

I separately replaced the installed inode with an external dangling symlink after each of the three move positions.
Every case left the external entry untouched and rolled back only earlier still-owned entries. I also created an
external staging source immediately before rollback; the publisher preserved both it and the invocation's final inode
and raised an explicit incomplete-rollback error. Finally, monkeypatching `os.replace` to raise did not affect a
successful task-local publication, directly proving there is no weaker fallback path.

The original two VETO reproductions now fail safely: a dangling final symlink is rejected before staging, and a
destination created after precheck is preserved by `RENAME_NOREPLACE`. File and nonempty-evidence-directory packages
both exercise the real kernel primitive on this host.

## Unchanged scientific and execution boundaries

The final model-free stdout is deterministic under Python hash seeds 0, 1, and 999 and exactly matches the saved file,
SHA-256 `35dfa1ed...`. It still reports compiled contract `526f2923...`, call manifest `0edd2541...`, exactly 8 calls,
192 row-side evaluations, 24 evidence files, 1,536 raw numeric bytes, zero model forwards/backwards/updates, an `ok`
passing fixture, and an all-null `hard_abort` failing fixture. It captures the repair amendment and provenance
correction but excludes runtime-only model/facade/canary sources.

The prior independent review's exhaustive 192-position target/foil reconstruction, native-forward equivalence
including `30*tanh(logits/30)`, runtime/checkpoint/canary checks, verified-module preload defense, manifest/coverage
attacks, projector purity, recursive forbidden-output checks, all-null failure, and phase closure continue to apply.
The repair diff changes only publication and its declarations/tests. I re-ran the focused repaired producer/adapter
suite (`37/37`) and the selected broader result-contract/framework/integration/task/compiler/producer/adapter suite
(`102/102`). `gate.py`, `test_fast.py`, dry-run execution, and `git diff --check` pass. No task-17 result, receipt, or
evidence namespace exists.

## Scope and remaining authorization procedure

This review approves `af7393a38` only as the repaired, still-blocked implementation. The next permitted steps are:

1. freeze a new prospective authorization amendment that cites this review's final SHA-256, the exact producer and
   blocked-adapter hashes above, and preserves the original scientific contract;
2. build an authorization-enabled adapter successor that binds that amendment and retains the full verified closure,
   pre-model namespace check, runtime/checkpoint/canary gates, exact call/evidence price, no-replace publisher, and
   real-branch failure behavior;
3. obtain a final independent review of that exact authorized adapter and every changed hash; and
4. only after current-file equality and the exact authorized adapter digest are rechecked, pass its absolute path once
   through `ops/enqueue.sh` and verify the managed queue receipt.

The queue currently stores a path rather than immutable bytes. Therefore the reviewed adapter bytes must be preserved
unchanged from final review through runner execution; a hash-pinned launcher/queue record is preferable, otherwise an
explicit repository write freeze and a runner-time digest check are required. This approval by itself is not an
enqueue receipt and must not be interpreted as model-execution authority.
