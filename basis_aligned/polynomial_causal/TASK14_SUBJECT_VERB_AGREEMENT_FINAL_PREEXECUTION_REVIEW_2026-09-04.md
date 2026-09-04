# Final preexecution review: task14 subject–verb agreement capability FIT

**Reviewed:** 2026-09-04 UTC

**Exact candidate commit:** `434f11a927669b86525bf6b9bdc050bd64de544b`

**Verdict:** **APPROVE** exact adapter SHA-256
`ea6acb2a0382a474bda5e48f3c21d368697ab4a7b56adeae489506eff0a25ecd` for exactly one later lane-1
enqueue through the separately reviewed hash-bound managed helper, with that digest supplied as
`EXPECTED_SHA256`. This review does not enqueue, execute, or authorize any other bytes or invocation.

I inspected immutable Git objects and performed CPU/model-free tests only. I did not enter real mode, import the
model, read the checkpoint, touch CUDA/GPU state, inspect a task14 outcome/result/evidence namespace, touch either
queue, enqueue a job, operate the runner/service, publish anything, or open localization or a later phase.

## Exact candidate and ancestry

The candidate changes only the prospective authorization amendment, authorization-enabled adapter, its tests,
checked-in dryrun, and an append-only board entry. Both the implementation and its prerequisite independent reviews
are strict ancestors:

| Object | Exact identity |
|---|---|
| repaired task14 authority build | `e9686bc9bbb40f872d8e8320b30fab4f019e524d` |
| repaired-authority approval | `ea7efad782c088ba91a2ce338a9f740563c4e7c1` |
| capability compiler build | `fc586c1158ddeee7df8f4b502deec54189609c4c` |
| compiler approval | `10afc5d6005d169879b07e92cb5fcb4e3a65f312` |
| producer/blocked-adapter build | `26d45e89797515240eec368bc313728925d5f48a` |
| producer approval | `753afa27e05b594acc39b0c1d84d72272c26e640` |

I recomputed these raw SHA-256 values from the exact `434f11a...` Git objects, independently of the worktree:

| Candidate object | SHA-256 |
|---|---|
| authorization amendment | `e20878d9dcbcf1c2ce0de289a6aed390b44167297a26fe89966c423a010bbee8` |
| authorization-enabled adapter | `ea6acb2a0382a474bda5e48f3c21d368697ab4a7b56adeae489506eff0a25ecd` |
| adapter adversarial tests | `c29b907050e3b5785d8c6b241e08d3fdd0e4e6e8206de95864281bc1c5dc8b74` |
| checked-in dryrun | `dde90907dc64e21c8b97a2ca74768f7068ccbc56d5883132b1b73d8ea42dc47e` |
| unchanged producer | `9ba9448fcebcd764aa2b91e91333b3bbb2549a899b1f8304f2ce3f83bf741e3e` |
| captured producer-review document | `fddb2bac0595f733b765669cb41de1d21ad81a17205df4156505f332c0ea1ccc` |

The exact objects byte-match their current worktree paths. The amendment in turn freezes the generator
`33d7b62b...`, complete authority logical digest `1cf6cf12...`, FIT authority `e88fd860...`, compiler source
`98b2d263...`, preregistration `06a9747b...`, compiler review `a1707dd8...`, compiled contract `84f8e1cf...`, call
manifest `4b4da44c...`, metric manifest `5da9f668...`, implementation preregistration `d84d345c...`, producer,
blocked adapter, and producer review. No upstream object was silently replaced.

## Prospective authorization and phase/price closure

The amendment was frozen prospectively at 08:40 UTC while explicitly awaiting this different-agent review. It changes
no authority row, prediction, threshold, metric, model computation, call request, evidence value, output namespace, or
continuation. Its authority is one managed invocation only, with no automatic retry:

- FIT is the sole captured authority. SELECT, TEST, and OOD are neither generated nor read, and no localization field,
  label, or namespace exists in the adapter closure.
- The literal order is base A1, A2, P, C, then donor A1, A2, P, C: eight forward calls, batch 32, exactly 256 unique
  row-side evaluations.
- Each call may retain only a finite C-contiguous `float32[32]` answer-logit array and matching opposite-copula
  foil-logit array. Sixteen arrays contain exactly 512 float32 values, hence exactly 2,048 raw numeric bytes.
- The only scientific alternatives are the frozen native-capability pass or its exact complement. Capability failure
  is a valid `hard_abort` with every projection field null and no later-phase opening. Malformed runtime, call, array,
  price, namespace, or package state is instrument invalidity, not a scientific failure or retry license.

The checked dryrun reproduces compiled/call/metric digests `84f8e1cf...` / `4b4da44c...` / `5da9f668...`, reports
8/256/2,048 and 24 planned evidence files, and reports zero model loads, forwards, backwards, updates, GPU access,
queue access, and publication. Its captured roles include both the producer review and amendment. Its eight runtime-only
roles—receipt, jacclust package and TT source, observed facade, MLP-in-situ dependency, fastloader, and both canaries—are
excluded. Fresh dryruns under `PYTHONHASHSEED=0`, `1`, and `999` were byte-identical (stdout SHA-256
`d4ce8abe2fba164ed004e054d1bd0743f922036f9a6422d683c129d228012fac`) and logically equal to the checked artifact.

## Real-closure review without real execution

Static inspection and synthetic module-identity tests establish the required ordering before `run_science`:

1. The adapter safely captures the complete base closure and both authorization roles, hash-validates every captured
   byte string, and loads the producer only from those bytes.
2. Real mode adds the captured receipt, jacclust package, `jacclust.tt_model`, observed facade, MLP-in-situ dependency,
   and fastloader in that exact order. The loaded-role set must be exact: neither missing nor enlarged.
3. The producer's compiler/package/framework identities must be the captured objects. The facade and jacclust package
   must bind the captured TT module. Imports performed from the dependency, fastloader, and producer must resolve to
   the corresponding captured objects, and both loading callables must retain their captured module globals.
4. Only after these checks does the exact `EXECUTION_AUTHORIZED=True` adapter delegate once to `run_science`.

The unchanged producer then retains the reviewed runtime/CUDA/version and topology gates; pre/post full checkpoint
rehash; pre/post canaries; bounded native forward path; exact call, label, array, and price validation; and terminal
package validation. A synthetic false authorization flag stopped before science delegation. This review executed no
real branch and loaded no actual model/runtime module.

## Namespace and publication safety

The producer is byte-identical to the approved blocked build. Its reviewed lifecycle remains: require a fresh final
namespace (including dangling-symlink occupation), stage only owned temporary inodes, install 24 evidence files with
Linux `renameat2(RENAME_NOREPLACE)`, install result create-only, and install the package-binding receipt last. A late
race cannot overwrite another inode; rollback removes only matching owned inode identities; a crash or occupied
namespace is instrument invalidity and provides no retry. FIT failure still publishes only the valid all-null scientific
hard-abort package. No producer or publication logic changed in the authorization successor.

## Independent mutation and test results

My additional 36-check in-memory battery accepted the exact dryrun/runtime closure and rejected: individual and joint
amendment/review/producer/compiler/authority byte changes; missing authorization roles; an extra outcome role; adapter
byte mutation; missing or enlarged runtime sets; facade, jacclust, TT, dependency, fastloader, and producer import
substitution; transplanted loading callables with foreign globals; and a false authorization flag. These were synthetic
modules and byte strings only. The checked suites additionally cover source/disk/`sys.modules` poisoning, symlink and
directory capture, ordering, compiled/manifest/price/array/gate mutations, runtime/canary/checkpoint gates, namespace
races, receipt-last publication, rollback ownership, and deterministic dryrun behavior.

- Focused producer + adapter suite: **56 passed** in 2.13 s.
- Relevant broad authority/compiler/producer/adapter/framework suite: **182 passed** in 9.73 s.
- Independently reviewed managed-queue suite: **10 passed** in 0.95 s.
- Static experiment gate: **PASS**; `py_compile`: **PASS**; `bash -n` for enqueue and runner: **PASS**.

All commands set `PYTHONDONTWRITEBYTECODE=1`, `BQLIB_NO_MODEL=1`, and `CUDA_VISIBLE_DEVICES=''` where applicable.

## Final decision and exact execution dependency

**APPROVE** this exact build for exactly one hash-bound managed lane-1 enqueue. The only approved entry is the adapter
at SHA-256 `ea6acb2a0382a474bda5e48f3c21d368697ab4a7b56adeae489506eff0a25ecd`, through independently reviewed
`ops/enqueue.sh` SHA-256 `35baab247d4d358dfaaa76e5862e5ce8fc53a17b181212b66af3212cb8c9649d` and trusted
`ops/bqrunner.sh` SHA-256 `a8b9aae2be074dea1a9f261a329a663ee16e2b45d7c5c8e262d2b5ea3cb40a1e`.
The caller must set `EXPECTED_SHA256` to the exact adapter digest. Enqueue safely preflights captured bytes, stores the
same digest in the queue record, and the runner safely captures and executes only bytes matching it.

This approval is consumed by one invocation. It does not approve direct producer execution, a path-only record,
another adapter digest, a second run, a retry, later phases, localization, or any relaxed scientific bar. No enqueue
was performed by this review.
