# R591 diagnostic builder handoff

Date: 2026-09-03 UTC

Status: **READY FOR DIFFERENT-AGENT REVIEW.** This is a builder handoff, not an
approval and not permission to enqueue.

## Exact candidate bytes

| Artifact | SHA-256 |
|---|---|
| `ops/induction_replay_native_numerics_rung591.py` | `fb8239ded4f3e99510f37ea72337c2d69e4640f7a2556748c9062aa82b2751bc` |
| `ops/test_induction_replay_native_numerics_rung591.py` | `8a24a9903d10ada8a4048c7adcb33cb4ef3e8aeef11d6f9718f8e50e57b6212c` |
| `induction_replay_native_numerics_rung591_dryrun.json` | `8a6331fb1a4d3800abff5ab6b7e291105872b06b41a43b003436312b6e50dc5d` |
| amended preregistration | `2dd8f918f767a6e5d91af357cfaa14770b79334ebac837d1bf52e8046ce190a5` |

The implementation independently verifies the exact R585 producer, owner test,
dry run, facade, induction helper, manifest, dependency lock, and amended
preregistration hashes before either dry-run construction or diagnostic work.
It also binds the phase-specific panel-method handoff v5 at commit
`f69f695da1cae57cbb79326a859d58c401f2473a`, SHA-256
`810d15aa7f86a9896ca56e48c7ea33c60b10f6b0d266acefa5f3441333c8fe80`.
It also binds shared handoff v6 at commit `4eb17c5cc`, SHA-256
`d1fdedd90ffff29e6790042b9c9a6ad84278849c3f66707cb586317832fdad1c`.
The checkpoint receipt must equal
`680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3`.

## Frozen computation

The full-FIT portion executes replay on the mixed schedule, native on the same
mixed token tensors, and native on the length-sorted schedule: 162 forwards.
The amended controlled panel uses the first 64 FIT endpoint IDs at each of
lengths 19, 20, 27, and 28. Native, factor-observer-with-no-write, and current
replay are crossed with native length, forced length 30, and fixed-shape mixed
membership: 72 forwards. Total price is exactly 234 forwards, zero backwards,
and zero updates.

The ordered panel hash is
`6b56a6740dbea7d0765d6a8668361ff43b06562152f091f6969ca8591522ebe4`;
the receipt now also emits all 256 ordered FIT endpoint IDs and their direct
ordered-list hash `cb0112c4b750bd4f3b104595efef127eed8aff4438e485f8cb5d8c0806fc7f7d`;
the ordered forward-call hash is
`1e838190752e72eed6f35119c3e99bfb7620e787ae73c7a052046160d600ad3f`.
Dispatcher counts are N=132, F=24, R=78. The F dispatcher returns the exact
native attention-write object. R clones that write and adds exactly
`term - canonical` at L5H5, L7H3, L8H3, and L8H4. All model calls use the
dynamic-shape facade path while the loaded model and checkpoint are validated
separately.

The model-free equality-support audit covers all 2,592 authority endpoints. It
finds support-count histogram `{0:432, 1:2160}`, zero unregistered canonical
positions, zero missing registered positions, and ordered census hash
`e2de29dcf3cb37187060ab72775533086612bbb349777d48bd9f8feb8911e9fa`.
This rules out omitted equality successors as the cause under the pinned
semantic authority.

## Boundary and output

`BQLIB_DRYRUN=1` executes only source, authority, schedule, support, and shape
validation. Endpoint authority is constructed from immutable snapshots of the
direct R578 rows and R585 manifest; it cannot parse R586/R587 outcome artifacts.
With the environment variable absent, the managed no-argument path
runs the diagnostic. The only diagnostic output is one strict-finite JSON object
on stdout. The script contains no R585 result, receipt, or evidence namespace;
no scoring, selection, scientific terminal, or publication function is called.
The absolute `1e-5` threshold and the registered observer/hook/padding/batch
interpretation remain unchanged. Only N comparisons can activate the registered
native padding and membership causes; auxiliary F/R comparisons remain descriptive.
Executable project dependencies are hash-snapshotted before import, and the managed
adapter executes an immutable snapshot of the checked producer bytes. Even a passing diagnostic cannot license R585
science.

## Builder verification

- owner suite: `19 passed`;
- deterministic managed dry run: exact match to the committed JSON;
- static gate: `PASS` with no findings;
- preflight: no findings;
- model/GPU/queue/outcome access: none.

A different agent should independently attack panel membership, realized call
census, dispatcher return aliases, the four-site delta, paired-comparison
coordinates, strict-finite maxima, classification precedence, and the absence of
all publication paths before any enqueue decision.
