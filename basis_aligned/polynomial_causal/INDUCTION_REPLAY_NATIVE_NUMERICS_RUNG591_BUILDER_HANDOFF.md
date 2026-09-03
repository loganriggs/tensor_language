# R591 diagnostic builder handoff

Date: 2026-09-03 UTC

Status: **READY FOR DIFFERENT-AGENT REVIEW.** This is a builder handoff, not an
approval and not permission to enqueue.

## Exact candidate bytes

| Artifact | SHA-256 |
|---|---|
| `ops/induction_replay_native_numerics_rung591.py` | `b2b266529f0f842211fea46856064133df5e3f4a8a7758c9095e7d29a94b6c49` |
| `ops/test_induction_replay_native_numerics_rung591.py` | `e756ba3d17d3ebee2f81e97e573dd216090555de1fd3f1cfc926268f902d9ce7` |
| `induction_replay_native_numerics_rung591_dryrun.json` | `161193de5d90da69aafcd681e375993fa91d32e99100f0ed02fb586d5a629d8b` |
| amended preregistration | `e72cb386d65c68f55b767c8141c3c4d774b3c8ad9387ac7f8ad43bebef118593` |

The implementation independently verifies the exact R585 producer, owner test,
dry run, facade, induction helper, manifest, dependency lock, and amended
preregistration hashes before either dry-run construction or diagnostic work.
It also binds the phase-specific panel-method handoff v5 at commit
`f69f695da1cae57cbb79326a859d58c401f2473a`, SHA-256
`810d15aa7f86a9896ca56e48c7ea33c60b10f6b0d266acefa5f3441333c8fe80`.
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
validation. With the environment variable absent, the managed no-argument path
runs the diagnostic. The only diagnostic output is one strict-finite JSON object
on stdout. The script contains no R585 result, receipt, or evidence namespace;
no scoring, selection, scientific terminal, or publication function is called.
The absolute `1e-5` threshold and the registered observer/hook/padding/batch
interpretation remain unchanged. Even a passing diagnostic cannot license R585
science.

## Builder verification

- owner suite: `15 passed`;
- deterministic managed dry run: exact match to the committed JSON;
- static gate: `PASS` with no findings;
- preflight: no findings;
- model/GPU/queue/outcome access: none.

A different agent should independently attack panel membership, realized call
census, dispatcher return aliases, the four-site delta, paired-comparison
coordinates, strict-finite maxima, classification precedence, and the absence of
all publication paths before any enqueue decision.
