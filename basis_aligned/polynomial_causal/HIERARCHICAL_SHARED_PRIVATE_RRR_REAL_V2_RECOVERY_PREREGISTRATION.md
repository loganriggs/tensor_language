# Hierarchical shared/private RRR real v2 recovery

**Status:** prospective one-change recovery. No v2 authority, row/model load, GPU
execution, result, failure, or receipt exists. The spent v1 artifacts are immutable
parents and their numerical values may not select or alter any arm, rank, budget,
prediction, gate, control, data role, or execution schedule.

## Exact spent parent

V1 completed all seven arms and wrote a result, then failed before receipt publication
because an in-memory diagnostic contained `HierarchicalPrice.private_ranks` as a tuple.
JSON publication canonically changed that tuple to a list, so the inherited strict
`json.loads(result) == in_memory_result` guard rejected it. Preserve and bind:

- v1 authority SHA256 `558d316eb5fdb4a4249eb58cdd5c2b80f0005873cdbccf517dfede7226c4d11c`;
- v1 result SHA256 `86315dcc855e9a27958b6abfd50ed5c6b7bb7108f00fe3684bfbf624405a772d`;
- v1 failure SHA256 `054db06c03525b3f78eefdd9ed8e0fa3daf3868175460c76a95e39b875ebc35c`;
- v1 receipt absent.

The failure must retain `error="shared-RRR result replay changed"`, authority and result
present with those exact hashes, and receipt absent. The v1 result may be semantically
replayed to establish lifecycle integrity but its scientific values are spent and may
not influence v2.

## Only licensed change

Immediately after each unchanged v1 `fit_program` returns and before its diagnostics
can enter result assembly, transform the diagnostics through exactly
`json.loads(json.dumps(diagnostics, sort_keys=True, allow_nan=False))`. This changes
only JSON container representation (not numerical values, tensor factors, hashes,
ranks, prices, gates, calls, or execution). The normalized object must be idempotent
under JSON round-trip and is the object evaluated, assembled, published, and reloaded.

No alternative fix, refit, result reuse, arm deletion, threshold change, or target
inspection is licensed. V2 reruns the original rows/model execution because v1 did not
produce a receipt-authorized scientific result.

## Fresh lifecycle

Use only `hierarchical_shared_private_rrr_real_v2_recovery_{authority,results,failure,receipt}.json`
and `/workspace/runs/.hierarchical_shared_private_rrr_real_v2_recovery.lock`. The v1
authority/result/failure are pinned inputs and protected at every frozen-input replay.
All source, checkpoint, row, shared-output parent, seven-arm, float32 deployment,
physical call, CE, control, resource, semantic result, create-only failure, and
receipt-last rules remain byte-for-byte v1 rules except for the normalization above.

Before authority and at every subsequent frozen-input boundary, re-read/hash/re-read
all three v1 parents, require v1 receipt absence, replay v1 authority/result/failure
joins and the v1 semantic result validator, and require the exact failure semantics.
The new source must be committed and pushed before authority. Receipt remains the final
write after exact result reload and all terminal source/input/checkpoint/parent checks.

Authority scope remains discovery-only with no validation, final, promotion,
generalization, semantic-coordinate, or serialized-program license.
