# MLP2 CMR v1R CPU-finalization amendment

Frozen after the v1 physical run completed all 48 batches and before any value from
its invalid result or sufficient-statistics ledger was inspected.

The v1 receipt-last replay failed only because the raw-payload guard treated the
metadata leaf `role_summary.tensor_hashes.rows` as though it were row data.  The leaf
is a lowercase SHA-256 string.  V1 is terminally spent: its authority, ledger,
result, failure, and lock are immutable, and its receipt is absent.

V1R performs no model forward, row deserialization, token access, fitting, selection,
or replication access.  It may only, after publishing a fresh create-only authority,
deserialize the exact hash-pinned v1 ledger/result and JSON metadata parents needed
for semantic replay.  It recomputes every protocol gate and scientific score from
the sufficient statistics, compares the complete role summary to the independently
published role-receipt summary, and publishes a distinct v1R decision and receipt
last.  A v1R success does not erase or replace the v1 failure.

The corrected raw-payload predicate keeps `rows` forbidden everywhere except the
exact path `role_summary.tensor_hashes.rows`; at that path the value must have type
`str` and match `[0-9a-f]{64}`.  Tensors, bytes, and actual raw logits, candidate or
native logits, per-token losses, validation targets, rows, tokens, targets, products,
states, and responses remain forbidden recursively.

The v1R authority pins the v1 authority, ledger, result, failure, and lock hashes,
the absence of the v1 receipt, all 17 v1 parent hashes, and the complete v1R source
closure.  V1R uses separate authority/result/receipt/failure/lock paths.  Any late
change to v1 artifacts, parents, source, lock identity, or receipt absence blocks a
v1R terminal artifact.  Receipt and failure are mutually exclusive.

