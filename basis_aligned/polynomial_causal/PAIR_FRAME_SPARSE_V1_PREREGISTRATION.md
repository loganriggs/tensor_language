# Explicit-pair sparse reader graph

September14 10:51UTC. Carrier corrections failed task transfer. Prior balanced
geometry used shared-output K32r8subspaces; this tests a sparse support graph
with fixed semantic sum/difference readers instead of learned shared-output
subspaces. Balanced geometry itself is not a new idea.

Use the six existing token pairs, block-Hadamard output frame with sum and
difference columns, and the original common head frame. Normalize each core
output row by its own weight-only Frobenius norm; retain the globally largest
normalized entries at EXACT original occupied count. Restore actual coefficient
values before FP32 storage. No output-basis fitting or native statistics.

A denseframe/FP32CSRreplay<=1e-6. B same occupiedcount and byteswithin1%.
C total coefficienterror<=.15 and aggregate difference coefficienterror
<=.9original. Report sum and difference relative errors separately. Frozen
candidate will require task-specific native validation; do not claim an
aggregate contrast improvement guarantees prompt-selected task preservation.
CPU120s/two threads/no adaptive extension.
