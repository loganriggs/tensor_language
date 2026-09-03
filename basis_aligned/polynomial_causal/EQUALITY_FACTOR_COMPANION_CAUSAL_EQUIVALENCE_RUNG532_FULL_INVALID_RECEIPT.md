# Rung 532 first full launch: structural instrument failure

**Completed:** 2026-09-03 12:32 UTC

The managed full launcher exited 1 after one direct-native forward, before any analytical arm or circuit/loss outcome
was accumulated. The circuit-mask count accumulator had shape `[mask_type, tag, batch, token]` but summed axes
`(tag,batch)` rather than `(batch,token)`. It therefore produced a `2 x 256` tensor for a `2 x 32` destination.

- failed core SHA-256: `2207288b731f69a5b540ab101d3b293d2f5ff7831f347d8e59ac00bf7e59e9e2`
- failure-log SHA-256: `9b6bee57ba2a91a50c62a0c5cb6f8d1099cc7d3aff44acb7381717b95f53e3b4`
- model forwards consumed: 1
- scientific outcomes retained: none

The correction sums axes `(batch,token)`, adds a focused shape test, and makes the next smoke execute the actual
support accumulator while still withholding loss/circuit outcomes. A separately named v3 smoke is required before
another full launch.
