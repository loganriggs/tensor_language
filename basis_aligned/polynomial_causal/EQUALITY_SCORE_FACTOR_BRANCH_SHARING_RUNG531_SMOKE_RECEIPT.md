# Rung 531 managed smoke receipt

**Completed:** 2026-09-03 12:11 UTC

The one-forward managed smoke completed with exit code 0. It opened no fitted scale, factor-sharing result, circuit
outcome, validation row, or OOD row.

## Frozen identity

- core SHA-256: `e2eb9bd2674247c1fa1c0e25a50d4e747b899a2883899f3074bf809bc676f71e`
- smoke-wrapper SHA-256: `4dfa9bc915ebfbaef701e049c5595f4b9b0a9e028c6b64ea74e756f9d1eb8eb7`
- smoke-log SHA-256: `b2ee34d8d512e5f61d5e72e2ac7e5e1d8ece18e7cdba49fbc011c7bbbdab0659`
- checkpoint SHA-256: `680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3`

## Instrument checks

- model forwards: exactly 1
- captured heads: all four registered heads
- maximum difference between the captured branch product and the independently replayed parent product: `0`
- maximum parent factor reconstruction error: `4.4179630098347173e-14`
- equality edges in the smoke batch after the query-64 cutoff: `1,175`
- peak allocated GPU memory: `3,159,521,280` bytes

The source-frozen 125-forward screen is therefore eligible for the managed queue.
