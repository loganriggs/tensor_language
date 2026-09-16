# Subject-number L11H3 subject-value upstream Möbius V2

Registered after V1 failed instrumentation. The failure was traced to a coding
error in component grouping: string-prefix matching made `attn_10` and `mlp_10`
match both the `*_1` early-layer prefixes and the explicit layer-10 late
prefixes. V1 is retained with terminal `invalid`.

V2 changes only component bookkeeping: every write is keyed by the tuple
`(kind, integer_layer)`, and groups use explicit integer membership in `0..3`,
`4..7`, or `8..10`. Authority, five ports, 32 corners, Möbius transform,
readouts, greedy rule, thresholds, and prices are unchanged. V2 additionally
binds the failed V1 runner and result.

Instrumentation and scientific gates are exactly those in V1. Passing V2
establishes that the earlier negative was bookkeeping rather than evidence
against an upstream decomposition; it does not alter V1's terminal.
