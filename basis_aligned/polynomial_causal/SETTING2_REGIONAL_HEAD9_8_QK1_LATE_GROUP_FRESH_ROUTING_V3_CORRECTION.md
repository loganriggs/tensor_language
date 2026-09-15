# Setting2 regional head9.8 QK1 late-group fresh routing V3 correction

V2 completed all 36 frozen model executions, then stopped before result creation
while forming paired reporting vectors. A helper intended for row vectors applied
`[::2]-[1::2]` to the leading arm axis of the `[6,48]` intervention array. This
produced `[3,48]` rather than the required `[6,24]` and raised an index error when
the full-head arm was requested. No scientific metric or terminal was emitted.

V3 replaces that helper with explicit row-axis pairing for the intervention array
and explicit one-dimensional pairing for the three folded vectors. It also reuses
the resulting native paired vector in the descriptive capability count. Rows,
arms, model calculations, source group, prices, metrics, predictions, thresholds,
and interpretation remain those frozen in the V1 preregistration. V1 and V2 are
implementation-invalid and do not count as evidence.
