# All-query-fixed correction of the O source-swap interaction test

V1 correctly freezes `query_current` to the recipient tensor while separately
patching source keys and values. The earlier `ODD_FRAMING_SOURCE_SWAP_V1`
instead passed a source-patched current tensor to `source_channels`, changing
queries at patched nonfinal positions as well as source keys/values. Both keep
the final query fixed, but their full-swap score tensors need not replay. V1's
`pred_a` incorrectly required that replay and failed by `.01098`; preserve it.

V2 keeps all V1 rows, five arms, values, 240-forward price, and predictions
B-E unchanged. Prediction A now requires the native anchor only, exact replay
of the generalized kernel when query/key/value currents coincide, exact
routing+value+mixed expansion, a live mixed-source term, finite outputs, and
240 forwards. The gap to the legacy query-mutating full swap is reported rather
than gated. This correction is frozen without using V1 to change B-E.

