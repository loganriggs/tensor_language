# Setting2 regional head9.8 QK-source fold V2 correction

V1 exited before writing a result because its block-9 raw-state audit allocated
one 1152-vector per row but attempted to store all sequence positions. This is an
instrument shape error; no QK terms, rankings, or predictions were observed.

V2 replaces that fixed-row audit buffer with accumulated squared numerator and
denominator norms across each variable-length batch. It separately reports exact
double-precision source recomposition and the bridge to the native float32 raw
state. All rows, source definitions, QK/value factors, downstream fold, price,
predictions, thresholds, outcome prohibitions, and ranking rules remain exactly
as preregistered. V2 binds this correction and the failed V1 runner before
execution.
