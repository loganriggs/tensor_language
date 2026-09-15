# Setting2 regional head9.8 QK-source fold V3 correction

V2 passed the repaired raw-state audit and completed all prefix computations, but
then exited before writing a result when its QK score audit concatenated batches
with different sequence lengths. V3 accumulates squared QK closure numerators and
denominators within each batch, exactly as V2 already does for raw-state closure.

No scientific values were emitted by V1 or V2. Rows, source definitions, native
denominators and values, downstream fold, price, predictions, thresholds, outcome
prohibitions, and ranking rules remain frozen. V3 binds both failed runners and
both correction records before execution.
