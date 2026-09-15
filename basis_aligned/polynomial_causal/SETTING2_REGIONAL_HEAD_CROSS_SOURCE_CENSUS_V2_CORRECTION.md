# V2 reporting correction for the source-token census

V1 completed all six native batches and failed before result creation when a
20-token semantic mask indexed the common 21-token padded storage tensor. V2
slices each row to `len(row["ids"])` before applying its unchanged semantic mask.
Rows, tensors, source decomposition, ranking, predictions, thresholds, and price
are unchanged. V1 produced no scientific receipt and remains preserved in its
runner log.
