# Terminal-copy fit-input exposure erratum

Status: **preserved engineering exposure; no E4 model forward or outcome occurred**.

Before a fit execution authority existed, development code and a unit test loaded the
published `fit_natural.pt` container with `torch.load`. The code indexed only `rows`
and `records`, but deserialization materialized the whole container, which also holds
copy-cell masks, synthetic rows, and the 257th next-token label column. Therefore a
future authority must not claim to precede all row or label-container access.

No bilin18 checkpoint was loaded by that test, no E4 model forward was run, no logits,
losses, candidate effects, thresholds, final rows, or OOD rows were inspected, and the
eight candidate bank was already frozen. The repair is a new source-closed,
outcome-blind projection transaction. It publishes exactly the first 256 input tokens
and ordered document IDs for the 192 licensed fit documents. The E4 fit runner may
load only that sanitized artifact. Its authority must say it was frozen before any E4
fit model forward **after this disclosed parent-container engineering exposure**.

The real-row unit test is replaced by a synthetic fixture. This erratum is permanent;
later successful receipts may bind it but may not erase or reinterpret it.
