# Rung448 preregistration: MLP0 teaching-family removal and composition labels

Status: registered before loading the model or TEACHING rows for a forward pass. GPU execution must use the managed
queue. SEALED_CONFIRMATION is forbidden.

## Objects

The five rung445 candidates are the physical mixed104 MLP0 context-input programs at ranks
`{256,384,448,512,640}`. Each is deterministically rebuilt from the frozen first24 rows of
`fineweb_n192_skip11000.pt` by the existing context covariance and reduced-rank program algorithm.

The fixed independent composition partner `Q` is the physical14,984-value, no-dense MLP16 rank2 quadratic program.
The fixed removal is a native attention16 mean knockout. Its1152-vector mean is estimated once from the first128
candidate-fit rows under the native model and then held identical for every arm.

## Exact computations

For each candidate `P`, collect per-token cross-entropy vectors on the96 TEACHING rows:

- native `CE_N` and native knockout `CE_NKO`;
- candidate `CE_P` and candidate knockout `CE_PKO`;
- fixed partner `CE_Q` and physical composition `CE_PQ`.

Removal effects are `R_N=CE_NKO-CE_N` and `R_P=CE_PKO-CE_P`. Record cosine, normalized error
`||R_P-R_N||/||R_N||`, norm ratio, and errors on target/collateral masks. The target mask is the top quartile of
`|R_N|`; collateral is its bottom half, both frozen before candidate metrics.

Composition compares physical damage `J_P=CE_PQ-CE_N` with the additive prediction
`A_P=(CE_P-CE_N)+(CE_Q-CE_N)`. Record cosine, normalized error `||J_P-A_P||/||A_P||`, mean interaction, and CE
damage. Every metric is reported on the full role and separately on rows0:48 and48:96.

## Frozen predictions and null

- **A — instrument:** exact checkpoint, bank, row receipt/tensor, fit-cache, and factor-program hashes; native replay
  max zero; five program encoders have exact registered ranks/shapes; candidate, partner, and knockout dispatches are
  live; SEALED row hash is never loaded.
- **B — structural ordering:** Spearman of rank with negative removal error and negative composition error is at
  least`.50` for both full-role labels.
- **C — learnable variation:** max-minus-min normalized removal error and composition error are each at least`.015`.
- **D — row stability:** between the two fixed waves, Spearman correlation of the five candidate removal-error
  orderings and composition-error orderings is at least`.70` each.

**Strong null:** native removal norm is numerically dead; fixed partner or every candidate has absolute CE damage
below`1e-4`; both label spans are below`.003`; either wave-order correlation is negative; or any instrument clause
fails. Under it, preserve labels but do not count this family toward learned-simplicity fitting.

A/B/C/D with no null makes this one teaching family eligible. It does not fit a predictor, open sealed labels, name
MLP0 semantics, or license compression/adoption. Literal new deployed price is zero; labels retain each candidate's
registered complete price and charge partner Q separately only for composition.

