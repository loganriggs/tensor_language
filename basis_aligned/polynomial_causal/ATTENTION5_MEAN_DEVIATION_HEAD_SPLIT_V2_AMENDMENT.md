# ATTENTION5_MEAN_DEVIATION_HEAD_SPLIT_V2 amendment

V1 stopped before any score was produced: its loss reshape used vocabulary
`50257` and context length`256`, copied from a different runner, while this
checkpoint and the frozen document cache use padded vocabulary`50304` and
context length`257`. V2 changes only those two instrument constants and the
output stem. The24/24 frozen split, all arms, heads5/7 prospective grouping,
predictions, thresholds, interpretation, and49-forward budget in the V1
preregistration remain unchanged. The V1 failure log is retained.
