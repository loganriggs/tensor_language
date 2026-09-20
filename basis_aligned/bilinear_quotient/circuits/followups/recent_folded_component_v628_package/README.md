# Extracted recent-path component

`execute.py` and the three hash-checked weight shards run without the original
checkpoint, model class, tokenizer, or repository imports. Requires PyTorch.

```python
import execute
w = execute.load_package("/path/to/this/directory", device="cpu")
result = execute.execute(w, h16, x0, v1, scale=1.0)
alpha = result["alpha"]
```

Required upstream ports, all from the same context:

* `h16`: residual immediately before MLP16, shape `[batch,tokens,1152]`.
* `x0`: normalized token embedding, same shape.
* `v1`: first-layer attention value, `[batch,tokens,9,128]` or equivalent
  contiguous flattened last dimension.

`scale` scales the entire MLP16 write, including its bias. The runtime
recomputes attention17, all relevant normalizations, and the selected
quadratic component. It returns its scalar, the complete recent h17 state,
MLP16 write, attention17 write, and background/cross/MLP16-square terms.
Multiply `alpha[...,None]` by `vocabulary_writer` for the projected numerator
contribution, or by `residual_writer` for the corresponding residual direction.
Full model logits still require the actual final residual, RMSNorm and softcap.

This is **conditional extraction**, not an independent token-to-output circuit.
The upstream generators of h16, x0 and v1 remain required. The program stores
24,235,715 tensor values (~97 MB); it is not a simplicity result. Its purpose
is an exact recent-path reference for subsequent shared/sparse decomposition.

Native validation is in `../recent_folded_component_v628_result.json`.
`isolated_smoke.json` records a separate `python -I` CPU execution with
synthetic ports and only these shards. That smoke proves package execution,
not behavioral validity of synthetic states. The manifest hashes the runtime
and all weight shards. The native receipt initially records the three shard
hashes; the runtime hash was added during packaging after the native run.
