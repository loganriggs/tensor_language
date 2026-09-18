# City reader at a residual6 prefix boundary

PyTorch-only executable for the complete head8.2 city-value removal write at
attention8 output. Attention7 and MLP7 city readers are generated from an earlier
residual6 prefix and weight-derived token tables. The full native later model
remains external. This export does not use the unconfirmed mixed8 RMS approximation.

```python
import torch
from execute import execute
program = {name: torch.load(name + ".pt", weights_only=True)
           for name in ("attention7", "readers", "head8")}
delta = execute(program, residual6, token_ids, city,
                rotated_queries, destination, input_rms)
```

Certified batch size1. `residual6` and matching token IDs extend through the city;
queries have shape[1,T,2,128]; destination is a T-element Boolean mask; input_rms
is native mixed8 city RMS with epsilon already included. All arrays must share
a device. Only396frozen token IDs are supported; unknown tokens raise ValueError.
Output shape[1,T,1152], float64, zero outside support. Add after casting to the
native attention8 output dtype, then recompute MLP8 and all later layers.

Price21,851,526FP32values /87,406,104bytes. AtT32/city12 the native inputs total
23,169scalars, compared with10,497at the preceding local reader boundary. The
input dependency moves earlier; weights and supplied-state count grow. No total
model compression, token-only extraction, fresh OOD/selectivity confirmation or
independent composition claim. Native query and RMS generation remain open.

The opened full-model CPU replay passed all registered gates against the saved
GPU reference. Isolated replay is recorded separately in the manifest. Do not
mistake the earlier reader-boundary GPU checks for a certificate of this package.
