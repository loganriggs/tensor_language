# Conditional head17.2 interaction core

This torch-only package computes the finite mixed write of head17.2 from three declared input conditions: native N, child removal C, and remainder removal R. It reconstructs the additive A corner internally. It is a conditional circuit interface, **not an autonomous language model**.

For every corner, supply five raw preattention-state projections in this order: Q1, K1, Q2, K2, current V. Each has shape `[batch, tokens, 128]`. Supply three norm arrays `[batch, tokens]`, each `mean(raw_state**2)+float32_epsilon`, and one cross array `mean((raw_C-raw_N)*(raw_R-raw_N))` of the same shape. Shared first-layer values have shape `[batch, tokens, 128]`. Inputs used for validation are FP64.

The package retains both normalized/rotated QK factors and the actual signed current/first value mixture. The output is only the final query position's mixed write. Source positions are all causal positions in the supplied prefix.

```python
import torch
from three_corner_head17_interaction_v1 import residual_write

program = torch.load("three_corner_head17_interaction_v1/program.pt",
                     weights_only=True)
write = residual_write(
    [native_projections, child_projections, remainder_projections],
    [native_norm, child_norm, remainder_norm],
    change_inner_product,
    first_values,
    float(program["mixture"]),
    output_matrix=program["output_matrix"].to(
        device=first_values.device, dtype=first_values.dtype),
    compact=False,
)
# write: [batch, 1152]
```

`compact=False` evaluates the full mixed-product expansion. `compact=True` uses the fixed three-contraction approximation. The latter was selected on development data and confirmed on new cities/constructions; it is not an exact identity.

To reproduce the tested behavioral effect, add the write to the specified final additive background `h_C + h_R - h_N` and use the model's actual final readout. Do not silently propagate this direct-write output through MLP17: that is a different conditional effect. Native background and readout generation are outside this package.

The underlying core passed the actual three-trajectory test on120 reused regional prefixes:360model forwards,360readouts,6.76seconds recorded experiment-body time. No P or A trajectory was executed. Saved predictions matched the earlier five-trajectory reconstruction. This is an interface validation result, not new OOD evidence. The exported copy and reloaded output weights have independently imported CPU equivalence checks, with identical outputs; the exported copy has not had a separate native GPU deployment.

`PRICE.json` charges147,457 locally stored scalars. It explicitly lists required external generators, weights, states and readout. Input width is1920 projection scalars plus four normalization/cross scalars and128 shared-value scalars per token. Only final-query projections are mathematically needed, but the present implementation carries all positions. Local storage is not whole-model storage.

Primary evidence: [derivation and native results](../../ADDITIVE_HEAD_RAW_PORTS_V1_MATH.md), [three-trajectory result](../../THREE_TRAJECTORY_HEAD_PREDICTOR_V1_RESULT.json), [export control](../../THREE_CORNER_HEAD_EXPORT_V1_CONTROL.json).
