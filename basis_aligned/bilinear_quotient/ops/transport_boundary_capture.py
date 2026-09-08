"""Reusable exact boundary capture for residual-path circuit transport tests."""
from __future__ import annotations


class TransportBoundaryError(RuntimeError):
    pass


def execute_with_boundaries(model, execute, *, block_layers, raw_attention_layer, mlp_input_layer):
    """Execute once while capturing block inputs, one attention write, and one MLP input."""
    layers = tuple(int(layer) for layer in block_layers)
    if len(layers) != len(set(layers)) or raw_attention_layer not in layers or mlp_input_layer not in layers:
        raise TransportBoundaryError("boundary layer inventory is inconsistent")
    saved, calls, handles = {}, {}, []
    for layer in layers:
        calls[f"block{layer}"] = 0
        def block_pre(_module, args, layer=layer):
            if len(args) < 3:
                raise TransportBoundaryError("block call does not expose x, v1, and x0")
            calls[f"block{layer}"] += 1
            saved[f"block{layer}_input"] = args[0].detach().clone()
            saved[f"block{layer}_x0"] = args[2].detach().clone()
        handles.append(model.transformer.h[layer].register_forward_pre_hook(block_pre))
    calls["attention_write"] = 0
    def attention_write(_module, _args, output):
        calls["attention_write"] += 1
        saved[f"attn{raw_attention_layer}_write"] = output.detach().clone()
    handles.append(model.transformer.h[raw_attention_layer].attn.c_proj.register_forward_hook(attention_write))
    calls["mlp_input"] = 0
    def mlp_input(_module, args):
        calls["mlp_input"] += 1
        saved[f"M{mlp_input_layer}_input"] = args[0].detach().clone()
    handles.append(model.transformer.h[mlp_input_layer].mlp.register_forward_pre_hook(mlp_input))
    try:
        result = execute()
    finally:
        for handle in handles:
            handle.remove()
    if any(value != 1 for value in calls.values()):
        raise TransportBoundaryError(f"boundary hook counts changed: {calls}")
    return result, saved, calls


def pre_mlp_raw_state(model, state, *, layer):
    """Reconstruct lambda-mixed residual plus captured attention write before RMSNorm."""
    block = model.transformer.h[layer]
    required = (f"block{layer}_input", f"block{layer}_x0", f"attn{layer}_write")
    if any(key not in state for key in required):
        raise TransportBoundaryError("captured state lacks a raw-state constituent")
    return (block.lambdas[0] * state[required[0]]
            + block.lambdas[1] * state[required[1]]
            + state[required[2]])
