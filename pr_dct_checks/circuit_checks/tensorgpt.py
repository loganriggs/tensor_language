"""Adapter: AJ's TensorGPT (tensor_model.py) -> the span interface.

Replicates `IntermediateMLPFeatures.forward` from
sparse-asymmetric/intermediate_mlp_pr_experiment.py, with three changes:
  1. the context state is closed over explicitly (no nearest-neighbour lookup);
  2. MLP hidden units and attention outputs can be frozen / patched to clean;
  3. hidden activations are recorded for ALL blocks in the span, including
     block `source` (AJ's PR uses only the exclusive range source+1..target-1).

UNTESTED against real weights (written without GPU access). Validate with
`scripts/check_adapter_parity.py` before trusting any numbers.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .core import FreezeSpec, apply_freeze


def hidden_features(mlp, x):
    """Pre-down-projection activations; copied from AJ's code."""
    if hasattr(mlp, "Left"):
        left = mlp.Left(x)
        if mlp.config.gated:
            left = left * torch.sigmoid(left)      # SiLU from primitives: F.silu's backward has no forward-mode rule, and E1-E3 need jvp-of-jvp
        return left * mlp.Right(x)
    h = mlp.c_fc(x)
    return h.square() if mlp.config.squared_mlp else F.relu(h).square()


def down_project(mlp, h):
    if hasattr(mlp, "Down"):
        return mlp.Down(h) + mlp.Down_bias
    return mlp.c_proj(h) + mlp.bias


def read_weights(mlp):
    """(W_left, W_right) rows for bilinear MLPs; None for other MLP types."""
    if hasattr(mlp, "Left") and not mlp.config.gated:
        return mlp.Left.weight, mlp.Right.weight
    return None


class TensorGPTSpans:
    def __init__(self, model, source_layer, target_layer, target_positions=slice(-3, None), final_norm=False):
        assert 0 < source_layer < target_layer <= model.config.n_layer
        self.model = model
        self.final_norm = final_norm            # apply the model's final rms_norm to the output (target = what the unembedding reads)
        self.source, self.target = source_layer, target_layer
        self.blocks = model.transformer.h[source_layer:target_layer]
        self.tp = target_positions
        self.d = model.config.n_embd

    @property
    def all_layers(self):
        return list(range(self.source, self.target))

    @property
    def aj_pr_layers(self):
        return list(range(self.source + 1, self.target))

    def mlp_width(self, layer):
        mlp = self.model.transformer.h[layer].mlp
        return (mlp.Left if hasattr(mlp, "Left") else mlp.c_fc).weight.shape[0]

    def _run(self, ctx, theta, freeze: FreezeSpec | None, stop_at_mlp_input=None, full=False):
        values, init, first = ctx
        values = values + theta
        hidden, hidden_full, attn_full = {}, {}, {}
        for rel, block in enumerate(self.blocks):
            L = self.source + rel
            values = block.lambdas[0] * values + block.lambdas[1] * init
            attn, first = block.attn(F.rms_norm(values, (self.d,)), first)
            if freeze is not None and L in freeze.attn_layers:
                clean = None if freeze.clean_attn is None else freeze.clean_attn[L]
                attn = apply_freeze(attn, torch.ones(attn.shape[-1], device=attn.device), clean)
            attn_full[L] = attn
            values = values + attn
            mlp_in = F.rms_norm(values, (self.d,))
            if stop_at_mlp_input == L:
                return mlp_in[:, self.tp].mean(1)[0]
            h = hidden_features(block.mlp, mlp_in)
            if freeze is not None and L in freeze.mlp_masks:
                clean = None if freeze.clean_hidden is None else freeze.clean_hidden[L]
                h = apply_freeze(h, freeze.mlp_masks[L].to(h.device), clean)
            hidden_full[L] = h
            hidden[L] = h[:, self.tp].mean(1)[0]
            values = values + down_project(block.mlp, h)
        if self.final_norm:
            values = F.rms_norm(values, (self.d,))
        out = values[:, self.tp].mean(1)[0]
        if full:
            return out, hidden_full, attn_full
        return out, hidden

    def make_span(self, ctx):
        """ctx = (values, initial_values, first_values), each [1, T, d], float32."""
        def span(theta, freeze=None):
            return self._run(ctx, theta, freeze)

        def clean_cache():
            with torch.no_grad():
                _, h, a = self._run(ctx, torch.zeros(self.d, device=ctx[0].device), None, full=True)
            return {"hidden": {k: v.detach() for k, v in h.items()},
                    "attn": {k: v.detach() for k, v in a.items()}}

        span.clean_cache = clean_cache
        return span

    def mlp_input_fn(self, ctx, layer):
        """theta -> normalized MLP input at `layer`, averaged over target positions."""
        return lambda theta: self._run(ctx, theta, None, stop_at_mlp_input=layer)
