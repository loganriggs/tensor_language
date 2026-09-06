"""Capture, assemble, and account for exact residual-block state components."""

# BQGATE: LIBRARY
from __future__ import annotations

import itertools
import math


COMPONENTS = ("entry", "attention", "mlp")


class BlockComponentStateError(RuntimeError):
    pass


def subsets():
    return tuple(
        subset for width in range(len(COMPONENTS) + 1)
        for subset in itertools.combinations(COMPONENTS, width)
    )


def arm_id(subset):
    return "empty" if not subset else "+".join(subset)


def assemble(base_components, changed_components, subset):
    selected = set(subset)
    pieces = tuple(
        changed_components[name] if name in selected else base_components[name]
        for name in COMPONENTS
    )
    return (pieces[0] + pieces[1]) + pieces[2]


def capture(backend, batch, block_index, call):
    block_index = int(block_index)
    if not 0 <= block_index < len(backend.model.transformer.h):
        raise BlockComponentStateError("block index is invalid")
    block = backend.model.transformer.h[block_index]
    captured = {}

    def attention_output(_module, _arguments, output):
        captured["attention"] = output[0].detach().clone()

    def mlp_output(_module, _arguments, output):
        captured["mlp"] = output.detach().clone()

    handles = [
        block.attn.register_forward_hook(attention_output),
        block.mlp.register_forward_hook(mlp_output),
    ]
    try:
        output, states = call()
    finally:
        for handle in handles:
            handle.remove()
    if set(captured) != {"attention", "mlp"} or len(states) <= block_index + 1:
        raise BlockComponentStateError("block component capture is incomplete")
    entry = (
        block.lambdas[0] * states[block_index]
        + block.lambdas[1] * states[0]
    ).detach().clone()
    components = {
        "entry": entry,
        "attention": captured["attention"],
        "mlp": captured["mlp"],
    }
    reconstructed = assemble(components, components, COMPONENTS)
    error = float((reconstructed.float() - states[block_index + 1].float()).abs().max())
    return output, states, components, error


def factorial_accounting(values):
    expected = set(subsets())
    if set(values) != expected:
        raise BlockComponentStateError("factorial subset coverage changed")
    count = len(COMPONENTS)
    shapley = {}
    for component in COMPONENTS:
        total = 0.0
        for subset in subsets():
            if component in subset:
                continue
            extended = tuple(
                name for name in COMPONENTS if name in set(subset) | {component}
            )
            weight = (
                math.factorial(len(subset))
                * math.factorial(count - len(subset) - 1)
                / math.factorial(count)
            )
            total += weight * (float(values[extended]) - float(values[subset]))
        shapley[component] = total
    error = abs(
        sum(shapley.values())
        - (float(values[COMPONENTS]) - float(values[()]))
    )
    return {"shapley": shapley, "efficiency_error": error}
