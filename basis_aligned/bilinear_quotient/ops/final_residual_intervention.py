"""Capture or replace the raw final residual at the last transformer block."""

# BQGATE: LIBRARY
from __future__ import annotations


class FinalResidualInterventionError(RuntimeError):
    pass


def replace_positions(state, replacement, position_rows):
    if state.shape != replacement.shape or len(position_rows) != state.shape[0]:
        raise FinalResidualInterventionError("final residual replacement shape changed")
    changed = state.clone()
    for index, positions in enumerate(position_rows):
        changed[index, list(positions)] = replacement[index, list(positions)].to(
            device=state.device, dtype=state.dtype)
    return changed


def execute(model, call, *, replacement=None, position_rows=None):
    """Run ``call`` once while capturing or replacing block-17's returned state."""
    captured = {}
    calls = 0

    def hook(_module, _inputs, output):
        nonlocal calls
        calls += 1
        if not isinstance(output, tuple) or not output or not hasattr(output[0], "detach"):
            raise FinalResidualInterventionError("final block output signature changed")
        state = output[0]
        captured["state"] = state.detach().clone()
        if replacement is None:
            return None
        if position_rows is None:
            raise FinalResidualInterventionError("replacement positions are missing")
        changed = replace_positions(state, replacement, position_rows)
        return (changed,) + tuple(output[1:])

    handle = model.transformer.h[-1].register_forward_hook(hook)
    try:
        result = call()
    finally:
        handle.remove()
    if calls != 1 or set(captured) != {"state"}:
        raise FinalResidualInterventionError("final residual hook coverage changed")
    return result, captured["state"], calls
