#!/usr/bin/env python3
# BQGATE: LIBRARY
"""Shared no-model preflight helpers for exact self interventions and predicates."""
from __future__ import annotations

import ast


def static_prediction_keys(source: str) -> list[str]:
    """Return statically declared prediction keys without scanning prose."""
    tree = ast.parse(source)
    found = []

    def add(value):
        if isinstance(value, str) and value.startswith("pred_") and value not in found:
            found.append(value)

    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key in node.keys:
                if isinstance(key, ast.Constant):
                    add(key.value)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id == "dict":
            for keyword in node.keywords:
                add(keyword.arg)
        elif isinstance(node, (ast.Tuple, ast.List)) and len(node.elts) >= 2:
            first = node.elts[0]
            if isinstance(first, ast.Constant):
                add(first.value)
    return found


def exact_static_first_value(model, cfg, token_matrix, functional):
    """Compute block-0 values with native full-batch shape and operation order."""
    if getattr(token_matrix, "ndim", None) != 2:
        raise ValueError("token matrix must have [batch, sequence] shape")
    dimension = int(cfg["n_embd"])
    heads = int(cfg["n_head"])
    if dimension <= 0 or heads <= 0 or dimension % heads:
        raise ValueError("invalid model dimensions")
    state = functional.rms_norm(model.transformer.wte(token_matrix), (dimension,))
    first_state = state
    block = model.transformer.h[0]
    state = block.lambdas[0] * state + block.lambdas[1] * first_state
    values = block.attn.c_v(functional.rms_norm(state, (dimension,)))
    if tuple(values.shape) != (*token_matrix.shape, dimension):
        raise ValueError("block-0 value projection shape changed")
    return values.view(*token_matrix.shape, heads, dimension // heads)
