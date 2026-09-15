#!/usr/bin/env python3
# BQGATE: LIBRARY
"""Shared no-model preflight helpers for exact self interventions and predicates."""
from __future__ import annotations

import ast


REQUIRED_RESULT_FIELDS = (
    "schema", "terminal", "predictions", "price", "runner_sha256", "binding_sha256"
)


def managed_execution_mode(environment) -> str:
    """Classify a hash-bound enqueue preflight or a queue-runner execution.

    The explicit marker prevents leaked BQLIB flags from turning a real queue run
    into a successful no-model dry run.
    """
    dry = environment.get("BQLIB_DRYRUN") == "1"
    no_model = environment.get("BQLIB_NO_MODEL") == "1"
    managed = environment.get("BQLIB_MANAGED_PREFLIGHT") == "1"
    if dry and no_model and managed:
        return "preflight"
    if not dry and not no_model and not managed:
        return "execute"
    raise RuntimeError(
        "ambiguous managed execution flags: expected all preflight flags or none"
    )


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


def validate_literal_prediction_registry(source: str, registry) -> None:
    """Require the source's literal pred_* keys to equal its declared registry."""
    literal = static_prediction_keys(source)
    declared = list(registry)
    if literal != declared:
        raise ValueError(
            f"literal prediction keys do not match registry: literal={literal} declared={declared}"
        )


def replace_query_native(full_tensor, positions, replacements):
    """Clone a full batch and replace one row per batch item in its native dtype."""
    if getattr(full_tensor, "ndim", None) is None or full_tensor.ndim < 2:
        raise ValueError("full tensor must have batch and sequence dimensions")
    if len(positions) != len(full_tensor) or len(replacements) != len(full_tensor):
        raise ValueError("one query position and replacement are required per batch row")
    output = full_tensor.clone()
    native = replacements.to(device=output.device, dtype=output.dtype)
    for row, position in enumerate(positions):
        output[row, int(position)] = native[row]
    return output


def clear_rotary_state(model) -> int:
    """Clear cached rotary tensors before an autograd-enabled forward."""
    cleared = 0
    for module in model.modules():
        if all(hasattr(module, name) for name in ("seq_len_cached", "cos_cached", "sin_cached")):
            module.seq_len_cached = None
            module.cos_cached = None
            module.sin_cached = None
            cleared += 1
    if not cleared:
        raise ValueError("model exposes no rotary cache state")
    return cleared


def bind_directed_answers(rows, endpoints):
    """Bind recipient endpoint records while taking answer IDs from directed rows."""
    endpoint_by_id = {}
    for endpoint in endpoints:
        endpoint_id = endpoint.get("endpoint_id")
        if endpoint_id is None or endpoint_id in endpoint_by_id:
            raise ValueError("endpoint IDs must be present and unique")
        endpoint_by_id[endpoint_id] = endpoint
    bound = []
    for row in rows:
        endpoint_id = row.get("recipient_endpoint_id")
        if endpoint_id not in endpoint_by_id:
            raise ValueError(f"unknown recipient endpoint: {endpoint_id}")
        if "recipient_answer_id" not in row:
            raise ValueError("directed row is missing recipient_answer_id")
        bound.append((endpoint_by_id[endpoint_id], int(row["recipient_answer_id"])))
    return bound


def validate_result_contract(result, prediction_registry) -> None:
    """Fail before publication if a managed result omits its review-critical fields."""
    missing = [field for field in REQUIRED_RESULT_FIELDS if field not in result]
    if missing:
        raise ValueError(f"result is missing required fields: {missing}")
    expected = list(prediction_registry)
    predictions = result["predictions"]
    if not isinstance(predictions, dict) or list(predictions) != expected:
        raise ValueError("result prediction keys do not match the registered literal order")
    if not all(type(value) is bool for value in predictions.values()):
        raise ValueError("every result prediction must be a bool")
    if not isinstance(result["price"], dict) or not result["price"]:
        raise ValueError("result price must be a nonempty mapping")
    for field in ("runner_sha256", "binding_sha256"):
        value = result[field]
        if not isinstance(value, str) or len(value) != 64 \
                or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"{field} must be a lowercase SHA-256 digest")


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
