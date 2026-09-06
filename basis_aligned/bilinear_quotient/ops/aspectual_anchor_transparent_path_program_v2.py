#!/usr/bin/env python3
"""Executable tensor equations for the paired aspectual-anchor causal path.

The functions are framework-agnostic: inputs may be PyTorch tensors, NumPy
arrays, or another array type implementing indexing and arithmetic. They do not
load a model, fit parameters, or perform a forward pass.
"""

from __future__ import annotations


PROGRAM_ID = "aspectual_anchor.has_vs_had.transparent_path_program_v2"
ATTENTION5_HEADS = (7, 1, 6, 8)
ATTENTION9_HEADS = (1, 4)
SOURCE_NAMES = ("last", "period", "determiner")
CROSSING_BOUNDARIES = (6, 7, 8, 9)


class ProgramInputError(ValueError):
    """A typed circuit-program input violates the frozen interface."""


def mlp4_hidden_response(left_base, right_base, left_donor, right_donor):
    """Return the frozen two-term hidden response, excluding the interaction."""
    left_change = (left_donor - left_base) * right_base
    right_change = left_base * (right_donor - right_base)
    return left_change + right_change


def linear_without_bias(hidden_response, down_weight):
    """Apply the checkpoint MLP4 Down matrix with no bias term."""
    return hidden_response @ down_weight.transpose(-1, -2)


def mlp4_write(left_base, right_base, left_donor, right_donor, down_weight):
    """Project the frozen two-term MLP4 response into residual width."""
    return linear_without_bias(
        mlp4_hidden_response(left_base, right_base, left_donor, right_donor),
        down_weight,
    )


def attention_source_term(pattern, effective_value, head: int, query: int, source: int):
    """Return one pattern[q,source,head] times effective-value[source,head] term."""
    if min(head, query, source) < 0:
        raise ProgramInputError("head, query, and source indices must be nonnegative")
    return pattern[head, query, source] * effective_value[source, head]


def attention_source_delta(
    base_pattern,
    base_effective_value,
    hybrid_pattern,
    hybrid_effective_value,
    *,
    query: int,
    source_positions: tuple[int, int, int],
    heads: tuple[int, ...],
):
    """Return per-head hybrid-minus-base transport from the frozen source bank."""
    if len(source_positions) != len(SOURCE_NAMES):
        raise ProgramInputError("source_positions must be (last, period, determiner)")
    if len(set(source_positions)) != len(source_positions):
        raise ProgramInputError("source positions must be distinct")
    if tuple(heads) not in (ATTENTION5_HEADS, ATTENTION9_HEADS):
        raise ProgramInputError("heads must be the frozen attention5 or attention9 tuple")
    values = []
    for head in heads:
        delta = None
        for source in source_positions:
            term = attention_source_term(
                hybrid_pattern, hybrid_effective_value, head, query, source
            ) - attention_source_term(
                base_pattern, base_effective_value, head, query, source
            )
            delta = term if delta is None else delta + term
        values.append(delta)
    return tuple(values)


def crossing_delta(
    lambda0,
    base_resid,
    hybrid_resid,
    base_attention,
    hybrid_attention,
    base_mlp,
    hybrid_mlp,
    *,
    factors: tuple[str, ...] = ("carried", "attention", "mlp"),
):
    """Compose a frozen carried/attention/MLP residual-boundary delta."""
    allowed = ("carried", "attention", "mlp")
    if len(set(factors)) != len(factors) or any(factor not in allowed for factor in factors):
        raise ProgramInputError("factors must be a unique subset of carried/attention/mlp")
    delta = None
    terms = {
        "carried": lambda0 * (hybrid_resid - base_resid),
        "attention": hybrid_attention - base_attention,
        "mlp": hybrid_mlp - base_mlp,
    }
    for factor in allowed:
        if factor in factors:
            delta = terms[factor] if delta is None else delta + terms[factor]
    if delta is None:
        return base_resid - base_resid
    return delta


def write_query_delta(state, query: int, delta):
    """Clone an array-like residual state and add a circuit delta at one query."""
    if query < 0:
        raise ProgramInputError("query must be nonnegative")
    if hasattr(state, "clone"):
        changed = state.clone()
    elif hasattr(state, "copy"):
        changed = state.copy()
    else:
        raise ProgramInputError("state must provide clone() or copy()")
    changed[query] = changed[query] + delta
    return changed


def program_manifest() -> dict[str, object]:
    """Return the exact stable inventory exposed by this executable module."""
    return {
        "program_id": PROGRAM_ID,
        "mlp4_terms": ("left_change", "right_change"),
        "attention5_heads": ATTENTION5_HEADS,
        "attention9_heads": ATTENTION9_HEADS,
        "source_names": SOURCE_NAMES,
        "crossing_boundaries": CROSSING_BOUNDARIES,
        "stored_fit_scalars": 0,
        "stored_fit_vectors": 0,
        "runtime_dependencies": (
            "checkpoint weights",
            "paired base/donor MLP4 states",
            "paired base/hybrid attention captures",
            "native checkpoint suffix",
        ),
    }
