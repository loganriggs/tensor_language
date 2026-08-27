"""Exact cross-layer gauge induced by bilin18's shared layer-0 value bus."""

from __future__ import annotations

import torch


def value_output_action(value_maps, output_maps, local_inputs, shared_input, mixing):
    """Evaluate one head's value/output contribution at every layer.

    Token routing acts to the left of the feature axis and therefore commutes with
    the feature-basis action represented here. ``local_inputs`` may already include
    any fixed token-routing operator.
    """
    if not (len(value_maps) == len(output_maps) == len(local_inputs) == len(mixing)):
        raise ValueError("one value, output, input, and mixing coefficient per layer")
    shared = value_maps[0] @ shared_input
    return [output @ ((1-alpha)*(value @ local) + alpha*shared)
            for value, output, local, alpha in
            zip(value_maps, output_maps, local_inputs, mixing)]


def apply_shared_gauge(value_maps, output_maps, gauge):
    """Apply one GL(head_dim) basis change to a head at every depth."""
    inverse = torch.linalg.inv(gauge)
    return ([gauge @ value for value in value_maps],
            [output @ inverse for output in output_maps])


def canonical_row_basis(matrix, relative_tolerance=1e-9):
    """Deterministic orthonormal row basis depending only on matrix rowspace.

    Project coordinate axes into the rowspace in lexical coordinate order and use
    the first independent projections. This is a generic-stratum canonical section:
    it deliberately rejects rank loss or a tolerance-boundary pivot.
    """
    matrix = matrix.detach().double().cpu()
    if matrix.ndim != 2 or matrix.shape[0] > matrix.shape[1]:
        raise ValueError("expected a wide full-row-rank matrix")
    singular = torch.linalg.svdvals(matrix)
    if singular.numel() == 0 or singular[-1] <= relative_tolerance*singular[0]:
        raise ValueError("shared value map is not generically full row rank")
    gram = matrix @ matrix.T
    projector = matrix.T @ torch.linalg.solve(gram, matrix)
    columns = []
    absolute_tolerance = relative_tolerance
    for coordinate in range(matrix.shape[1]):
        candidate = projector[:, coordinate].clone()
        for basis in columns:
            candidate -= torch.dot(basis, candidate)*basis
        norm = torch.linalg.vector_norm(candidate)
        if norm > absolute_tolerance:
            candidate /= norm
            first = torch.nonzero(candidate.abs() > absolute_tolerance)
            if first.numel() and candidate[first[0, 0]] < 0:
                candidate = -candidate
            columns.append(candidate)
            if len(columns) == matrix.shape[0]:
                break
    if len(columns) != matrix.shape[0]:
        raise ValueError("canonical pivot sequence is numerically degenerate")
    return torch.stack(columns, dim=1).T.contiguous()


def canonicalize_shared_value_bus(value_maps, output_maps, relative_tolerance=1e-9):
    """Return a representative invariant to the exact shared GL(head_dim) gauge."""
    if not value_maps or len(value_maps) != len(output_maps):
        raise ValueError("matched nonempty value/output map sequences required")
    values = [value.detach().double().cpu() for value in value_maps]
    outputs = [output.detach().double().cpu() for output in output_maps]
    shared = values[0]
    basis = canonical_row_basis(shared, relative_tolerance)
    # The unique left action taking full-row-rank shared to the canonical basis.
    gauge = basis @ shared.T @ torch.linalg.inv(shared @ shared.T)
    canonical_values, canonical_outputs = apply_shared_gauge(values, outputs, gauge)
    torch.testing.assert_close(canonical_values[0], basis, atol=2e-8, rtol=2e-8)
    return canonical_values, canonical_outputs


def generic_parameter_dimension(layers, model_dimension, head_dimension, heads=1):
    """Behavioral dimension before discrete common-head permutation quotienting."""
    if min(layers, model_dimension, head_dimension, heads) <= 0:
        raise ValueError("dimensions must be positive")
    if head_dimension > model_dimension:
        raise ValueError("full-row-rank shared values require head_dim <= model_dim")
    raw = layers*2*model_dimension*head_dimension*heads
    gauge = head_dimension*head_dimension*heads
    return {"raw_parameters": raw, "continuous_gauge_dimension": gauge,
            "quotient_dimension": raw-gauge,
            "incorrect_independent_layer_quotient_dimension":
                raw-layers*gauge}
