"""Exact checkpoint contractions for causally admitted head and MLP pieces."""

# BQGATE: LIBRARY
from __future__ import annotations

import torch

import normalized_weight_reader_contract as normalized_reader


class CausalCheckpointTranslationError(ValueError):
    pass


def _finite_tensor(name, tensor, *, minimum_rank=1):
    if (not isinstance(tensor, torch.Tensor) or tensor.ndim < minimum_rank
            or not tensor.is_floating_point()):
        raise CausalCheckpointTranslationError(
            f"{name} must be a floating tensor with rank at least {minimum_rank}")
    if not bool(torch.isfinite(tensor).all()):
        raise CausalCheckpointTranslationError(f"{name} must be finite")


def _linear_write(delta, weight, *, name):
    """Apply a PyTorch-Linear checkpoint weight `[out, in]` to a delta."""
    _finite_tensor(name, delta)
    _finite_tensor("weight", weight, minimum_rank=2)
    if weight.ndim != 2 or delta.shape[-1] != weight.shape[1]:
        raise CausalCheckpointTranslationError(
            f"{name} trailing width must equal weight input width")
    return torch.matmul(delta.float(), weight.float().transpose(0, 1))


def _orthonormal_basis(name, basis, *, ambient_width):
    """Validate a column-orthonormal physical subspace basis `[ambient, rank]`."""
    _finite_tensor(name, basis, minimum_rank=2)
    if basis.ndim != 2 or basis.shape[0] != ambient_width or basis.shape[1] < 1:
        raise CausalCheckpointTranslationError(
            f"{name} must have shape [ambient_width, positive_rank]")
    candidate = basis.float()
    gram = candidate.transpose(0, 1) @ candidate
    identity = torch.eye(candidate.shape[1], dtype=candidate.dtype,
                         device=candidate.device)
    if not torch.allclose(gram, identity, atol=1e-5, rtol=1e-5):
        raise CausalCheckpointTranslationError(
            f"{name} columns must be orthonormal")
    return candidate


def restricted_writer_operator(writer_weight, output_basis):
    """Fold a causal output subspace into a weight-only writer `[rank, input]`.

    `writer_weight` has PyTorch Linear orientation `[residual, input]`. This is
    the exact `U_out^T W_write` operator and opens no activation dataset.
    """
    _finite_tensor("writer_weight", writer_weight, minimum_rank=2)
    if writer_weight.ndim != 2:
        raise CausalCheckpointTranslationError("writer_weight must be rank two")
    basis = _orthonormal_basis(
        "output_basis", output_basis, ambient_width=writer_weight.shape[0]
    )
    return basis.transpose(0, 1) @ writer_weight.float()


def restricted_reader_operator(reader_weight, input_basis):
    """Fold a causal input subspace into a weight-only reader `[output, rank]`."""
    _finite_tensor("reader_weight", reader_weight, minimum_rank=2)
    if reader_weight.ndim != 2:
        raise CausalCheckpointTranslationError("reader_weight must be rank two")
    basis = _orthonormal_basis(
        "input_basis", input_basis, ambient_width=reader_weight.shape[1]
    )
    return reader_weight.float() @ basis


def restricted_bilinear_core(left_weight, right_weight, down_weight,
                             input_basis, output_basis):
    """Return the exact weight-only bilinear core `T[out_rank,in_rank,in_rank]`.

    All three weights use PyTorch Linear orientation. The returned tensor obeys
    `y_U[a] = sum_bc T[a,b,c] x_U[b] x_U[c]` for inputs inside `input_basis`.
    It describes algebraically possible computation; empirical reachability is
    deliberately handled by `reachable_subspace_coordinates` after this core is fixed.
    """
    for name, tensor in (("left_weight", left_weight),
                         ("right_weight", right_weight),
                         ("down_weight", down_weight)):
        _finite_tensor(name, tensor, minimum_rank=2)
        if tensor.ndim != 2:
            raise CausalCheckpointTranslationError(f"{name} must be rank two")
    if left_weight.shape != right_weight.shape:
        raise CausalCheckpointTranslationError(
            "left_weight and right_weight must share [hidden, residual] shape")
    hidden, residual = left_weight.shape
    if down_weight.shape != (residual, hidden):
        raise CausalCheckpointTranslationError(
            "down_weight must have shape [residual, hidden]")
    input_u = _orthonormal_basis(
        "input_basis", input_basis, ambient_width=residual
    )
    output_u = _orthonormal_basis(
        "output_basis", output_basis, ambient_width=residual
    )
    down_u = output_u.transpose(0, 1) @ down_weight.float()
    left_u = left_weight.float() @ input_u
    right_u = right_weight.float() @ input_u
    return torch.einsum("ai,ib,ic->abc", down_u, left_u, right_u)


def restricted_qk_core(query_weight, key_weight, input_basis):
    """Return the exact QK bilinear form available inside an input subspace."""
    _finite_tensor("query_weight", query_weight, minimum_rank=2)
    _finite_tensor("key_weight", key_weight, minimum_rank=2)
    if query_weight.ndim != 2 or query_weight.shape != key_weight.shape:
        raise CausalCheckpointTranslationError(
            "query_weight and key_weight must share [head_width, residual] shape")
    basis = _orthonormal_basis(
        "input_basis", input_basis, ambient_width=query_weight.shape[1]
    )
    query_u, key_u = query_weight.float() @ basis, key_weight.float() @ basis
    return query_u.transpose(0, 1) @ key_u


def restricted_ov_core(value_weight, output_weight, input_basis, output_basis):
    """Return the exact value-to-output map between fixed causal subspaces.

    This is the OV content map for a fixed attention coefficient. QK routing is
    represented separately by `restricted_qk_core` and must not be folded into it.
    """
    _finite_tensor("value_weight", value_weight, minimum_rank=2)
    _finite_tensor("output_weight", output_weight, minimum_rank=2)
    if (value_weight.ndim != 2 or output_weight.ndim != 2
            or output_weight.shape[1] != value_weight.shape[0]):
        raise CausalCheckpointTranslationError(
            "value/output weights must have [head,residual] and [residual,head] shapes")
    input_u = _orthonormal_basis(
        "input_basis", input_basis, ambient_width=value_weight.shape[1]
    )
    output_u = _orthonormal_basis(
        "output_basis", output_basis, ambient_width=output_weight.shape[0]
    )
    return (output_u.transpose(0, 1) @ output_weight.float()
            @ value_weight.float() @ input_u)


def reachable_subspace_coordinates(states, basis):
    """Project observed states after a weight-defined subspace has been frozen.

    Unlike the functions above this is explicitly dataset-dependent: it estimates
    which part of the possible operator domain is visited, without redefining the
    operator or selecting a basis from the same observations.
    """
    _finite_tensor("states", states)
    frozen_basis = _orthonormal_basis(
        "basis", basis, ambient_width=states.shape[-1]
    )
    return states.float() @ frozen_basis


def optimal_shared_context_subspace(head_maps, *, rank, component_weights=None):
    """Solve the common-context projector problem exactly for a fixed rank.

    A rank-three tensor `head_maps[h, context, private]`, or a nonempty sequence of
    matrices `[context, private_h]`, may use an independent orthogonal gauge and a
    different private width for every component. The top left singular vectors of the horizontal unfolding
    minimize `sum_h ||M_h - U U^T M_h||_F^2` over column-orthonormal `U` of the
    requested rank. Optional strictly positive `component_weights` make the fitting
    norm explicit rather than letting heterogeneous reader widths silently choose it.
    The function fixes no rank or weights and makes no causal claim; it returns
    the common context basis, private adapters `U^T M_h`, and residual tails needed
    for a prospective common-core-versus-tail intervention.
    """
    tensor_input = isinstance(head_maps, torch.Tensor)
    if tensor_input:
        _finite_tensor("head_maps", head_maps, minimum_rank=3)
        if head_maps.ndim != 3 or head_maps.shape[0] < 1:
            raise CausalCheckpointTranslationError(
                "head_maps must have shape [positive_components, context, private]")
        maps = tuple(value.float() for value in head_maps.unbind(0))
    else:
        if not isinstance(head_maps, (list, tuple)) or not head_maps:
            raise CausalCheckpointTranslationError(
                "head_maps must be a rank-three tensor or nonempty matrix sequence")
        maps = tuple(head_maps)
        for index, value in enumerate(maps):
            _finite_tensor(f"head_maps[{index}]", value, minimum_rank=2)
            if value.ndim != 2 or value.shape[1] < 1:
                raise CausalCheckpointTranslationError(
                    "each context map must have shape [context, positive_private]")
        context_widths = {int(value.shape[0]) for value in maps}
        if len(context_widths) != 1:
            raise CausalCheckpointTranslationError(
                "all context maps must share their context width")
        maps = tuple(value.float() for value in maps)
    context_width = maps[0].shape[0]
    if not isinstance(rank, int) or rank < 1 or rank > context_width:
        raise CausalCheckpointTranslationError(
            "rank must be a positive integer no larger than context width")
    if component_weights is None:
        weights = maps[0].new_ones(len(maps))
    else:
        try:
            weights = torch.as_tensor(component_weights, dtype=maps[0].dtype,
                                      device=maps[0].device)
        except (TypeError, ValueError, RuntimeError) as weight_error:
            raise CausalCheckpointTranslationError(
                "component_weights must be finite positive scalars") from weight_error
        if (weights.ndim != 1 or weights.numel() != len(maps)
                or not bool(torch.isfinite(weights).all())
                or not bool((weights > 0).all())):
            raise CausalCheckpointTranslationError(
                "component_weights must have one finite positive value per map")
    try:
        unfolding = torch.cat(tuple(value * weights[index].sqrt()
                                    for index, value in enumerate(maps)), dim=1)
    except RuntimeError as concatenation_error:
        raise CausalCheckpointTranslationError(
            "all context maps must share dtype-compatible device placement") from concatenation_error
    if rank > min(unfolding.shape):
        raise CausalCheckpointTranslationError(
            "rank cannot exceed the smaller unfolding dimension")
    left, singular_values, _ = torch.linalg.svd(unfolding, full_matrices=False)
    basis = left[:, :rank]
    adapter_parts = tuple(basis.transpose(0, 1) @ value for value in maps)
    common_parts = tuple(basis @ value for value in adapter_parts)
    tail_parts = tuple(value - common for value, common in zip(maps, common_parts))
    squared_error = sum(weights[index] * value.square().sum()
                        for index, value in enumerate(tail_parts))
    component_energy = tuple(value.square().sum() for value in maps)
    component_squared_error = tuple(value.square().sum() for value in tail_parts)
    component_relative_squared_error = tuple(
        error / energy.clamp_min(1e-30)
        for error, energy in zip(component_squared_error, component_energy)
    )
    certified_optimum = singular_values[rank:].square().sum()
    total = sum(weights[index] * value.square().sum()
                for index, value in enumerate(maps)).clamp_min(1e-30)
    next_value = (singular_values[rank] if rank < singular_values.numel()
                  else singular_values.new_zeros(()))
    gap = singular_values[rank - 1] - next_value
    adapters = torch.stack(adapter_parts) if tensor_input else adapter_parts
    common = torch.stack(common_parts) if tensor_input else common_parts
    tails = torch.stack(tail_parts) if tensor_input else tail_parts
    return {
        "basis": basis,
        "private_adapters": adapters,
        "common_maps": common,
        "private_tails": tails,
        "singular_values": singular_values,
        "relative_squared_error": squared_error / total,
        "component_energy": component_energy,
        "component_squared_error": component_squared_error,
        "component_relative_squared_error": component_relative_squared_error,
        "component_captured_energy_fraction": tuple(
            1.0 - value for value in component_relative_squared_error
        ),
        "optimal_squared_error_certificate": certified_optimum,
        "certificate_absolute_error": (squared_error - certified_optimum).abs(),
        "boundary_spectral_gap": gap,
        "private_widths": tuple(int(value.shape[1]) for value in maps),
        "component_weights": weights,
    }


def shared_context_leave_one_out(head_maps, *, rank, component_weights=None):
    """Test whether a fixed-rank common context transfers to every omitted component.

    Each fold learns the context projector from all but one map and scores the omitted
    map by its unweighted captured Frobenius-energy fraction. Component weights affect
    only the training objective. This is a zero-forward stability diagnostic: callers
    must still freeze the rank/weights prospectively and test causal interchange.
    """
    if isinstance(head_maps, torch.Tensor):
        _finite_tensor("head_maps", head_maps, minimum_rank=3)
        if head_maps.ndim != 3:
            raise CausalCheckpointTranslationError(
                "head_maps must be a rank-three tensor or matrix sequence")
        maps = tuple(head_maps.unbind(0))
    elif isinstance(head_maps, (list, tuple)):
        maps = tuple(head_maps)
    else:
        raise CausalCheckpointTranslationError(
            "head_maps must be a rank-three tensor or matrix sequence")
    if len(maps) < 2:
        raise CausalCheckpointTranslationError(
            "leave-one-out transfer requires at least two component maps")

    pooled = optimal_shared_context_subspace(
        maps, rank=rank, component_weights=component_weights
    )
    weights = pooled["component_weights"]
    folds = []
    for heldout_index, heldout in enumerate(maps):
        training_maps = tuple(
            value for index, value in enumerate(maps) if index != heldout_index
        )
        training_weights = torch.stack(tuple(
            weights[index] for index in range(len(maps)) if index != heldout_index
        ))
        training = optimal_shared_context_subspace(
            training_maps, rank=rank, component_weights=training_weights
        )
        heldout_value = heldout.float()
        basis = training["basis"]
        tail = heldout_value - basis @ (basis.transpose(0, 1) @ heldout_value)
        energy = heldout_value.square().sum()
        squared_error = tail.square().sum()
        relative_error = squared_error / energy.clamp_min(1e-30)
        folds.append({
            "heldout_index": heldout_index,
            "basis": basis,
            "heldout_energy": energy,
            "heldout_squared_error": squared_error,
            "heldout_relative_squared_error": relative_error,
            "heldout_captured_energy_fraction": 1.0 - relative_error,
            "training_optimal_squared_error_certificate":
                training["optimal_squared_error_certificate"],
            "training_certificate_absolute_error":
                training["certificate_absolute_error"],
        })
    captured = torch.stack(tuple(
        fold["heldout_captured_energy_fraction"] for fold in folds
    ))
    return {
        "rank": rank,
        "component_weights": weights,
        "folds": tuple(folds),
        "mean_heldout_captured_energy_fraction": captured.mean(),
        "worst_heldout_captured_energy_fraction": captured.min(),
        "pooled_report": pooled,
    }


def attention_head_write(head_delta, c_proj_weight, *, head, num_heads):
    """Translate one pre-`c_proj` head delta into its physical residual write.

    `c_proj_weight` follows `torch.nn.Linear` orientation `[residual, heads*width]`.
    Leading batch/token dimensions on `head_delta` are preserved.
    """
    _finite_tensor("head_delta", head_delta)
    _finite_tensor("c_proj_weight", c_proj_weight, minimum_rank=2)
    if c_proj_weight.ndim != 2 or not isinstance(num_heads, int) or num_heads < 1:
        raise CausalCheckpointTranslationError(
            "c_proj_weight must be rank two and num_heads must be positive")
    if not isinstance(head, int) or head < 0 or head >= num_heads:
        raise CausalCheckpointTranslationError("head index is out of range")
    input_width = int(c_proj_weight.shape[1])
    if input_width % num_heads:
        raise CausalCheckpointTranslationError(
            "c_proj input width is not divisible by num_heads")
    head_width = input_width // num_heads
    if head_delta.shape[-1] != head_width:
        raise CausalCheckpointTranslationError(
            "head delta width does not match the selected c_proj slice")
    start = head * head_width
    weight_slice = c_proj_weight[:, start:start + head_width]
    return _linear_write(head_delta, weight_slice, name="head_delta")


def bilinear_product_factors(live_left, live_right, writer_left, writer_right):
    """Return the exact arm-local Left, Right, and interaction product deltas."""
    tensors = (live_left, live_right, writer_left, writer_right)
    for name, tensor in zip(
        ("live_left", "live_right", "writer_left", "writer_right"), tensors
    ):
        _finite_tensor(name, tensor)
    shapes = {tuple(tensor.shape) for tensor in tensors}
    if len(shapes) != 1:
        raise CausalCheckpointTranslationError(
            "live and writer Left/Right tensors must share one shape")
    left0, right0, left1, right1 = (tensor.float() for tensor in tensors)
    delta_left = left1 - left0
    delta_right = right1 - right0
    return {
        "left": delta_left * right0,
        "right": left0 * delta_right,
        "interaction": delta_left * delta_right,
    }


def mlp_factor_write(factor, down_weight):
    """Translate one product-factor delta through `Down.weight [residual, hidden]`."""
    return _linear_write(factor, down_weight, name="factor")


def mlp_factor_writes(live_left, live_right, writer_left, writer_right, down_weight):
    """Return all three exact residual-write tensors and their exact sum."""
    factors = bilinear_product_factors(
        live_left, live_right, writer_left, writer_right
    )
    writes = {name: mlp_factor_write(value, down_weight)
              for name, value in factors.items()}
    writes["all_three"] = sum(writes.values())
    return writes


def normalized_reader_output(residual_state, residual_write, reader_weight, *, eps=None):
    """Return the exact finite RMSNorm-aware change seen by a checkpoint reader.

    RMSNorm is evaluated in the state tensor's deployed dtype. The resulting finite
    input secant and checkpoint reader weight are contracted in float32. This is a
    weight-based reader nomination; causal reader-side interchange remains a separate
    identification test.
    """
    _finite_tensor("residual_state", residual_state, minimum_rank=2)
    _finite_tensor("residual_write", residual_write, minimum_rank=2)
    _finite_tensor("reader_weight", reader_weight, minimum_rank=2)
    if residual_state.shape != residual_write.shape:
        raise CausalCheckpointTranslationError(
            "residual state and write must share one shape")
    if reader_weight.ndim != 2 or reader_weight.shape[1] != residual_state.shape[-1]:
        raise CausalCheckpointTranslationError(
            "reader weight must have shape [output, residual_width]")
    if eps is None:
        eps = torch.finfo(residual_state.dtype).eps
    try:
        epsilon = float(eps)
    except (TypeError, ValueError):
        raise CausalCheckpointTranslationError("RMSNorm epsilon must be finite")
    if not torch.isfinite(torch.tensor(epsilon)) or epsilon < 0:
        raise CausalCheckpointTranslationError(
            "RMSNorm epsilon must be finite and nonnegative")
    normalized0 = torch.nn.functional.rms_norm(
        residual_state, (residual_state.shape[-1],), eps=epsilon
    )
    normalized1 = torch.nn.functional.rms_norm(
        residual_state + residual_write,
        (residual_state.shape[-1],), eps=epsilon,
    )
    return _linear_write(
        normalized1.float() - normalized0.float(), reader_weight,
        name="normalized_input_delta",
    )


def normalized_reader_report(residual_state, residual_write, reader_weight, *, eps=None):
    """Reuse the canonical raw/tangent/exact reader diagnostic on any leading shape."""
    _finite_tensor("residual_state", residual_state, minimum_rank=2)
    _finite_tensor("residual_write", residual_write, minimum_rank=2)
    _finite_tensor("reader_weight", reader_weight, minimum_rank=2)
    if residual_state.shape != residual_write.shape:
        raise CausalCheckpointTranslationError(
            "residual state and write must share one shape")
    if reader_weight.ndim != 2 or reader_weight.shape[1] != residual_state.shape[-1]:
        raise CausalCheckpointTranslationError(
            "reader weight must have shape [output, residual_width]")
    epsilon = torch.finfo(residual_state.dtype).eps if eps is None else eps
    flat_state = residual_state.reshape(-1, residual_state.shape[-1]).float()
    flat_write = residual_write.reshape(-1, residual_write.shape[-1]).float()
    try:
        return normalized_reader.reader_response(
            torch, reader_weight.float(), flat_state, flat_write, eps=epsilon
        )
    except (TypeError, ValueError, RuntimeError):
        raise CausalCheckpointTranslationError(
            "normalized reader diagnostic failed")


def reader_response_match(candidate, reference):
    """Measure whether two task writes are operationally equivalent to one reader."""
    _finite_tensor("candidate", candidate)
    _finite_tensor("reference", reference)
    if candidate.shape != reference.shape:
        raise CausalCheckpointTranslationError(
            "candidate and reference reader responses must share one shape")
    candidate, reference = candidate.float(), reference.float()
    reference_norm = reference.norm().clamp_min(1e-30)
    candidate_norm = candidate.norm()
    cosine = float(
        (candidate.reshape(-1) @ reference.reshape(-1))
        / (candidate_norm * reference_norm).clamp_min(1e-30)
    )
    return {
        "cosine": cosine,
        "relative_l2": float((candidate - reference).norm() / reference_norm),
        "norm_ratio": float(candidate_norm / reference_norm),
        "sign_agreement": float(
            ((candidate == 0) & (reference == 0)
             | (candidate.sign() == reference.sign())).float().mean()
        ),
    }
