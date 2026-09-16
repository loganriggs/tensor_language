"""Execute the M4 factor graph from four residual corners and frozen Q/K weights."""
import torch
import torch.nn.functional as F

try:
    from extracted_circuits.equality_l5h5_m4_raw_port_factor_graph_v1 import node as raw_node
except ModuleNotFoundError:
    from basis_aligned.polynomial_causal.extracted_circuits.equality_l5h5_m4_raw_port_factor_graph_v1 import node as raw_node

NODE_NAMES = raw_node.NODE_NAMES


def _raw_ports(residual: torch.Tensor, weights, head_index: int,
               head_width: int):
    if len(weights) != 4:
        raise ValueError("weights must contain Q1,K1,Q2,K2")
    state = F.rms_norm(residual, (residual.shape[-1],))
    lo, hi = head_index * head_width, (head_index + 1) * head_width
    return [F.linear(state, weight)[..., lo:hi] for weight in weights]


def decompose(baseline_residual: torch.Tensor, child_residual: torch.Tensor,
              remainder_residual: torch.Tensor, joint_residual: torch.Tensor,
              weights, cos: torch.Tensor, sin: torch.Tensor,
              head_index: int, head_width: int, return_ports: bool = False):
    """Project residual corners and construct the no-oracle factor graph."""
    residuals = (baseline_residual, child_residual, remainder_residual,
                 joint_residual)
    if len({tuple(value.shape) for value in residuals}) != 1:
        raise ValueError("all residual ports must have matching shapes")
    baseline_raw = _raw_ports(baseline_residual, weights, head_index, head_width)
    child_raw = _raw_ports(child_residual, weights, head_index, head_width)
    remainder_raw = _raw_ports(remainder_residual, weights, head_index, head_width)
    joint_raw = _raw_ports(joint_residual, weights, head_index, head_width)
    denominator = lambda value: (value.float().square().mean(
        dim=-1, keepdim=True) + torch.finfo(value.dtype).eps).sqrt()
    child_denominator = denominator(child_residual)
    derived = []
    for index in range(4):
        value = joint_raw[index].float() * (denominator(joint_residual) /
                                            child_denominator)
        value = value - remainder_raw[index].float() * (
            denominator(remainder_residual) / child_denominator)
        value = value + baseline_raw[index].float() * (
            denominator(baseline_residual) / child_denominator)
        derived.append(value.to(child_residual.dtype))
    graph = raw_node.decompose(derived, child_raw, cos, sin)
    if return_ports:
        return graph, {"baseline": baseline_raw, "child": child_raw,
                       "remainder": remainder_raw, "joint": joint_raw,
                       "derived": derived}
    return graph


compose = raw_node.compose
remove_node = raw_node.remove_node
replace_with_rolled_arithmetic = raw_node.replace_with_rolled_arithmetic
