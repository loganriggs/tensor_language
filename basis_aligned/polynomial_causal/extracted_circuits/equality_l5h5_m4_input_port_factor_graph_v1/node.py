"""Execute the M4 precision factor graph from normalized MLP4 input."""
import torch
import torch.nn.functional as F

try:
    from extracted_circuits.equality_l5h5_m4_product_port_factor_graph_v1 import node as product_node
except ModuleNotFoundError:
    from basis_aligned.polynomial_causal.extracted_circuits.equality_l5h5_m4_product_port_factor_graph_v1 import node as product_node

NODE_NAMES = product_node.NODE_NAMES


def construct_product(normalized_state: torch.Tensor, left_weight: torch.Tensor,
                      right_weight: torch.Tensor) -> torch.Tensor:
    """Match the native bias-free CastedLinear Left/Right product."""
    left = F.linear(normalized_state, left_weight.to(normalized_state.dtype))
    right = F.linear(normalized_state, right_weight.to(normalized_state.dtype))
    return left * right


def construct_residuals(normalized_state: torch.Tensor, m2: torch.Tensor,
                        a3: torch.Tensor, m3: torch.Tensor,
                        a4: torch.Tensor, left_weight: torch.Tensor,
                        right_weight: torch.Tensor, down_bias: torch.Tensor,
                        basis: torch.Tensor, writers: torch.Tensor,
                        residual_scale, dtype: torch.dtype,
                        child_rank: int = 128, selected_rank: int = 256):
    product = construct_product(normalized_state, left_weight, right_weight)
    return product_node.construct_residuals(
        product, m2, a3, m3, a4, down_bias, basis, writers,
        residual_scale, dtype, child_rank, selected_rank,
    )


def decompose(normalized_state: torch.Tensor, m2: torch.Tensor,
              a3: torch.Tensor, m3: torch.Tensor, a4: torch.Tensor,
              left_weight: torch.Tensor, right_weight: torch.Tensor,
              down_bias: torch.Tensor, basis: torch.Tensor,
              writers: torch.Tensor, residual_scale, dtype: torch.dtype,
              qk_weights, cos: torch.Tensor, sin: torch.Tensor,
              head_index: int, head_width: int, child_rank: int = 128,
              selected_rank: int = 256, return_intermediates: bool = False):
    product = construct_product(normalized_state, left_weight, right_weight)
    graph, residuals = product_node.decompose(
        product, m2, a3, m3, a4, down_bias, basis, writers,
        residual_scale, dtype, qk_weights, cos, sin, head_index, head_width,
        child_rank, selected_rank, return_residuals=True,
    )
    return (graph, product, residuals) if return_intermediates else graph


compose = product_node.compose
remove_node = product_node.remove_node
