"""Construct four M4 residual corners and execute their precision factor graph."""
import torch
import torch.nn.functional as F

try:
    from extracted_circuits.equality_l5h5_m4_residual_port_factor_graph_v1 import node as residual_node
except ModuleNotFoundError:
    from basis_aligned.polynomial_causal.extracted_circuits.equality_l5h5_m4_residual_port_factor_graph_v1 import node as residual_node

NODE_NAMES = residual_node.NODE_NAMES


def _mode_write(product: torch.Tensor, down_bias: torch.Tensor,
                basis: torch.Tensor, writers: torch.Tensor,
                indices: torch.Tensor, residual_scale) -> torch.Tensor:
    selected_basis = basis.index_select(0, indices).to(product.device)
    selected_writers = writers.index_select(1, indices).to(product.device)
    if indices.numel() == 0:
        write = torch.zeros((*product.shape[:-1], writers.shape[0]),
                            device=product.device, dtype=torch.float32)
    else:
        coefficients = F.linear(product.float(), selected_basis.float())
        write = F.linear(coefficients, selected_writers.float())
    quantized = (write + down_bias.to(product.device).float()).to(product.dtype).float()
    return quantized * residual_scale


def _merge(m2: torch.Tensor, a3: torch.Tensor, m3: torch.Tensor,
           a4: torch.Tensor, m4: torch.Tensor, dtype: torch.dtype):
    layer2 = m2
    layer3 = a3 + m3
    layer4 = a4 + m4
    return (torch.zeros_like(layer2) + layer2 + layer3 + layer4).to(dtype)


def construct_residuals(product: torch.Tensor, m2: torch.Tensor,
                        a3: torch.Tensor, m3: torch.Tensor,
                        a4: torch.Tensor, down_bias: torch.Tensor,
                        basis: torch.Tensor, writers: torch.Tensor,
                        residual_scale, dtype: torch.dtype,
                        child_rank: int = 128, selected_rank: int = 256):
    """Build baseline, child, remainder, and joint residual corners."""
    if basis.shape[0] != writers.shape[1]:
        raise ValueError("basis/writer storage ranks are inconsistent")
    if not 0 < child_rank < selected_rank <= basis.shape[0]:
        raise ValueError("basis/writer rank and child split are inconsistent")
    modes = torch.arange(selected_rank, device=product.device)
    selections = {"baseline": modes[:0], "child": modes[:child_rank],
                  "remainder": modes[child_rank:], "joint": modes}
    residuals = {}
    for name, indices in selections.items():
        m4 = _mode_write(product, down_bias, basis, writers, indices,
                         residual_scale)
        residuals[name] = _merge(m2, a3, m3, a4, m4, dtype)
    return residuals


def decompose(product: torch.Tensor, m2: torch.Tensor, a3: torch.Tensor,
              m3: torch.Tensor, a4: torch.Tensor, down_bias: torch.Tensor,
              basis: torch.Tensor, writers: torch.Tensor, residual_scale,
              dtype: torch.dtype, qk_weights, cos: torch.Tensor,
              sin: torch.Tensor, head_index: int, head_width: int,
              child_rank: int = 128, selected_rank: int = 256,
              return_residuals: bool = False):
    residuals = construct_residuals(product, m2, a3, m3, a4, down_bias,
                                    basis, writers, residual_scale, dtype,
                                    child_rank, selected_rank)
    graph = residual_node.decompose(
        residuals["baseline"], residuals["child"], residuals["remainder"],
        residuals["joint"], qk_weights, cos, sin, head_index, head_width)
    return (graph, residuals) if return_residuals else graph


compose = residual_node.compose
remove_node = residual_node.remove_node
