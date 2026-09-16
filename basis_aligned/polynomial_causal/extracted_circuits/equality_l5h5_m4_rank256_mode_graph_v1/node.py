"""Weight-only rank-256 contracted-mode executor for the M4 equality-score port."""
import torch
import torch.nn.functional as F

try:
    from extracted_circuits.equality_l5h5_residual_score_node_v1 import node as score_node
except ModuleNotFoundError:
    from basis_aligned.polynomial_causal.extracted_circuits.equality_l5h5_residual_score_node_v1 import node as score_node


def compile_program(qk_weights, down_weight: torch.Tensor, rank: int = 256):
    """Compile right-singular product modes from four downstream Q/K readers."""
    if len(qk_weights) != 4 or down_weight.ndim != 2:
        raise ValueError("four Q/K matrices and one M4 Down matrix are required")
    stacked = torch.cat([weight.double() for weight in qk_weights], dim=0)
    contracted = stacked @ down_weight.double()
    left, values, right = torch.linalg.svd(contracted, full_matrices=False)
    if not 0 < rank <= right.shape[0]:
        raise ValueError("rank must be positive and no larger than contracted rank")
    reconstruction_error = ((left * values.unsqueeze(0)) @ right - contracted).norm() / contracted.norm().clamp_min(1e-30)
    basis = right[:rank].float()
    writers = down_weight.float() @ basis.T
    return {"basis": basis, "writers": writers, "contracted_reconstruction_relative_l2": float(reconstruction_error)}


def execute_m4(normalized_input: torch.Tensor, left_weight: torch.Tensor,
               right_weight: torch.Tensor, down_bias: torch.Tensor,
               program, residual_scale=1.0) -> torch.Tensor:
    """Execute dense scalar bilinear modes; all native L/R products remain live."""
    if normalized_input.ndim != 3 or left_weight.shape != right_weight.shape:
        raise ValueError("normalized M4 input and matching L/R matrices are required")
    product = F.linear(normalized_input, left_weight.to(normalized_input.dtype)) * F.linear(normalized_input, right_weight.to(normalized_input.dtype))
    basis = program["basis"].to(device=product.device, dtype=torch.float32)
    writers = program["writers"].to(device=product.device, dtype=torch.float32)
    coefficients = F.linear(product.float(), basis)
    write = F.linear(coefficients, writers) + down_bias.to(device=product.device, dtype=torch.float32)
    return write.to(normalized_input.dtype).float() * residual_scale


def execute_score(mlp2_write: torch.Tensor, attention3_write: torch.Tensor,
                  mlp3_write: torch.Tensor, attention4_write: torch.Tensor,
                  normalized_mlp4_input: torch.Tensor, left_weight: torch.Tensor,
                  right_weight: torch.Tensor, down_bias: torch.Tensor, program,
                  residual_scale, qk_weights, cos: torch.Tensor,
                  sin: torch.Tensor) -> torch.Tensor:
    m4_write = execute_m4(normalized_mlp4_input, left_weight, right_weight,
                          down_bias, program, residual_scale)
    layer2 = mlp2_write.float()
    layer3 = attention3_write.float() + mlp3_write.float()
    layer4 = attention4_write.float() + m4_write
    residual = torch.zeros_like(layer2) + layer2 + layer3 + layer4
    return score_node.execute(residual.to(mlp2_write.dtype), qk_weights, cos, sin)
