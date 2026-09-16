"""Correction-free L2+L3+L4 boundary graph for the L5H5 equality score."""
import torch
import torch.nn.functional as F


def _rotate(value: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = value.shape[-1] // 2
    first, second = value[..., :half], value[..., half:]
    return torch.cat((first * cos + second * sin,
                      first * (-sin) + second * cos), dim=-1).type_as(value)


def execute_residual(residual: torch.Tensor, weights, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    if residual.ndim != 3 or len(weights) != 4:
        raise ValueError("one residual and four Q/K head-weight ports are required")
    state = F.rms_norm(residual, (residual.shape[-1],))
    ports = []
    for weight in weights:
        if weight.ndim != 2 or weight.shape[1] != residual.shape[-1]:
            raise ValueError("each weight must be [head_dim,residual_width]")
        raw = F.linear(state, weight.to(device=state.device, dtype=state.dtype))
        ports.append(_rotate(F.rms_norm(raw, (raw.shape[-1],)), cos, sin))
    q1, k1, q2, k2 = ports
    first = torch.einsum("bqd,bkd->bqk", q1, k1) / q1.shape[-1]
    second = torch.einsum("bqd,bkd->bqk", q2, k2) / q2.shape[-1]
    score = first * second
    causal = torch.ones(score.shape[-2:], dtype=torch.bool, device=score.device).tril()
    return score.masked_fill(~causal, 0)


def execute(layer2_write: torch.Tensor, layer3_write: torch.Tensor,
            layer4_write: torch.Tensor, weights, cos: torch.Tensor,
            sin: torch.Tensor) -> torch.Tensor:
    if layer2_write.shape != layer3_write.shape or layer2_write.shape != layer4_write.shape:
        raise ValueError("L2/L3/L4 write ports must have matching shapes")
    residual = layer2_write.float() + layer3_write.float() + layer4_write.float()
    return execute_residual(residual.to(layer2_write.dtype), weights, cos, sin)
