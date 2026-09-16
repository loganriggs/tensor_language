"""Five-write canonical boundary graph for the approximate L5H5 equality score."""
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
        raw = F.linear(state, weight.to(device=state.device, dtype=state.dtype))
        ports.append(_rotate(F.rms_norm(raw, (raw.shape[-1],)), cos, sin))
    q1, k1, q2, k2 = ports
    first = torch.einsum("bqd,bkd->bqk", q1, k1) / q1.shape[-1]
    second = torch.einsum("bqd,bkd->bqk", q2, k2) / q2.shape[-1]
    score = first * second
    causal = torch.ones(score.shape[-2:], dtype=torch.bool, device=score.device).tril()
    return score.masked_fill(~causal, 0)


def execute(mlp2_write: torch.Tensor, attention3_write: torch.Tensor,
            mlp3_write: torch.Tensor, attention4_write: torch.Tensor,
            mlp4_write: torch.Tensor, weights, cos: torch.Tensor,
            sin: torch.Tensor) -> torch.Tensor:
    shapes = {tuple(value.shape) for value in (mlp2_write, attention3_write,
                                               mlp3_write, attention4_write,
                                               mlp4_write)}
    if len(shapes) != 1:
        raise ValueError("all five module-write ports must have matching shapes")
    layer2 = mlp2_write.float()
    layer3 = attention3_write.float() + mlp3_write.float()
    layer4 = attention4_write.float() + mlp4_write.float()
    residual = torch.zeros_like(layer2) + layer2 + layer3 + layer4
    return execute_residual(residual.to(mlp2_write.dtype), weights, cos, sin)
