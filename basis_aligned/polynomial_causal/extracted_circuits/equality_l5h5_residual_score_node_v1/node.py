"""L5H5 score executor from one residual-state port and frozen head weights."""
import torch
import torch.nn.functional as F


def project_head(state: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if state.ndim != 3 or weight.ndim != 2 or weight.shape[1] != state.shape[-1]:
        raise ValueError("state must be [batch,position,width] and weight [head_dim,width]")
    return F.linear(state, weight.to(device=state.device, dtype=state.dtype))


def rotate_head(value: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    if value.ndim != 3 or value.shape[-1] % 2 or cos.shape != sin.shape:
        raise ValueError("invalid rotary ports")
    half = value.shape[-1] // 2
    if cos.shape != (1, value.shape[1], half):
        raise ValueError("rotary ports must be [1,position,head_dim/2]")
    first, second = value[..., :half], value[..., half:]
    return torch.cat((first * cos + second * sin,
                      first * (-sin) + second * cos), dim=-1).type_as(value)


def qk_ports(residual: torch.Tensor, weights, cos: torch.Tensor, sin: torch.Tensor):
    if len(weights) != 4:
        raise ValueError("four frozen Q/K weight ports are required")
    state = F.rms_norm(residual, (residual.shape[-1],))
    raw = [project_head(state, weight) for weight in weights]
    return tuple(rotate_head(F.rms_norm(value, (value.shape[-1],)), cos, sin)
                 for value in raw)


def execute(residual: torch.Tensor, weights, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    q1, k1, q2, k2 = qk_ports(residual, weights, cos, sin)
    score1 = torch.einsum("bqd,bkd->bqk", q1, k1) / q1.shape[-1]
    score2 = torch.einsum("bqd,bkd->bqk", q2, k2) / q2.shape[-1]
    score = score1 * score2
    causal = torch.ones(score.shape[-2:], dtype=torch.bool, device=score.device).tril()
    return score.masked_fill(~causal, 0)


def merge_sources(sources, correction: torch.Tensor) -> torch.Tensor:
    """Merge additive FP32 provenance ports and their explicit BF16 correction."""
    if not sources:
        return correction
    return sum((source.float() for source in sources), start=torch.zeros_like(correction)) + correction
