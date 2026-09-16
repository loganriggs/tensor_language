"""Zero-parameter equality-edge executor with explicit score/payload ports."""
import torch


def execute(score: torch.Tensor, payload: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    """Return the equality-supported score/payload contraction.

    score:   [batch, query, key]
    payload: [batch, key, residual]
    support: [batch, query, key], boolean or numeric
    """
    if score.ndim != 3 or payload.ndim != 3 or support.shape != score.shape:
        raise ValueError("equality node port shape mismatch")
    if score.shape[0] != payload.shape[0] or score.shape[2] != payload.shape[1]:
        raise ValueError("equality node key/batch dimensions mismatch")
    return torch.bmm(score * support.to(score.dtype), payload.to(score.dtype))
