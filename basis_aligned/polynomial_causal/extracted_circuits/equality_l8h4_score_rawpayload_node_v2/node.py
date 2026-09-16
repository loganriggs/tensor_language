"""Exact-order equality edge: contract raw values, then output-project."""
import torch
import torch.nn.functional as F


def execute(score: torch.Tensor, raw_payload: torch.Tensor, support: torch.Tensor, output_weight: torch.Tensor) -> torch.Tensor:
    """Execute the equality edge in the deployed arithmetic order."""
    if score.ndim != 3 or raw_payload.ndim != 3 or support.shape != score.shape:
        raise ValueError("equality node port shape mismatch")
    if score.shape[0] != raw_payload.shape[0] or score.shape[2] != raw_payload.shape[1]:
        raise ValueError("equality node key/batch dimensions mismatch")
    if output_weight.ndim != 2 or output_weight.shape[1] != raw_payload.shape[2]:
        raise ValueError("equality node output-weight shape mismatch")
    dtype = raw_payload.dtype
    contracted = torch.bmm(score.to(dtype) * support.to(dtype), raw_payload)
    return F.linear(contracted, output_weight.to(device=contracted.device, dtype=dtype))
