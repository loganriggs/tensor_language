"""Reversible BF16 graph wrapper for the exact-order equality edge."""
import torch
import torch.nn.functional as F


def execute(score: torch.Tensor, raw_payload: torch.Tensor, support: torch.Tensor, output_weight: torch.Tensor) -> torch.Tensor:
    if score.ndim != 3 or raw_payload.ndim != 3 or support.shape != score.shape:
        raise ValueError("equality node port shape mismatch")
    if score.shape[0] != raw_payload.shape[0] or score.shape[2] != raw_payload.shape[1]:
        raise ValueError("equality node key/batch dimensions mismatch")
    if output_weight.ndim != 2 or output_weight.shape[1] != raw_payload.shape[2]:
        raise ValueError("equality node output-weight shape mismatch")
    dtype = raw_payload.dtype
    contracted = torch.bmm(score.to(dtype) * support.to(dtype), raw_payload)
    return F.linear(contracted, output_weight.to(device=contracted.device, dtype=dtype))


def split(full_write: torch.Tensor, term: torch.Tensor):
    """Remove a term and retain only the FP32 correction needed to invert BF16."""
    if full_write.shape != term.shape or full_write.dtype != term.dtype:
        raise ValueError("split operands must have matching shape and dtype")
    removed = full_write - term
    rounded_reassembly = removed + term
    correction = full_write.float() - rounded_reassembly.float()
    return removed, correction


def merge(removed: torch.Tensor, term: torch.Tensor, correction: torch.Tensor):
    """Reassemble the exact BF16 write from semantic and arithmetic ports."""
    if removed.shape != term.shape or correction.shape != removed.shape:
        raise ValueError("merge operands must have matching shapes")
    rounded_reassembly = removed + term.to(removed.dtype)
    return (rounded_reassembly.float() + correction.float()).to(removed.dtype)
