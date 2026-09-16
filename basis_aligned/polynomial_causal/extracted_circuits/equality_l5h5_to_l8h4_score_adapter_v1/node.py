"""Frozen scalar adapter from the L5H5 equality score to the L8H4 port."""
import torch

SCORE_SCALE = 0.5371214943951729


def execute(l5h5_score: torch.Tensor) -> torch.Tensor:
    if l5h5_score.ndim != 3 or not l5h5_score.is_floating_point():
        raise ValueError("score port must be a floating [batch,query,key] tensor")
    return l5h5_score * SCORE_SCALE
