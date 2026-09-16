"""Möbius interaction wrapper for baseline, child, remainder, and joint scores."""
import torch


def decompose(baseline: torch.Tensor, child: torch.Tensor,
              remainder: torch.Tensor, joint: torch.Tensor):
    shapes = {tuple(value.shape) for value in (baseline, child, remainder, joint)}
    if len(shapes) != 1:
        raise ValueError("all four score ports must have matching shapes")
    child_effect = child - baseline
    remainder_effect = remainder - baseline
    interaction = joint - child - remainder + baseline
    return {"baseline": baseline, "child_effect": child_effect,
            "remainder_effect": remainder_effect,
            "interaction": interaction}


def compose(nodes) -> torch.Tensor:
    """Canonical graph arithmetic used by the validated executor."""
    additive = (nodes["baseline"] + nodes["child_effect"]) + nodes["remainder_effect"]
    return additive + nodes["interaction"]


def remove_interaction(nodes) -> torch.Tensor:
    return (nodes["baseline"] + nodes["child_effect"]) + nodes["remainder_effect"]


def replace_with_rolled_interaction(nodes, shifts: int = 1) -> torch.Tensor:
    additive = remove_interaction(nodes)
    return additive + torch.roll(nodes["interaction"], shifts=shifts, dims=1)
