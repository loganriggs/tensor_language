"""Four-node factor graph preserving native finite-precision child scores."""
import torch

NODE_NAMES = ("first", "second", "cross", "arithmetic")


def _causal(value: torch.Tensor) -> torch.Tensor:
    mask = torch.ones(value.shape[-2:], dtype=torch.bool, device=value.device).tril()
    return value.masked_fill(~mask, 0)


def decompose(first_derived: torch.Tensor, first_native: torch.Tensor,
              second_derived: torch.Tensor, second_native: torch.Tensor,
              native_child_score: torch.Tensor):
    """Build exact nodes from the two derived/native QK-factor pairs."""
    values = (first_derived, first_native, second_derived, second_native,
              native_child_score)
    if len({tuple(value.shape) for value in values}) != 1:
        raise ValueError("all factor and score ports must have matching shapes")
    base = _causal(first_derived * second_derived)
    first = _causal((first_native - first_derived) * second_derived)
    second = _causal(first_derived * (second_native - second_derived))
    cross = _causal((first_native - first_derived) *
                    (second_native - second_derived))
    algebraic = ((base + first) + second) + cross
    arithmetic = native_child_score - algebraic
    return {"base": base, "first": first, "second": second,
            "cross": cross, "arithmetic": arithmetic}


def compose(nodes) -> torch.Tensor:
    return (((nodes["base"] + nodes["first"]) + nodes["second"]) +
            nodes["cross"]) + nodes["arithmetic"]


def remove_node(nodes, name: str) -> torch.Tensor:
    if name not in NODE_NAMES:
        raise ValueError(f"unknown removable node: {name}")
    return compose(nodes) - nodes[name]


def replace_with_rolled_arithmetic(nodes, shifts: int = 1) -> torch.Tensor:
    return remove_node(nodes, "arithmetic") + torch.roll(
        nodes["arithmetic"], shifts=shifts, dims=1)
