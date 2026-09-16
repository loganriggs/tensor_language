"""Execute the exact factor graph directly from derived/native raw Q/K ports."""
import torch
import torch.nn.functional as F

NODE_NAMES = ("first", "second", "cross", "arithmetic")


def _rotate(value: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    half = value.shape[-1] // 2
    first, second = value[..., :half], value[..., half:]
    return torch.cat((first * cos + second * sin,
                      first * (-sin) + second * cos), dim=-1).type_as(value)


def _transform(raw: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    return _rotate(F.rms_norm(raw, (raw.shape[-1],)), cos, sin)


def _dot(query: torch.Tensor, key: torch.Tensor):
    return torch.einsum("bqd,bkd->bqk", query, key) / query.shape[-1]


def _causal(value: torch.Tensor):
    mask = torch.ones(value.shape[-2:], dtype=torch.bool, device=value.device).tril()
    return value.masked_fill(~mask, 0)


def decompose(derived_raw_ports, native_raw_ports, cos: torch.Tensor,
              sin: torch.Tensor):
    """Return exact nodes without accepting an oracle child-score input."""
    if len(derived_raw_ports) != 4 or len(native_raw_ports) != 4:
        raise ValueError("derived and native port collections must each contain Q1,K1,Q2,K2")
    shapes = {tuple(value.shape) for value in (*derived_raw_ports, *native_raw_ports)}
    if len(shapes) != 1:
        raise ValueError("all raw Q/K ports must have matching shapes")
    derived = [_transform(value, cos, sin) for value in derived_raw_ports]
    native = [_transform(value, cos, sin) for value in native_raw_ports]

    native_first = _dot(native[0], native[1])
    native_second = _dot(native[2], native[3])
    native_child_score = _causal(native_first * native_second).float()

    first_derived = _dot(derived[0].float(), derived[1].float())
    first_native = _dot(native[0].float(), native[1].float())
    second_derived = _dot(derived[2].float(), derived[3].float())
    second_native = _dot(native[2].float(), native[3].float())
    base = _causal(first_derived * second_derived)
    first = _causal((first_native - first_derived) * second_derived)
    second = _causal(first_derived * (second_native - second_derived))
    cross = _causal((first_native - first_derived) *
                    (second_native - second_derived))
    algebraic = ((base + first) + second) + cross
    arithmetic = native_child_score - algebraic
    return {"base": base, "first": first, "second": second,
            "cross": cross, "arithmetic": arithmetic}


def compose(nodes):
    return (((nodes["base"] + nodes["first"]) + nodes["second"]) +
            nodes["cross"]) + nodes["arithmetic"]


def remove_node(nodes, name: str):
    if name not in NODE_NAMES:
        raise ValueError(f"unknown removable node: {name}")
    return compose(nodes) - nodes[name]


def replace_with_rolled_arithmetic(nodes, shifts: int = 1):
    return remove_node(nodes, "arithmetic") + torch.roll(
        nodes["arithmetic"], shifts=shifts, dims=1)
