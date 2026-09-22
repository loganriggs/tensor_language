"""Core primitives: mixed Hessians, participation ratio, and unit freezing.

Conventions
-----------
A *span function* has signature

    span(theta: Tensor[d_source], freeze: FreezeSpec | None) -> (out, hidden)

where `theta` is added to the residual stream at the source layer,
`out` is the (target-position-averaged) residual at the target layer, and
`hidden` is a dict {layer_index: Tensor[d_mlp]} of pre-down-projection MLP
activations (target-position-averaged) for every layer the span records.

Freezing a unit means replacing its activation with its *clean* value
(the value at theta = 0). At the infinitesimal level this is identical to
detaching it (its tangent becomes zero), so the same mechanism serves both
the mixed-Hessian completeness check (E1) and finite ablations (E3).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Set

import torch
from torch.func import jvp


@dataclass
class FreezeSpec:
    """Which units to hold at their clean values.

    mlp_masks:   {layer: bool/float Tensor[d_mlp]}, 1 = freeze that hidden unit.
    attn_layers: layers whose entire attention output is frozen.
    clean_hidden / clean_attn: cached clean activations {layer: Tensor}.
        If absent, freezing falls back to `.detach()`, which is only correct
        for derivatives taken at theta = 0 (the mixed-Hessian use case).
    """
    mlp_masks: Dict[int, torch.Tensor] = field(default_factory=dict)
    attn_layers: Set[int] = field(default_factory=set)
    clean_hidden: Optional[Dict[int, torch.Tensor]] = None
    clean_attn: Optional[Dict[int, torch.Tensor]] = None


def apply_freeze(h: torch.Tensor, mask: Optional[torch.Tensor],
                 clean: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Hold masked coordinates of h at their clean value (last dim is units).

    With clean=None this detaches the masked coordinates; the forward value is
    unchanged and the tangent is zero, which is what we want at theta = 0.
    """
    if mask is None:
        return h
    mask = mask.to(h.dtype)
    frozen = h.detach() if clean is None else clean.to(h.dtype).expand_as(h)
    return mask * frozen + (1 - mask) * h


def mixed_hessian(fn: Callable[[torch.Tensor], torch.Tensor],
                  l: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """d^2/(d alpha d beta) fn(alpha*l + beta*r) at alpha = beta = 0.

    fn maps a source-space vector to any tensor; the result has fn's output
    shape. Uses forward-over-forward mode, same as AJ's
    `directional_hessian_outputs`.
    """
    zero = torch.zeros_like(l)

    def first(base):
        return jvp(fn, (base,), (l,))[1]

    return jvp(first, (zero,), (r,))[1]


def finite_mixed_difference(fn: Callable[[torch.Tensor], torch.Tensor],
                            l: torch.Tensor, r: torch.Tensor,
                            alpha: float, beta: float) -> torch.Tensor:
    """fn(al+br) - fn(al) - fn(br) + fn(0): the finite-size interaction term.

    Divided by alpha*beta it converges to mixed_hessian as alpha, beta -> 0.
    """
    zero = torch.zeros_like(l)
    return (fn(alpha * l + beta * r) - fn(alpha * l) - fn(beta * r) + fn(zero))


def participation_ratio(c: torch.Tensor, eps: float = 1e-30) -> torch.Tensor:
    """(sum c^2)^2 / sum c^4 over the last dim. Matches AJ's definition.

    Scale-invariant: PR says how *concentrated* a response is, not how *large*.
    That is exactly why E1 (completeness) is needed alongside it.
    """
    e = c.float().square()
    return e.sum(-1).square() / e.square().sum(-1).clamp_min(eps)


def concat_hidden(hidden: Dict[int, torch.Tensor], layers) -> torch.Tensor:
    return torch.cat([hidden[k] for k in layers], dim=-1)
