"""Deeper bilinear models for the depth-ladder study (session 5): interleave bilinear
attention with bilinear MLP layers, keeping the whole model polynomial (tensor-friendly).

Bilinear MLP (user's spec): y = D( (L x) ⊙ (R x) ) — a gated linear unit with NO
elementwise nonlinearity, so it stays a polynomial in x (a genuine tensor/bilinear form),
unlike GeLU/ReLU MLPs. Wrapped in a residual like the attention block.

DeepModel takes a layer SPEC — a list of "attn"/"mlp" — so we can build:
    ["attn", "attn"]           2-layer attn-only baseline (≈ induction depth)
    ["attn", "mlp", "attn"]    2 attn with 1 bilinear MLP in the middle
    ["attn", "attn", "attn"]   3-layer attn-only
and compare which token categories need the extra depth.

RMSNorm is left OFF by default: standard RMSNorm divides by an input-dependent norm,
which is not polynomial, so it would break the tensor property. A tensor-compatible norm
(to be supplied) can be slotted into `make_norm`; until then norm=False keeps the model a
clean polynomial. The reference bilinear-attention block from model.py is reused verbatim.
"""

import torch
from torch import nn

from model import Attention


def make_norm(d_model: int, norm) -> nn.Module:
    """norm=False -> Identity (polynomial). norm='rms' -> affine-free RMSNorm (NOT
    polynomial; use only when the tensor-norm variant is not required)."""
    if norm in (False, None, "none"):
        return nn.Identity()
    if norm in (True, "rms"):
        return nn.RMSNorm(d_model, elementwise_affine=False)
    raise ValueError(f"unknown norm {norm!r}")


class BilinearMLP(nn.Module):
    """y = D( (L x) ⊙ (R x) ), residual-wrapped. Polynomial (degree-2) in x."""

    def __init__(self, d_model: int, d_hidden: int | None = None, scale: float = 0.5,
                 residual: str = "add", norm=False) -> None:
        super().__init__()
        d_hidden = d_hidden or 4 * d_model
        self.residual = residual
        self.scale = scale
        self.norm = make_norm(d_model, norm)
        self.L = nn.Linear(d_model, d_hidden, bias=False)
        self.R = nn.Linear(d_model, d_hidden, bias=False)
        self.D = nn.Linear(d_hidden, d_model, bias=False)

    def forward(self, x):
        h = self.norm(x)
        y = self.D(self.L(h) * self.R(h))
        return x + y if self.residual == "add" else torch.lerp(x, y, self.scale)


class DeepModel(nn.Module):
    """Embed -> [attn|mlp]* -> unembed, per a layer spec. No layernorm by default."""

    def __init__(self, n_vocab: int, d_model: int, n_head: int, spec, n_ctx: int,
                 d_hidden: int | None = None, scale: float = 0.5, norm=False,
                 residual: str = "lerp", attention: str = "bilinear",
                 mlp_residual: str = "add") -> None:
        super().__init__()
        self.spec = list(spec)
        self.embed = nn.Embedding(n_vocab, d_model)
        layers = []
        for kind in self.spec:
            if kind == "attn":
                layers.append(Attention(d_model, n_head, n_ctx, scale=scale, norm=norm,
                                        residual=residual, attention=attention))
            elif kind == "mlp":
                layers.append(BilinearMLP(d_model, d_hidden, scale=scale,
                                          residual=mlp_residual, norm=norm))
            else:
                raise ValueError(f"unknown layer kind {kind!r}")
        self.layers = nn.ModuleList(layers)
        self.head = nn.Linear(d_model, n_vocab, bias=False)

    def forward(self, tokens):
        x = self.embed(tokens)
        for layer in self.layers:
            x = layer(x)
        return self.head(x)

    def residuals(self, tokens):
        x, stream = self.embed(tokens), []
        for layer in self.layers:
            x = layer(x)
            stream.append(x)
        return stream

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


SPECS = {
    "attn2": ["attn", "attn"],
    "attn-mlp-attn": ["attn", "mlp", "attn"],
    "attn3": ["attn", "attn", "attn"],
}
