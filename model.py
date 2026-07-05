"""Attention-only transformer with bilinear attention.

The Attention module is adapted from tdooms/tensor-similarity
(src/components/attention.py): scores are a *product of two dot products*
(q1·k1)(q2·k2) / d_head², passed through a multiplicative causal mask —
no softmax anywhere, so the whole model is a polynomial in its inputs.
Positions enter only through rotary embeddings on q1/k1/q2/k2.
"""

import torch
from torch import nn
from einops import rearrange, einsum


class Rotary(nn.Module):
    """Rotary position encoding (verbatim from the reference repo)."""

    def __init__(self, dim: int, n_ctx: int, base: int = 10_000) -> None:
        super().__init__()
        freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        ctx = torch.arange(n_ctx).type_as(freq)
        freqs = torch.einsum("i,j->ij", ctx, freq)

        cos, sin = freqs.cos(), freqs.sin()
        self.cos_cached = nn.Buffer(torch.cat([cos, cos], dim=-1)[None, :, None, :], persistent=False)
        self.sin_cached = nn.Buffer(torch.cat([sin, sin], dim=-1)[None, :, None, :], persistent=False)

    def forward(self, x):
        a, b = x.chunk(2, dim=-1)
        y = torch.cat((-b, a), dim=-1)
        return (x * self.cos_cached[:, : x.size(-3)]) + (y * self.sin_cached[:, : x.size(-3)])


class Attention(nn.Module):
    """Bilinear (quadratic-scoring) attention with a lerp residual.

    scale=0 is the identity, scale=1 replaces the stream entirely;
    scale=0.5 gives an even residual mix.
    """

    def __init__(self, d_model: int, n_head: int, n_ctx: int, scale: float = 0.5,
                 norm: bool = False, residual: str = "lerp", attention: str = "bilinear") -> None:
        super().__init__()
        self.d_head = d_model // n_head
        self.n_head = n_head
        self.scale = scale
        self.residual = residual
        self.attention = attention

        self.norm = nn.RMSNorm(d_model, elementwise_affine=False) if norm else nn.Identity()
        self.rotary = Rotary(self.d_head, n_ctx)
        self.mask = nn.Buffer(torch.tril(torch.ones(n_ctx, n_ctx)), persistent=False)

        self.q1 = nn.Linear(d_model, d_model, bias=False)
        self.k1 = nn.Linear(d_model, d_model, bias=False)
        if attention == "bilinear":
            self.q2 = nn.Linear(d_model, d_model, bias=False)
            self.k2 = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)

    def pattern(self, x):
        h = self.norm(x)
        seq = x.size(-2)
        heads = "... (n_head d_head) -> ... n_head d_head"
        q1 = self.rotary(rearrange(self.q1(h), heads, n_head=self.n_head))
        k1 = self.rotary(rearrange(self.k1(h), heads, n_head=self.n_head))
        scores1 = einsum(q1, k1, "... q n_head h, ... k n_head h -> ... n_head q k")
        if self.attention == "softmax":
            scores1 = scores1 / self.d_head**0.5
            return scores1.masked_fill(self.mask[:seq, :seq] == 0, -torch.inf).softmax(-1)
        q2 = self.rotary(rearrange(self.q2(h), heads, n_head=self.n_head))
        k2 = self.rotary(rearrange(self.k2(h), heads, n_head=self.n_head))
        scores2 = einsum(q2, k2, "... q n_head h, ... k n_head h -> ... n_head q k")
        return (scores1 * scores2) / self.d_head**2 * self.mask[:seq, :seq]

    def forward(self, x):
        v = rearrange(self.v(self.norm(x)), "... (n_head d_head) -> ... n_head d_head", n_head=self.n_head)
        z = einsum(self.pattern(x), v, "... n_head q k, ... k n_head h -> ... q n_head h")
        z = rearrange(z, "... seq n_head h -> ... seq (n_head h)")
        return x + self.o(z) if self.residual == "add" else torch.lerp(x, self.o(z), self.scale)


class CycleModel(nn.Module):
    """Embed → n_layer bilinear attention layers → unembed. No MLPs, no norms."""

    def __init__(self, n_vocab: int, d_model: int, n_head: int, n_layer: int, n_ctx: int, scale: float = 0.5,
                 norm: bool = False, residual: str = "lerp", attention: str = "bilinear") -> None:
        super().__init__()
        self.embed = nn.Embedding(n_vocab, d_model)
        self.layers = nn.ModuleList([Attention(d_model, n_head, n_ctx, scale=scale, norm=norm,
                                               residual=residual, attention=attention) for _ in range(n_layer)])
        self.head = nn.Linear(d_model, n_vocab, bias=False)

    def forward(self, tokens):
        x = self.embed(tokens)
        for layer in self.layers:
            x = layer(x)
        return self.head(x)

    def residuals(self, tokens):
        """Residual stream after the embedding and after each layer."""
        x, stream = self.embed(tokens), []
        for layer in self.layers:
            x = layer(x)
            stream.append(x)
        return stream

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def config_kwargs(config: dict) -> dict:
    """Architecture kwargs recorded in a run's history config (values may be str-encoded)."""
    kwargs = {k: config[k] for k in ("scale", "norm", "residual", "attention") if k in config}
    if "norm" in kwargs:
        kwargs["norm"] = kwargs["norm"] in (True, "True", "true")
    if "scale" in kwargs:
        kwargs["scale"] = float(kwargs["scale"])
    return kwargs
