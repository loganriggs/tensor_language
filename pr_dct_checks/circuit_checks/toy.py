"""A norm-free, attention-free residual stack of bilinear MLPs.

Used to validate the checks before touching the real model, and for the exact
weight-space computation in E5. Each block:  z <- z + D_k [(A_k z) * (B_k z)].

Because there are no norms, everything is an exact polynomial in the input:
- one block: the mixed Hessian along (l, r) is D[(A l)*(B r) + (A r)*(B l)],
  independent of x (a pure weight object);
- two blocks: it is a degree-2 polynomial in x, so its data average depends on
  the data only through E[x] and E[x x^T] (see expected_mixed_hessian_2layer).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch

from .core import FreezeSpec, apply_freeze


@dataclass
class BilinearStack:
    A: List[torch.Tensor]  # [d_mlp, d]
    B: List[torch.Tensor]  # [d_mlp, d]
    D: List[torch.Tensor]  # [d, d_mlp]

    @property
    def n_layers(self):
        return len(self.A)

    @staticmethod
    def random(d=16, d_mlp=32, n_layers=3, scale=0.3, seed=0):
        g = torch.Generator().manual_seed(seed)
        mk = lambda *s: torch.randn(*s, generator=g) * scale / s[-1] ** 0.5
        return BilinearStack(
            A=[mk(d_mlp, d) for _ in range(n_layers)],
            B=[mk(d_mlp, d) for _ in range(n_layers)],
            D=[mk(d, d_mlp) for _ in range(n_layers)],
        )

    def span_fn(self, x: torch.Tensor):
        """Return span(theta, freeze) -> (out, hidden) for context x."""
        def span(theta, freeze: Optional[FreezeSpec] = None):
            z = x + theta
            hidden = {}
            for k in range(self.n_layers):
                h = (self.A[k] @ z) * (self.B[k] @ z)
                if freeze is not None and k in freeze.mlp_masks:
                    clean = None if freeze.clean_hidden is None else freeze.clean_hidden[k]
                    h = apply_freeze(h, freeze.mlp_masks[k], clean)
                hidden[k] = h
                z = z + self.D[k] @ h
            return z, hidden
        return span

    def mlp_input_fn(self, x: torch.Tensor, layer: int):
        """theta -> residual entering block `layer` (the MLP input; no norm here)."""
        span_prefix = BilinearStack(self.A[:layer], self.B[:layer], self.D[:layer])
        return lambda theta: span_prefix.span_fn(x)(theta)[0]


def single_layer_mixed_hessian(A, B, D, l, r):
    """Closed form for one block: D[(A l)*(B r) + (A r)*(B l)]. No x anywhere."""
    return D @ ((A @ l) * (B @ r) + (A @ r) * (B @ l))


def expected_mixed_hessian_2layer(stack: BilinearStack, mu, Sigma, l, r):
    """Exact E_x[ d^2 out / d alpha d beta ] for a 2-block stack.

    Only E[x] = mu and Cov[x] = Sigma enter. This is the 'weight-space' version
    of the DCT objective for this architecture: weights contracted with input
    moments, no per-datapoint forward passes.
    """
    A1, B1, D1 = stack.A[0], stack.B[0], stack.D[0]
    A2, B2, D2 = stack.A[1], stack.B[1], stack.D[1]
    M2 = Sigma + torch.outer(mu, mu)                    # E[x x^T]

    # Layer 1 response is constant in x.
    z_lr = single_layer_mixed_hessian(A1, B1, D1, l, r)

    # z_v = dz/d(direction v) = v + D1[(A1 x)*(B1 v) + (A1 v)*(B1 x)] is affine in x:
    # z_v = c_v + G_v x
    def affine(v):
        c = v
        G = D1 @ ((B1 @ v)[:, None] * A1 + (A1 @ v)[:, None] * B1)
        return c, G

    cl, Gl = affine(l)
    cr, Gr = affine(r)
    # z = x + D1[(A1 x)*(B1 x)]  (quadratic in x)

    def E_prod(p, q, Gp, Gq, cp, cq):
        """E[(p.(cp+Gp x)) (q.(cq+Gq x))] for fixed row vectors p, q (batched over units)."""
        pc = p @ cp; qc = q @ cq
        pG = p @ Gp; qG = q @ Gq                     # [units, d]
        return (pc * qc + pc * (qG @ mu) + qc * (pG @ mu)
                + torch.einsum('ui,ij,uj->u', pG, M2, qG))

    # Term 1: (a.z_l)(b.z_r) + (a.z_r)(b.z_l)
    t1 = E_prod(A2, B2, Gl, Gr, cl, cr) + E_prod(A2, B2, Gr, Gl, cr, cl)
    # Term 2: (a.z)(b.z_lr) + (a.z_lr)(b.z);  E[z] = mu + D1 E[(A1 x)*(B1 x)]
    Ez = mu + D1 @ torch.einsum('ui,ij,uj->u', A1, M2, B1)
    t2 = (A2 @ Ez) * (B2 @ z_lr) + (A2 @ z_lr) * (B2 @ Ez)
    return z_lr + D2 @ (t1 + t2)


def plant_circuit(stack: BilinearStack, layer: int, unit: int,
                  l: torch.Tensor, r: torch.Tensor, u: torch.Tensor, gain: float):
    """Make `unit` in `layer` read (l, r) and write u with the given gain."""
    stack.A[layer][unit] = l * gain ** 0.5
    stack.B[layer][unit] = r * gain ** 0.5
    stack.D[layer][:, unit] = u
    return stack
