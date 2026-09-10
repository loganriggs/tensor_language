"""Exact bilinear finite differences and output-folded scalar context readers."""
import torch
import torch.nn.functional as F


def folded_delta(left, right, down, e, u, delta):
    le, re = left @ e, right @ e
    lu, ru = F.linear(u, left), F.linear(u, right)
    product = delta[:, None] * (le * ru + lu * re)
    product = product + delta[:, None].square() * le * re
    return F.linear(product, down)


def context_reader(left, right, down, e, v):
    c = v @ down
    return left.T @ (c * (right @ e)) + right.T @ (c * (left @ e))


def controls():
    rng = torch.Generator().manual_seed(9111420)
    def rand(*shape):
        return torch.randn(*shape, generator=rng, dtype=torch.float64)
    l, r, d, u, e, v, delta = rand(9, 5), rand(9, 5), rand(5, 9), rand(7, 5), rand(5), rand(5), rand(7)
    e = e / e.norm()
    f = lambda x: ((x @ l.T) * (x @ r.T)) @ d.T
    exact = f(u + delta[:, None] * e) - f(u)
    calculated = folded_delta(l, r, d, e, u, delta)
    k = context_reader(l, r, d, e, v)
    scalar = delta * (u @ k) + delta.square() * ((v @ d) * (l @ e) * (r @ e)).sum()
    error = float((exact - calculated).abs().max())
    reader_error = float((scalar - calculated @ v).abs().max())
    assert error <= 1e-10 and reader_error <= 1e-10
    return {'delta_max_abs': error, 'reader_max_abs': reader_error, 'gpu_accessed': False}
