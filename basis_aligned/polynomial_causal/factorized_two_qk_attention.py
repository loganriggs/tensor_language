"""Exact conditional attention with shared branch factors instead of pair tensors.

The conditional interface matches projected_two_qk_attention. Storage avoids
T*T*r*r score coefficients; execution still performs quadratic attention.
Backgrounds and cached first-layer values remain charged dependencies.
"""
import torch
from projected_two_qk_attention import _rotate_base


def compile_attention(weights, background, P, Q, first_value, mixture, heads,
                      cos, sin, input_eps, head_eps):
    batch, length, width = background.shape
    hd = width // heads
    branches = {
        name: dict(base=(background @ weights[name].T).reshape(batch, length, heads, hd),
                   basis=(weights[name] @ P).reshape(heads, hd, P.shape[1]))
        for name in ('q', 'k', 'q2', 'k2')
    }
    output = torch.einsum('da,dhk->hak', Q, weights['o'].reshape(width, heads, hd))
    raw_value = (background @ weights['v'].T).reshape(batch, length, heads, hd)
    value_basis = (weights['v'] @ P).reshape(heads, hd, P.shape[1])
    return dict(format='factorized_two_qk_v1', branches=branches,
        global_norm=dict(base=background.square().mean(-1) + input_eps,
                         linear=2 * (background @ P) / width, gram=(P.T @ P) / width),
        value_base=(1-mixture)*torch.einsum('bthk,hak->btha', raw_value, output),
        value_basis=(1-mixture)*torch.einsum('hkp,hak->hap', value_basis, output),
        cached_value=mixture*torch.einsum('bthk,hak->btha', first_value, output),
        cos=cos.clone(), sin=sin.clone(), head_eps=head_eps, head_width=hd, length=length)


def execute(program, z):
    g = program['global_norm']
    squared_scale = (g['base'] + (g['linear']*z).sum(-1)
                     + torch.einsum('pq,btp,btq->bt', g['gram'], z, z))
    def branch(name):
        b = program['branches'][name]
        raw = b['base'] + torch.einsum('hkp,btp->bthk', b['basis'], z)
        denominator = (raw.square().mean(-1)
                       + program['head_eps']*squared_scale[..., None]).sqrt()
        return _rotate_base(raw / denominator[..., None], program['cos'], program['sin'])
    def score(q, k):
        return torch.einsum('bthk,bshk->bhts', branch(q), branch(k)) / program['head_width']
    pattern = score('q', 'k') * score('q2', 'k2')
    mask = torch.ones(program['length'], program['length'], device=z.device, dtype=torch.bool).tril()
    pattern = pattern.masked_fill(~mask, 0)
    value = (program['value_base'] + torch.einsum('hap,btp->btha', program['value_basis'], z))
    value = value / squared_scale.sqrt()[:, :, None, None] + program['cached_value']
    return torch.einsum('bhts,bsha->bta', pattern, value)
