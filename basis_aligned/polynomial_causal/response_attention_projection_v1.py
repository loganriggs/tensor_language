"""Share all five attention projections across a fixed three-vector edit family."""
import torch
from raw_attention_response_v1 import EPS


def prepare(raw,basis,matrices):
    return dict(raw=raw,basis=basis,baseline=tuple(raw@m.T for m in matrices),
                projected_basis=tuple(basis@m.T for m in matrices))


def changed(coefficients,context):
    projections=tuple(p+torch.einsum('...k,...kd->...d',coefficients,b)
        for p,b in zip(context['baseline'],context['projected_basis']))
    delta=torch.einsum('...k,...kd->...d',coefficients,context['basis'])
    rho=(context['raw']+delta).square().mean(-1,keepdim=True)+EPS
    return projections,rho
