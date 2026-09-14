"""Exact bidegree(2,2) MLP9 response when mixed strength equals a*b.

The generic three-source program uses ten quadratic coefficients. Coupling its
third input to a*b makes two monomials identical, so their coefficients add.
No term is discarded. This is a constrained intervention interface.
"""
import torch
MAP=[0,1,2,3,4,3,5,6,7,8]


def prepare_for_suffix(postattention_state,source_changes,left,right,down,bias):
    """Compile added routing/value/mixed writes at the native block9 boundary."""
    from sro_mlp9_rational_v1 import prepare
    changes=[source_changes[k] for k in ['routing','values','mixed']]
    if any(x.shape!=postattention_state.shape for x in changes):
        raise ValueError('Each source write must match the post-attention state')
    # Generic preparation subtracts its source directions; these edits add writes.
    directions=-torch.stack(changes,-2)
    generic=prepare(postattention_state,directions,left,right,down,bias)
    return compile_program(generic),generic


def compile_program(generic):
    n=generic['numerator'];d=generic['denominator']
    numerator=n.new_zeros(*n.shape[:-2],9,n.shape[-1])
    denominator=d.new_zeros(*d.shape[:-1],9)
    for old,new in enumerate(MAP):
        numerator[...,new,:]+=n[...,old,:]
        denominator[...,new]+=d[...,old]
    return dict(numerator=numerator,denominator=denominator,
                linear_basis=generic['linear_basis'].clone(),bias=generic['bias'].clone())


def execute(program,a,b):
    ref=program['numerator']
    a,b=[torch.as_tensor(x,dtype=ref.dtype,device=ref.device) for x in [a,b]]
    if a.ndim or b.ndim:raise ValueError('Two scalar strengths required')
    m=torch.stack([torch.ones_like(a),a,b,a*b,a*a,a*a*b,b*b,a*b*b,a*a*b*b])
    residual=torch.einsum('...kd,k->...d',program['linear_basis'],m[:4])
    numerator=torch.einsum('...kd,k->...d',ref,m)
    return residual+numerator/(program['denominator']@m)[...,None]+program['bias']
