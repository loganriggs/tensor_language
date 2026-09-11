"""Exact numerator/normalizer accounting for a finite normalized router edit."""
import torch
from folded_normalized_router_v1 import rotary,EPS


def factors(weights,query,source,query_position,source_position):
    width=weights[0].shape[0]
    rt=rotary(query_position,width).to(query);rs=rotary(source_position,width).to(query)
    numerator=torch.ones(query.shape[0],dtype=query.dtype,device=query.device)
    norm_product=torch.ones_like(numerator)
    for q,k in [weights[:2],weights[2:]]:
        qx=query@q.T;ks=source@k.T
        numerator*=((qx@rt.T)*(ks@rs.T)).sum(-1)/width
        norm_product*=(qx.square().mean(-1)+EPS)*(ks.square().mean(-1)+EPS)
    return numerator,norm_product.sqrt()


def response(weights,query,source,edited_source,query_position,source_position):
    n,d=factors(weights,query,source,query_position,source_position)
    no,do=factors(weights,query,edited_source,query_position,source_position)
    numerator=(n-no)/d
    normalizer=no*(1/d-1/do)
    return dict(numerator=numerator,normalizer=normalizer,full=n/d,edited=no/do,
                full_denominator=d,edited_denominator=do)


def normalizer_ports(key,basis):
    """Relative quadratic coefficient energy; epsilon is separately retained."""
    gram=key@key.T/key.shape[0];projected=key@basis
    inside=(projected.T@projected/key.shape[0]).square().sum()
    ge=key.T@projected/key.shape[0]
    mixed=2*(ge.square().sum()-inside)
    total=gram.square().sum();outside=total-inside-mixed
    return {k:float(v/total) for k,v in [('inside',inside),('mixed',mixed),('outside',outside)]}
