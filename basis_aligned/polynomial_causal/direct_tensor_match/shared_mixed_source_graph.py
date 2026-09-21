"""Two components share mixed products; a third has a private paired core.
Inputs z,h are native states. All source products are evaluated once per call.
"""
import torch
from quadratic_pair_blocks import products

def source_reads(z,program):
    shared=program['shared_mixed'];private=program['private_pair']
    shared_terms=(z@shared['left_reader'])*(z@shared['right_reader'])
    first=shared_terms@shared['product_weights']+z@shared['source_linear']+shared['source_bias']
    last=products(z@private['shared_reader'],private['product_indices'])@private['product_weights']
    last=last+torch.stack([z@private[k+'_linear']+private[k+'_bias'] for k in ['a','b']],-1)
    return torch.cat([first,last],-1)

def component_scalars(z,h,program):
    shared=program['shared_mixed'];private=program['private_pair'];reads=source_reads(z,program)
    hreaders=torch.cat([shared['h_readers'],private['h_reader'][:,None]],1)
    alpha=torch.cat([shared['alpha'],private['alpha'].reshape(1)]);beta=torch.cat([shared['beta'],private['beta'].reshape(1)])
    scale=(h.square().mean(-1)+torch.finfo(torch.float32).eps).sqrt()[...,None]
    return ((h@hreaders-.5*reads[...,::2])/scale-alpha)*(reads[...,1::2]/scale-beta)

def residual_write(z,h,program,components=None):
    values=component_scalars(z,h,program)
    if components is not None:values=values[...,components]
    return values.sum(-1)[...,None]*program['residual_writer']
