"""Three-output-vector approximation to the exact five-bank self interaction."""
import torch
from response_product_basis_v2 import coefficients
from conic_amplitude_polynomial_v1 import compile_degree2,evaluate as polynomial


def prepare(context,left,right,down,bias):
    basis=context['basis'];lb=basis@left.T;rb=basis@right.T
    bank,scale=compile_degree2(lb,rb,down,context['response'])
    return dict(context=context,left=left,right=right,down=down,bias=bias,
        left_basis=lb,right_basis=rb,bank=bank,scale=scale)


def evaluate(z,amplitude,prepared):
    context=prepared['context'];u=coefficients(amplitude,context['response'])
    residual=torch.einsum('...k,...kd->...d',u,context['basis'])
    background=z-residual
    yl=background@prepared['left'].T;yr=background@prepared['right'].T
    rl=torch.einsum('...k,...kh->...h',u,prepared['left_basis'])
    rr=torch.einsum('...k,...kh->...h',u,prepared['right_basis'])
    self_term=polynomial(amplitude/prepared['scale'],prepared['bank'],prepared['scale'],context['response'],2)
    numerator=(yl*yr+yl*rr+rl*yr)@prepared['down'].T+.5*self_term
    rho=z.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps
    return z+numerator/rho+prepared['bias']
