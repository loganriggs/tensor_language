"""Full normalized MLP branch with exact five-vector residual self-product."""
import torch
from response_product_basis_v2 import coefficients
from response_diagonal_conic_v1 import prepare_direct,execute

def prepare(context,left,right,down,bias):
    basis=context['basis']
    return dict(context=context,left=left,right=right,down=down,bias=bias,
        left_basis=basis@left.T,right_basis=basis@right.T,
        bank=prepare_direct(basis,left,right,down,context['response']))

def evaluate(z,amplitude,prepared):
    context=prepared['context'];u=coefficients(amplitude,context['response'])
    residual=torch.einsum('...k,...kd->...d',u,context['basis'])
    background=z-residual
    yl=background@prepared['left'].T;yr=background@prepared['right'].T
    rl=torch.einsum('...k,...kh->...h',u,prepared['left_basis'])
    rr=torch.einsum('...k,...kh->...h',u,prepared['right_basis'])
    numerator=(yl*yr+yl*rr+rl*yr)@prepared['down'].T+.5*execute(u,prepared['bank'])
    rho=z.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps
    return z+numerator/rho+prepared['bias']
