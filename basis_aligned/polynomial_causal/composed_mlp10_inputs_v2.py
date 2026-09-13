"""Shared-response graph through full attention, including self-generated norms."""
import torch
from response_product_basis_v2 import prepare_basis,coefficients,prepare_products,combine
from response_attention_projection_v2 import prepare_global,prepare,changed
from raw_attention_response_v1 import execute as attention,EPS


def execute(z9,biasfree_mlp9,raw10,pristine_z10,first_values,a,b,response_program,
            reentry_scale,attention_matrices,attention_mixture,attention_output,global_cache=None):
    response,basis=prepare_basis(z9.double(),biasfree_mlp9,response_program,reentry_scale)
    if global_cache is None:global_cache=prepare_global(response_program,attention_matrices,reentry_scale)
    context=prepare(raw10.double(),response,global_cache,attention_matrices)
    rho=raw10.double().square().mean(-1,keepdim=True)+EPS
    baseline=attention(context['baseline'],rho,first_values,attention_mixture,attention_output)
    coeff=[coefficients(x,response) for x in [a,b]]
    residuals=[torch.einsum('...k,...kd->...d',u,basis) for u in coeff]
    partners=[]
    for amplitude in [a,b]:
        ports,norm=changed(amplitude,context)
        partners.append(attention(ports,norm,first_values,attention_mixture,attention_output)-baseline)
    c,r=[x+y for x,y in zip(residuals,partners)]
    joint=(pristine_z10.double()+c+r).square().mean(-1,keepdim=True)+EPS
    return dict(child=c,remainder=r,child_residual=residuals[0],remainder_residual=residuals[1],
        child_attention=partners[0],remainder_attention=partners[1],joint_rho=joint,
        response_basis=basis,response_coefficients=coeff)


def product(parts,left,right,down):
    pairs,bank=prepare_products(parts['response_basis'],left,right,down)
    rr=combine(*parts['response_coefficients'],pairs,bank)
    def cross(x,y):return ((x@left.T)*(y@right.T)+(y@left.T)*(x@right.T))@down.T
    mixed=cross(parts['child_residual'],parts['remainder_attention'])+cross(parts['child_attention'],parts['remainder_residual'])
    aa=cross(parts['child_attention'],parts['remainder_attention'])
    return (rr+mixed+aa)/parts['joint_rho']
