"""Reusable full interaction context; sum RR/mixed/AA before shared Down map."""
import torch
from response_product_basis_v2 import prepare_basis,coefficients
from response_attention_projection_v2 import prepare_global,prepare as projection_context,changed
from raw_attention_response_v1 import execute as attention,EPS


def prepare(z9,biasfree_mlp9,raw10,pristine_z10,first_values,response_program,
            reentry_scale,attention_matrices,attention_mixture,attention_output):
    response,basis=prepare_basis(z9.double(),biasfree_mlp9,response_program,reentry_scale)
    fixed=prepare_global(response_program,attention_matrices,reentry_scale)
    projections=projection_context(raw10.double(),response,fixed,attention_matrices)
    rho=raw10.double().square().mean(-1,keepdim=True)+EPS
    baseline=attention(projections['baseline'],rho,first_values,attention_mixture,attention_output)
    return dict(response=response,basis=basis,projections=projections,baseline_attention=baseline,
        pristine_z10=pristine_z10.double(),first_values=first_values,mixture=attention_mixture,output=attention_output)


def evaluate(a,b,context):
    residuals=[];partners=[]
    for amplitude in [a,b]:
        u=coefficients(amplitude,context['response'])
        residuals.append(torch.einsum('...k,...kd->...d',u,context['basis']))
        ports,rho=changed(amplitude,context['projections'])
        partners.append(attention(ports,rho,context['first_values'],context['mixture'],context['output'])-context['baseline_attention'])
    c,r=[x+y for x,y in zip(residuals,partners)]
    rho=(context['pristine_z10']+c+r).square().mean(-1,keepdim=True)+EPS
    return dict(child=c,remainder=r,joint_rho=rho,child_residual=residuals[0],remainder_residual=residuals[1],child_attention=partners[0],remainder_attention=partners[1])


def product(parts,left,right,down):
    c,r=parts['child'],parts['remainder']
    return ((c@left.T)*(r@right.T)+(r@left.T)*(c@right.T))@down.T/parts['joint_rho']
