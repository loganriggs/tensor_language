"""Twelve-vector attention input basis retaining the shared RMS denominator."""
import torch
from coupled_attention10_ports_v1 import EPS,MAPS,monomials,attention_write


def compile_program(source,x0,lambdas,weights,first_values):
    alpha,beta=lambdas.double()
    basis=alpha*source['linear_basis'].double()
    basis[...,0,:]+=alpha*source['bias'].double()+beta*x0.double()
    rho=source['denominator']
    numerator=alpha*source['numerator']+rho[...,None]*basis[...,0,:].unsqueeze(-2)
    vectors=torch.cat([basis[...,1:,:],numerator],-2)
    program={name:vectors@weights[native].double().T for name,native in MAPS.items()}
    program.update(norm_gram=(vectors@vectors.transpose(-1,-2))/vectors.shape[-1],
                   denominator=rho.clone(),first_values=first_values.clone(),
                   output=weights['c_proj'].clone(),mixture=weights['lamb'].clone())
    return program


def normalized_ports(program,a,b):
    m9,_=monomials(a,b,program['q1'])
    rho=program['denominator']@m9
    if not bool((rho>0).all()):raise ValueError('Positive original RMS denominator required')
    scalars=torch.cat([rho[...,None]*m9[1:4],m9.expand(*rho.shape,9)],-1)
    r=torch.einsum('...i,...ij,...j->...',scalars,program['norm_gram'],scalars)+EPS*rho.square()
    if not bool((r>0).all()):raise ValueError('Nonpositive input radicand')
    ports={}
    for name in MAPS:
        raw=torch.einsum('...kd,...k->...d',program[name],scalars)
        raw=raw.reshape(*raw.shape[:-1],9,128)
        ports[name]=raw/r.sqrt()[...,None,None] if name=='v' else raw/(raw.square().mean(-1,keepdim=True)+EPS*r[...,None,None]).sqrt()
    return ports


def execute(program,a,b):
    first=program['first_values'];batch,tokens=program['q1'].shape[:2]
    if first.shape not in ((batch,tokens,1152),(batch,tokens,9,128)):
        raise ValueError('First values must be B,T,1152 or B,T,9,128')
    return attention_write(normalized_ports(program,a,b),program['output'],program['mixture'],first.reshape(batch,tokens,1152))
