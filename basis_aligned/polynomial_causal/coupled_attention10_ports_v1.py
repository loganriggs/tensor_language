"""Exact conditional attention consumer of a coupled rational MLP state.

Dense Q/K/V maps are applied during preparation. Runtime retains projected
polynomial coefficients, a norm Gram matrix, original output map and first values.
Original model/context generators remain preparation dependencies.
"""
import torch
import torch.nn.functional as F
EPS=torch.finfo(torch.float32).eps
EXPS9=[(0,0),(1,0),(0,1),(1,1),(2,0),(2,1),(0,2),(1,2),(2,2)]
EXPS4=EXPS9[:4]
MAPS={'q1':'c_q','k1':'c_k','q2':'c_q2','k2':'c_k2','v':'c_v'}


def monomials(a,b,reference):
    a,b=[torch.as_tensor(x,dtype=reference.dtype,device=reference.device) for x in [a,b]]
    m9=torch.stack([a**i*b**j for i,j in EXPS9])
    m16=torch.stack([a**i*b**j for i in range(4) for j in range(4)])
    return m9,m16


def compile_program(source,x0,lambdas,weights,first_values):
    alpha,beta=lambdas.double()
    basis=alpha*source['linear_basis'].double()
    basis[...,0,:]+=alpha*source['bias'].double()+beta*x0.double()
    n=source['numerator'];rho=source['denominator']
    coeff=n.new_zeros(*n.shape[:-2],16,n.shape[-1])
    for k,(i,j) in enumerate(EXPS9):
        coeff[...,4*i+j,:]+=alpha*n[...,k,:]
        for h,(u,v) in enumerate(EXPS4):
            coeff[...,4*(i+u)+(j+v),:]+=rho[...,k,None]*basis[...,h,:]
    program={name:coeff@weights[native].double().T for name,native in MAPS.items()}
    program.update(norm_gram=(coeff@coeff.transpose(-1,-2))/coeff.shape[-1],
                   denominator=rho.clone(),first_values=first_values.clone(),
                   output=weights['c_proj'].clone(),mixture=weights['lamb'].clone())
    return program


def normalized_ports(program,a,b):
    m9,m16=monomials(a,b,program['q1'])
    rho=program['denominator']@m9
    if not bool((rho>0).all()):raise ValueError('Positive original RMS denominator required')
    r=torch.einsum('i,...ij,j->...',m16,program['norm_gram'],m16)+EPS*rho.square()
    if not bool((r>0).all()):raise ValueError('Nonpositive normalized-input radicand')
    ports={}
    for name in MAPS:
        raw=torch.einsum('...kd,k->...d',program[name],m16)
        raw=raw.reshape(*raw.shape[:-1],9,128)
        if name=='v':ports[name]=raw/r.sqrt()[...,None,None]
        else:ports[name]=raw/(raw.square().mean(-1,keepdim=True)+EPS*r[...,None,None]).sqrt()
    return ports


def attention_write(ports,output,mixture,first_values):
    q1,k1,q2,k2=[ports[k] for k in ['q1','k1','q2','k2']]
    n=q1.shape[1];device=q1.device
    inv=1/(10000**(torch.arange(0,128,2,dtype=torch.float32)/128))
    angles=torch.outer(torch.arange(n,dtype=torch.float32),inv)
    co,si=angles.cos().bfloat16().to(device)[None,:,None,:],angles.sin().bfloat16().to(device)[None,:,None,:]
    def rotate(x):
        a,b=x.chunk(2,-1);return torch.cat((a*co+b*si,-a*si+b*co),-1)
    q1,k1,q2,k2=[rotate(x) for x in [q1,k1,q2,k2]]
    pattern=(torch.einsum('bthd,bshd->bhts',q1,k1)/128)*(torch.einsum('bthd,bshd->bhts',q2,k2)/128)
    pattern=pattern.masked_fill(~torch.ones(n,n,dtype=torch.bool,device=device).tril(),0)
    lam=mixture.double();v=(1-lam)*ports['v']+lam*first_values.double().reshape_as(ports['v'])
    channels=torch.einsum('bhts,bshd->bthd',pattern,v).reshape(*first_values.shape)
    return channels@output.double().T


def execute(program,a,b):
    return attention_write(normalized_ports(program,a,b),program['output'],program['mixture'],program['first_values'])
