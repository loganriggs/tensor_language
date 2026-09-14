"""Exact conditional MLP response on three native interaction-write directions.

All coefficients derive from model weights and pristine states. No response
targets are fitted. Native FP32 RMS epsilon and Down_bias are explicit.
"""
import torch
EPS=torch.finfo(torch.float32).eps
PAIRS=[(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)]


def monomials(amplitudes,reference):
    a=torch.as_tensor(amplitudes,dtype=reference.dtype,device=reference.device)
    if a.shape!=(3,):raise ValueError('Three scalar intervention strengths required')
    return torch.stack([torch.ones_like(a[0]),*a,*[a[i]*a[j] for i,j in PAIRS]])


def prepare(z,directions,left,right,down,bias):
    """z[...,D], directions[...,3,D]; returns explicit per-context coefficients."""
    z=z.double();directions=directions.double()
    l,r,d,b=[x.double() for x in [left,right,down,bias]]
    # x(a)=z-sum_i a_i directions_i. Project each source once.
    basis=torch.cat([z.unsqueeze(-2),-directions],-2)
    lb,rb=basis@l.T,basis@r.T
    numerator=[lb[...,0,:]*rb[...,0,:]]
    numerator.extend(lb[...,0,:]*rb[...,i+1,:]+lb[...,i+1,:]*rb[...,0,:] for i in range(3))
    for i,j in PAIRS:
        term=lb[...,i+1,:]*rb[...,j+1,:]
        if i!=j:term=term+lb[...,j+1,:]*rb[...,i+1,:]
        numerator.append(term)
    numer=torch.stack(numerator,-2)@d.T
    width=z.shape[-1]
    denom=[z.square().mean(-1)+EPS]
    denom.extend(2*(basis[...,0,:]*basis[...,i+1,:]).mean(-1) for i in range(3))
    denom.extend((1 if i==j else 2)*(basis[...,i+1,:]*basis[...,j+1,:]).sum(-1)/width for i,j in PAIRS)
    return dict(numerator=numer,denominator=torch.stack(denom,-1),linear_basis=basis,bias=b)


def execute(program,amplitudes,freeze_rms=False):
    m=monomials(amplitudes,program['numerator'])
    numerator=torch.einsum('...kd,k->...d',program['numerator'],m)
    rho=(program['denominator'][...,0] if freeze_rms else program['denominator']@m)
    residual=torch.einsum('...kd,k->...d',program['linear_basis'],m[:4])
    return residual+numerator/rho[...,None]+program['bias']
