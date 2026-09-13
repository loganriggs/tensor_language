"""Exact full/full, mixed full/inside, and twice inside/inside scalar terms."""
import torch
from contracted_qk_response_v1 import EPS

def terms(reads,rho2):
    n=reads.shape[-2];inv=1/(10000**(torch.arange(0,128,2,dtype=torch.float32)/128))
    angle=torch.outer(torch.arange(n,dtype=torch.float32),inv)
    co=angle.cos().bfloat16().to(reads.device);si=angle.sin().bfloat16().to(reads.device)
    def rot(x):
        a,b=x.chunk(2,-1);return torch.cat((a*co+b*si,-a*si+b*co),-1)
    full=[];inside=[]
    for qi,ki,ii in [(0,128,512),(256,384,640)]:
        q=reads[...,qi:qi+128];k=reads[...,ki:ki+128]
        q=rot(q/(q.square().mean(-1,keepdim=True)+EPS*rho2).sqrt())
        den=(k.square().mean(-1,keepdim=True)+EPS*rho2).sqrt()
        full.append(q@rot(k/den).transpose(-1,-2)/128)
        inside.append(q@rot(reads[...,ii:ii+128]/den).transpose(-1,-2)/128)
    gammas=torch.stack([full[0]*full[1],-full[0]*inside[1]-inside[0]*full[1],2*inside[0]*inside[1]])
    gammas=gammas.masked_fill(~torch.ones(n,n,dtype=torch.bool,device=reads.device).tril(),0)
    return (gammas@(reads[...,-1:]/rho2.sqrt()))[...,0]
