"""Affine CP degree-zero/one/two Hermite projection, exact under N(0,I)."""
import itertools
import torch
from noncentral_gaussian_cp import affine_moment

def project(C,f,b):
    m=C@affine_moment(f,b);lin=C.new_zeros((len(C),f[0].shape[1]));Q=C.new_zeros((len(C),f[0].shape[1],f[0].shape[1]))
    for i in range(4):
        rest=[j for j in range(4) if j!=i];w=affine_moment([f[j] for j in rest],[b[j] for j in rest]);lin+=(C*w)@f[i]
    for i,j in itertools.combinations(range(4),2):
        rest=[k for k in range(4) if k not in (i,j)];w=affine_moment([f[k] for k in rest],[b[k] for k in rest])
        for v,c in enumerate(C):
            a=f[i].T@((c*w)[:,None]*f[j]);Q[v]+=(a+a.T)/2
    return m,lin,Q

def controls():
    import numpy as np
    nodes,ws=np.polynomial.hermite.hermgauss(5);z=torch.tensor(list(itertools.product(nodes*2**.5,repeat=3)),dtype=torch.float64);weights=torch.tensor([a*b*c for a,b,c in itertools.product(ws/np.pi**.5,repeat=3)],dtype=torch.float64);eye=torch.eye(3,dtype=z.dtype);checks=[]
    for seed in range(5):
        torch.manual_seed(20000+seed);f=[torch.randn(4,3,dtype=z.dtype) for _ in range(4)];b=[torch.randn(4,dtype=z.dtype) for _ in range(4)];C=torch.randn(2,4,dtype=z.dtype)
        if seed==1:b=[torch.zeros_like(t) for t in b]
        if seed==2:f=[f[0]]*4
        if seed==3:C[1]=C[0]*2
        if seed==4:f[3]=torch.zeros_like(f[3])
        y=torch.stack([z@a.T+c for a,c in zip(f,b)]).prod(0)@C.T
        reference=[(weights[:,None]*y).sum(0),torch.einsum('n,nv,ni->vi',weights,y,z),.5*torch.einsum('n,nv,nij->vij',weights,y,z[:,:,None]*z[:,None,:]-eye)]
        errors=[float((a-v).abs().max()) for a,v in zip(project(C,f,b),reference)];assert max(errors)<1e-9
        checks.append(dict(seed=seed,errors=errors))
    return checks
