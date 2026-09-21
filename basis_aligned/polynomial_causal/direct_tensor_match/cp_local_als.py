"""Batched small-core ALS with balanced factors and retained best iterates."""
import torch

@torch.no_grad()
def fit(core, initial, steps=12000, checkpoints=(0,1000,4000,11000,12000)):
    a,b,c=[v.clone() for v in initial];rank=a.shape[-1];batch=a.shape[0]
    den=core.square().sum();best=torch.full((batch,),float('inf'),dtype=core.dtype)
    saved=[v.clone() for v in [a,b,c]];history=[]
    def solve(rhs,x,y):
        gram=torch.einsum('bdi,bdj->bij',x,x)*torch.einsum('bdi,bdj->bij',y,y)
        ridge=1e-12*gram.diagonal(dim1=1,dim2=2).mean(1).clamp_min(1e-30)
        return torch.linalg.solve(gram+ridge[:,None,None]*torch.eye(rank,dtype=gram.dtype),rhs.transpose(1,2)).transpose(1,2)
    for step in range(steps+1):
        if step%10==0 or step==steps:
            pred=torch.einsum('bir,bjr,bkr->bijk',a,b,c)
            error=(pred-core).square().sum((1,2,3))/den
            changed=error<best;best=torch.minimum(best,error)
            for q,v in zip(saved,[a,b,c]):q[changed]=v[changed]
            if step in checkpoints:history.append(dict(step=step,errors=error.tolist(),best=best.tolist()))
        if step==steps:break
        a=solve(torch.einsum('ijk,bjr,bkr->bir',core,b,c),b,c)
        b=solve(torch.einsum('ijk,bir,bkr->bjr',core,a,c),a,c)
        c=solve(torch.einsum('ijk,bir,bjr->bkr',core,a,b),a,b)
        norms=torch.stack([v.norm(dim=1).clamp_min(1e-30) for v in [a,b,c]])
        target=norms.prod(0).pow(1/3)
        a*=(target/norms[0])[:,None,:];b*=(target/norms[1])[:,None,:];c*=(target/norms[2])[:,None,:]
    return saved,best,history
