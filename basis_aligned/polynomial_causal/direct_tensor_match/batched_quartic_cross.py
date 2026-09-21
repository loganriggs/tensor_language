"""Exact quartic teacher/bank contraction with bounded directional batch size."""
import torch
from quartic_cp import directional

def cross(teacher,u,v,pair_batch=8):
    if pair_batch<1:raise ValueError('pair_batch must be positive')
    i,j=torch.triu_indices(len(u),len(u),device=u.device);k,d=u.shape[1:];pieces=[]
    for start in range(0,len(i),pair_batch):
        a,b=i[start:start+pair_batch],j[start:start+pair_batch];n=len(a)
        vectors=[u[a,:,None,:].expand(n,k,k,d).reshape(n*k*k,d),v[a,:,None,:].expand(n,k,k,d).reshape(n*k*k,d),u[b,None,:,:].expand(n,k,k,d).reshape(n*k*k,d),v[b,None,:,:].expand(n,k,k,d).reshape(n*k*k,d)]
        pieces.append(directional(*teacher,vectors).reshape(n,k*k,-1).sum(1).T)
    return torch.cat(pieces,1)

def controls():
    from shared_quadratic_bank import native_bank_cross
    rows=[]
    for width,batch in [(2,1),(3,4),(5,7)]:
        torch.manual_seed(3000+width);rand=lambda *s:torch.randn(*s,dtype=torch.float64)
        teacher=[rand(*shape) for shape in [(3,4),(4,5),(4,5),(5,6),(6,3),(6,3)]];u=rand(width,2,3).requires_grad_();v=rand(width,2,3).requires_grad_();reference=native_bank_cross(teacher,u,v);actual=cross(teacher,u,v,batch);error=float(((actual-reference).norm()/reference.norm()).detach());ga=torch.autograd.grad(actual.square().sum(),(u,v),retain_graph=True);gb=torch.autograd.grad(reference.square().sum(),(u,v));grad=max(float((a-b).norm()/b.norm()) for a,b in zip(ga,gb));rows.append(dict(width=width,batch=batch,value_error=error,gradient_error=grad))
    assert max(max(r['value_error'],r['gradient_error']) for r in rows)<1e-12
    return rows
if __name__=='__main__':
    import json
    from pathlib import Path
    rows=controls();Path(__file__).with_name('BATCHED_QUARTIC_CROSS_CONTROLS_V1.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows))
