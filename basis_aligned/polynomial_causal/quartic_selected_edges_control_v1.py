"""Selected graph versus full kernel, with variable-projection tangent gradient.

A coefficient/feature identities<=1e-10; B tangent FD relative error<=1e-5;
C small projected-gradient step descends. Synthetic only.
"""
from pathlib import Path
import json
import torch
import quadratic_product_core_v1 as full
import quartic_selected_edges_v1 as selected

torch.set_default_dtype(torch.float64);torch.set_num_threads(2);torch.manual_seed(91921)
b=torch.linalg.qr(torch.randn(4,8,2),mode='reduced')[0]
n=torch.randn(4,2);n/=n.norm(dim=-1,keepdim=True)
ids=torch.tensor([0,1,4,6,8]);edges=full.pairs(4)[:,ids]
# Independent symmetric target: sum of signed fourth powers of linear readers.
readers=torch.randn(7,8);writes=torch.randn(7,2)
def oracle(slots):return (slots@readers.T).prod(1)@writes
k=selected.gram(b,n,edges);c=selected.target_cross(b,n,edges,oracle)
kref=full.gram(b,n)[ids][:,ids];cref=full.target_cross(b,n,oracle)[ids]
x=torch.randn(11,8)
def relative(a,z):return float((a-z).norm()/z.norm().clamp_min(1e-30))
errors=[relative(k,kref),relative(c,cref),relative(selected.features(b,n,edges,x),full.features(b,n,x)[:,ids])]
def objective(b,n):
    k=selected.gram(b,n,edges);c=selected.target_cross(b,n,edges,oracle)
    with torch.no_grad():mix=torch.linalg.solve(k,c)
    return ((mix*(k@mix)).sum()-2*(mix*c).sum())
def tangent(b,n,db,dn):
    z=b.transpose(-1,-2)@db
    return db-b@((z+z.transpose(-1,-2))/2),dn-(dn*n).sum(-1,keepdim=True)*n
def retract(b,n,db,dn,t):
    q,r=torch.linalg.qr(b+t*db,mode='reduced')
    q=q*r.diagonal(dim1=-2,dim2=-1).sign()[:,None,:]
    nn=n+t*dn
    return q,nn/nn.norm(dim=-1,keepdim=True)
b.requires_grad_();n.requires_grad_();loss=objective(b,n)
gb,gn=tangent(b,n,*torch.autograd.grad(loss,(b,n)))
db,dn=tangent(b,n,torch.randn_like(b),torch.randn_like(n))
norm=(db.square().sum()+dn.square().sum()).sqrt();db/=norm;dn/=norm
with torch.no_grad():
    eps=1e-6
    numerical=float((objective(*retract(b,n,db,dn,eps))-objective(*retract(b,n,db,dn,-eps)))/(2*eps))
    analytic=float((gb*db).sum()+(gn*dn).sum())
    fd=abs(analytic-numerical)/max(abs(analytic),abs(numerical),1e-12)
    norm=(gb.square().sum()+gn.square().sum()).sqrt()
    after=float(objective(*retract(b,n,-gb/norm,-gn/norm,1e-4)))
result=dict(pred_a=max(errors)<=1e-10,pred_b=fd<=1e-5,pred_c=after<float(loss.detach()),
            identity_errors=errors,tangent_fd_error=fd,initial_objective=float(loss.detach()),
            after_small_step=after,edges=edges.tolist(),scope='Synthetic fixed mixed graph, not native timing, convergence or circuit recovery.')
out=Path(__file__).with_name('QUARTIC_SELECTED_EDGES_V1_CONTROL.json');assert not out.exists()
out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
assert result['pred_a'] and result['pred_b'] and result['pred_c']
