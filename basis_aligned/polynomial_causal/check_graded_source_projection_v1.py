"""Dense coefficient-group and tangent-gradient control for graded source norm."""
from pathlib import Path
import itertools,json,hashlib
import torch
from graded_source_projection_v1 import graded_norms,retained_grades,balanced_loss
from coupled_source_projection_v1 import tangent
P=Path(__file__).resolve().parent
def main():
    torch.set_num_threads(2);torch.manual_seed(73141);dtype=torch.float64;d=3
    a=torch.randn(3,d,d,dtype=dtype);a=(a+a.transpose(-1,-2))/2
    root=torch.randn(d,d,dtype=dtype);root=root@root.T+torch.eye(d,dtype=dtype);w=torch.randn(4,2,dtype=dtype);wg=w.T@w
    p=torch.linalg.qr(torch.randn(d,2,dtype=dtype)).Q.requires_grad_()
    t=torch.cat([torch.eye(d,dtype=dtype),root],-1);s=torch.cat([torch.eye(d,dtype=dtype),root@p@p.T],-1)
    def dense(t):
        f=t.T@a@t;raw=torch.einsum('ab,jcd->jabcd',f[0],f[1:]);sym=sum(raw.permute(0,*[i+1 for i in perm]) for perm in itertools.permutations(range(4)))/24
        return torch.einsum('oj,jabcd->oabcd',w,sym)
    indices=torch.cartesian_prod(*[torch.arange(2*d)]*4);degree=(indices>=d).sum(-1)
    def norms(x):
        x=x.flatten(1);return torch.stack([x[:,degree==i].square().sum() for i in range(5)])
    full=graded_norms(a,root@root.T,wg);pred=retained_grades(a,root,p,wg)
    actual=norms(dense(s));target=norms(dense(t))
    loss=balanced_loss(a,root,p,wg,full);dense_loss=(1-actual[1:]/target[1:]).mean()
    g=tangent(p,torch.autograd.grad(loss,p,retain_graph=True)[0]);gd=tangent(p,torch.autograd.grad(dense_loss,p)[0])
    scaled=graded_norms(a,9*root@root.T,wg)
    errors=dict(full=float(((full-target)/target).abs().max()),projected=float(((pred-actual)/actual).abs().max()),
                gradient=float((g-gd).norm()/gd.norm()),source_scale=float(((scaled/full)-full.new_tensor([9**k for k in range(5)])).abs().max()/9**4))
    assert max(errors.values())<1e-10
    result=dict(errors=errors,source_sha=hashlib.sha256((P/'graded_source_projection_v1.py').read_bytes()).hexdigest());out=P/'GRADED_SOURCE_PROJECTION_V1_CONTROL.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(result)
if __name__=='__main__':main()
