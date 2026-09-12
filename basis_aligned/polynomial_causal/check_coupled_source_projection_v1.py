"""Dense quartic and finite-difference checks of common-source projection."""
from pathlib import Path
import itertools,json,hashlib
import torch
from coupled_source_projection_v1 import norm,retained,tangent,retract

P=Path(__file__).resolve().parent
def main():
    torch.set_num_threads(2);torch.manual_seed(73140);dtype=torch.float64;d=4
    forms=torch.randn(3,d,d,dtype=dtype);forms=(forms+forms.transpose(-1,-2))/2
    root=torch.randn(d,d,dtype=dtype);root=root@root.T+torch.eye(d,dtype=dtype)
    w=torch.randn(5,2,dtype=dtype);wg=w.T@w
    p=torch.linalg.qr(torch.randn(d,2,dtype=dtype)).Q.requires_grad_()
    def dense(t):
        a=t.T@forms@t
        raw=torch.einsum('ab,jcd->jabcd',a[0],a[1:])
        sym=sum(raw.permute(0,*[i+1 for i in perm]) for perm in itertools.permutations(range(4)))/24
        return torch.einsum('oj,jabcd->oabcd',w,sym)
    t=torch.cat([torch.eye(d,dtype=dtype),root],-1)
    s=torch.cat([torch.eye(d,dtype=dtype),root@p@p.T],-1)
    native=dense(t);approx=dense(s);value=retained(forms,root,p,wg)
    errors=dict(norm=float((value-approx.square().sum()).abs()/value),
                projection=float(((native*approx).sum()-value).abs()/value),
                loss=float(((native-approx).square().sum()-(norm(forms,t@t.T,wg)-value)).abs()/native.square().sum()))
    grad=torch.autograd.grad(value,p,retain_graph=True)[0]
    dense_grad=torch.autograd.grad(approx.square().sum(),p)[0]
    # Ambient derivatives differ off the Stiefel manifold: compare tangents.
    tg=tangent(p,grad);dg=tangent(p,dense_grad)
    errors['tangent_gradient']=float((tg-dg).norm()/dg.norm())
    direction=tangent(p,torch.randn_like(p));direction=direction/direction.norm();eps=1e-5
    fd=(retained(forms,root,retract(p,eps*direction),wg)-retained(forms,root,retract(p,-eps*direction),wg))/(2*eps)
    analytic=(tg*direction).sum();errors['finite_difference']=float((fd-analytic).abs()/analytic.abs())
    assert max(errors.values())<1e-7
    result=dict(errors=errors,source_sha=hashlib.sha256((P/'coupled_source_projection_v1.py').read_bytes()).hexdigest(),scope='Exact formal source coefficient metric; no model data.')
    out=P/'COUPLED_SOURCE_PROJECTION_V1_CONTROL.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(result)
if __name__=='__main__':main()
