"""Streamed gradients versus full autodiff and profiled finite differences."""
import json
from pathlib import Path
import torch
from shared_mixed_gradient import coefficient_rect,gaussian_rect,streamed_gradient
from shared_gaussian_moments import gram as gg,native_cross as gx
from sparse_quartic_bank import gram as cg,native_cross as cx,support
from noncentral_gaussian_cp import project_shifted
P=Path(__file__).resolve().parent

def controls():
    torch.set_num_threads(2);rows=[];dtype=torch.float64
    for seed,family in enumerate(['independent','shared_input','shared_output','squares','cancellation']):
        torch.manual_seed(12100+seed);u=torch.randn(4,2,3,dtype=dtype);v=torch.randn_like(u)
        if family=='shared_input':u[1]=u[0]
        if family=='squares':v=u.clone()
        if family=='cancellation':u[1]=u[0];v[1]=-v[0]
        teacher=[torch.randn(*shape,dtype=dtype) for shape in [(2,3),(3,3),(3,3),(3,4),(4,3),(4,3)]]
        if family=='shared_output':teacher[0][1]=teacher[0][0]
        pairs=support(4,7,99);S=torch.randn(3,3,dtype=dtype)+3*torch.eye(3,dtype=dtype);mu=torch.randn(3,dtype=dtype);loc=torch.linalg.solve(S,mu);tr=teacher[:4]+[teacher[4]@S,teacher[5]@S];projection=project_shifted(tr,loc);lam=37.;ridge=1e-2
        def fit(a,b):
            G=(gg(a@S,b@S,pairs,a@mu,b@mu)+lam*cg(a,b,pairs))/(1+lam)
            X=(gx(tr,loc,projection,a@S,b@S,pairs,a@mu,b@mu)+lam*cx(teacher,a,b,pairs))/(1+lam)
            C=torch.linalg.solve(G+ridge*torch.eye(len(G),dtype=dtype),X.T).T
            return G,X,C
        with torch.no_grad():G,X,C=fit(u,v)
        U=u.clone().requires_grad_();V=v.clone().requires_grad_();gf,xf,_=fit(U,V)
        loss=(gf*(C.T@C)).sum()-2*(C*xf).sum();reference=torch.autograd.grad(loss,[U,V]);direct_value=float(loss.detach())
        U=u.clone().requires_grad_();V=v.clone().requires_grad_();value=streamed_gradient(U,V,pairs,C,teacher,tr,loc,projection,S,mu,lam,chunk=2)
        rel=lambda a,b:float((a-b).norm()/b.norm().clamp_min(1e-15))
        gradient=rel(torch.cat([U.grad.flatten(),V.grad.flatten()]),torch.cat([a.flatten() for a in reference]));value_error=abs(value-direct_value)/(1+abs(direct_value))
        rect0=rel(coefficient_rect(u,v,pairs[:,:3],pairs[:,3:]),cg(u,v,pairs)[:3,3:]);rect1=rel(gaussian_rect(u@S,v@S,pairs[:,:3],pairs[:,3:],u@mu,v@mu),gg(u@S,v@S,pairs,u@mu,v@mu)[:3,3:])
        du,dv=torch.randn_like(u),torch.randn_like(v)
        def objective(a,b):
            g,x,c=fit(a,b);return ((g*(c.T@c)).sum()-2*(c*x).sum()+ridge*c.square().sum()).item()
        analytic=float((U.grad*du).sum()+(V.grad*dv).sum());ladder=[]
        for step in [1e-5,1e-6,1e-7]:
            with torch.no_grad():fd=(objective(u+step*du,v+step*dv)-objective(u-step*du,v-step*dv))/(2*step)
            ladder.append(dict(step=step,relative_error=abs(fd-analytic)/(1+abs(analytic))))
        assert gradient<1e-9 and value_error<1e-10 and max(rect0,rect1)<1e-10 and ladder[-1]['relative_error']<1e-5,(family,gradient,value_error,ladder)
        rows.append(dict(family=family,gradient_error=gradient,value_error=value_error,coefficient_rectangle_error=rect0,gaussian_rectangle_error=rect1,profile_finite_difference_ladder=ladder,ridge=ridge))
    return rows
if __name__=='__main__':
    rows=controls();(P/'SHARED_MIXED_GRADIENT_CONTROLS_V1.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))
