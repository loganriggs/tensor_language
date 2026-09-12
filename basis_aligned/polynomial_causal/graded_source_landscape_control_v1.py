"""Strict nonglobal minimum of the actual degree-balanced source objective."""
from pathlib import Path
import json,hashlib,time
import torch
from graded_source_projection_v1 import graded_norms,balanced_loss
from coupled_source_projection_v1 import tangent,retract
P=Path(__file__).resolve().parent

def main():
    torch.set_num_threads(2);dtype=torch.float64;tic=time.perf_counter();alpha=.8
    root=torch.eye(2,dtype=dtype)
    forms=torch.stack([torch.diag(torch.tensor([1.,-alpha**.5],dtype=dtype)),torch.diag(torch.tensor([1.,alpha**.5],dtype=dtype))])
    wg=torch.ones(1,1,dtype=dtype);full=graded_norms(forms,root,wg)
    def value(p):return balanced_loss(forms,root,p,wg,full)
    stationary=[]
    for theta in [0.,torch.pi/2]:
        t=torch.tensor(theta,dtype=dtype,requires_grad=True);p=torch.stack([t.cos(),t.sin()])[:,None]
        loss=value(p);g=torch.autograd.grad(loss,t,create_graph=True)[0];h=torch.autograd.grad(g,t)[0]
        stationary.append(dict(theta=theta,loss=float(loss.detach()),gradient=float(g.detach()),curvature=float(h.detach())))
    expected_losses=[alpha**2/(1+alpha**2),1/(1+alpha**2)]
    expected_curvatures=[(10-alpha**2)/(2*(1+alpha**2)),(10*alpha**2-1)/(2*(1+alpha**2))]
    errors=[abs(x['loss']-l)+abs(x['curvature']-h)+abs(x['gradient']) for x,l,h in zip(stationary,expected_losses,expected_curvatures)]
    assert max(errors)<1e-12 and stationary[1]['curvature']>0 and stationary[1]['loss']>stationary[0]['loss']
    fits=[]
    # Same QR/Armijo rule as the native runner; starts near each distinct basin.
    for angle in [.2,1.4]:
        p=torch.tensor([[torch.cos(torch.tensor(angle)).item()],[torch.sin(torch.tensor(angle)).item()]],dtype=dtype)
        p=torch.linalg.qr(p).Q;step=4.;initial=float(value(p));monotone=True
        for iteration in range(1000):
            p=p.detach().requires_grad_();loss=value(p);tg=tangent(p,torch.autograd.grad(loss,p)[0]);gn=float(tg.norm().detach());lv=float(loss.detach())
            if gn<=1e-8:break
            with torch.no_grad():
                accepted=False
                for _ in range(20):
                    proposal=retract(p,-step*tg);new=float(value(proposal))
                    if new<=lv-1e-4*step*gn*gn:accepted=True;break
                    step/=2
                if not accepted:break
                monotone=monotone and new<=lv+1e-10;p=proposal;step=min(100.,step*1.5)
        p=p.detach().requires_grad_();loss=value(p);tg=tangent(p,torch.autograd.grad(loss,p)[0]);gn=float(tg.norm().detach())
        fits.append(dict(initial_angle=angle,initial_loss=initial,loss=float(loss.detach()),gradient=gn,iterations=iteration,monotone=monotone,axis1_squared=float(p[0,0].detach().square())))
    result=dict(stationary_points=stationary,analytic_errors=errors,fits=fits,
                strict_bad_local_minimum=True,execution_seconds=time.perf_counter()-tic,
                source_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                scope='Exact two-dimensional counterexample, not proof of a native-model trap or a global-recovery algorithm.')
    out=P/'GRADED_SOURCE_LANDSCAPE_CONTROL_V1_RESULT.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
