"""Exact last-slot conditional expectation and parameter-gradient control."""
from pathlib import Path
import itertools,json,time
import torch
from integrated_coefficient_slot_v1 import integrate
from mixed_repeated_contraction_v1 import contract
P=Path(__file__).resolve().parent
def main():
    torch.set_num_threads(2);torch.manual_seed(73250);dtype=torch.float64;tic=time.perf_counter()
    l=torch.randn(3,2,dtype=dtype);r=torch.randn_like(l);d=torch.randn(2,3,dtype=dtype)
    a=torch.randn(3,2,2,dtype=dtype);a=(a+a.transpose(-1,-2))/2
    w=torch.randn(2,3,dtype=dtype);wg=w@w.T
    theta=torch.tensor(.4,dtype=dtype,requires_grad=True);p=torch.stack([theta.cos(),theta.sin()])[:,None];dd=p@(p.T@d)
    reports=[]
    for k in range(1,5):
        b=torch.randn(12,4-k,2,dtype=dtype);x=torch.randn(12,2*k-1,2,dtype=dtype)
        def fun(z):
            xx=torch.cat([x,z[:,None]],dim=1)
            return contract(b,xx,l,r,dd,a,k)-contract(b,xx,l,r,d,a,k)
        rb=integrate(fun,torch.zeros(12,2,dtype=dtype),wg)
        values=[];gradients=[]
        for signs in itertools.product((-1.,1.),repeat=2):
            y=fun(torch.tensor(signs,dtype=dtype).expand(12,-1));value=torch.einsum('ni,ij,nj->n',y,wg,y);values.append(value)
            gradients.append(torch.stack([torch.autograd.grad(v,theta,retain_graph=True)[0] for v in value]))
        values=torch.stack(values);gradients=torch.stack(gradients)
        exact=values.mean(0);grb=torch.stack([torch.autograd.grad(v,theta,retain_graph=True)[0] for v in rb])
        err=float(((rb-exact).norm()/exact.norm()).detach());ge=float(((grb-gradients.mean(0)).norm()/grb.norm()).detach())
        total=float(gradients.var(unbiased=False));conditional=float(gradients.var(dim=0,unbiased=False).mean());remaining=float(gradients.mean(0).var(unbiased=False))
        reports.append(dict(grade=k,value_error=err,gradient_error=ge,finite_design_variance_identity_error=abs(total-conditional-remaining)/max(total,1e-30),finite_design_remaining_variance_fraction=remaining/total))
    assert max(z[key] for z in reports for key in ('value_error','gradient_error','finite_design_variance_identity_error'))<1e-9
    out=P/'INTEGRATED_COEFFICIENT_SLOT_V1_CONTROL.json';assert not out.exists();result=dict(reports=reports,execution_seconds=time.perf_counter()-tic,scope='Exact conditional integration under identity covariance; finite toy design only. No native variance or cost conclusion.');out.write_text(json.dumps(result,indent=2)+'\n');print(result)
if __name__=='__main__':main()
