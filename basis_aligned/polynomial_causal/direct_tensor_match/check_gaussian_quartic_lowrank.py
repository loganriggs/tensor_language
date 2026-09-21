"""Compare lowrank Gaussian loss and gradients to independent dense kernel."""
import json
from pathlib import Path
import torch
from gaussian_quartic_moment import squared_error_by_degree as dense
from gaussian_quartic_lowrank import prepare_teacher,squared_error_by_degree
torch.set_num_threads(2);torch.manual_seed(538);rows=[]
for n,r in [(7,3),(11,5)]:
 A=torch.randn(n,n,dtype=torch.float64);A=(A+A.T)/2
 B=torch.randn(n,n,dtype=torch.float64);B=(B+B.T)/2
 X=torch.randn(n,r,dtype=torch.float64,requires_grad=True);Y=torch.randn(n,r,dtype=torch.float64,requires_grad=True)
 a=torch.randn(r,dtype=torch.float64,requires_grad=True);b=torch.randn(r,dtype=torch.float64,requires_grad=True)
 teacher=prepare_teacher(A,B)
 direct=dense(A,B,(X*a)@X.T,(Y*b)@Y.T)
 fast=squared_error_by_degree(teacher,X,a,Y,b)
 g1=torch.autograd.grad(direct.sum(),[X,a,Y,b],retain_graph=True)
 g2=torch.autograd.grad(fast.sum(),[X,a,Y,b])
 err=float((direct-fast).detach().norm()/direct.detach().norm())
 grad=max(float((u-v).norm()/u.norm()) for u,v in zip(g1,g2))
 assert err<1e-10 and grad<1e-10
 rows.append(dict(dimension=n,rank=r,value_relative_error=err,gradient_relative_error=grad))
out=dict(passed=True,records=rows)
Path(__file__).with_name('GAUSSIAN_QUARTIC_LOWRANK_CHECK_V1.json').write_text(json.dumps(out,indent=2)+'\n')
print(out)
