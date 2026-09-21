from pathlib import Path
import torch,json
from quartic_pair_metric import squared_error as dense_error,inner
from quartic_lowrank_metric import squared_error
p=Path(__file__).resolve().parent;torch.set_num_threads(2);torch.manual_seed(2507);records=[]
for n,r in [(7,3),(11,5)]:
 A=torch.randn(n,n,dtype=torch.float64);A=(A+A.T)/2;B=torch.randn(n,n,dtype=torch.float64);B=(B+B.T)/2
 params=[torch.randn(n,r,dtype=torch.float64,requires_grad=True),torch.randn(r,dtype=torch.float64,requires_grad=True),torch.randn(n,r,dtype=torch.float64,requires_grad=True),torch.randn(r,dtype=torch.float64,requires_grad=True)];X,a,Y,b=params;C=(X*a)@X.T;D=(Y*b)@Y.T;reference=dense_error(A,B,C,D);fast=squared_error(A,B,inner(A,B,A,B),X,a,Y,b);g0=torch.autograd.grad(reference,params,retain_graph=True);g1=torch.autograd.grad(fast,params);err=float((reference.detach()-fast.detach()).abs()/reference.detach());grad=max(float((x-y).norm()/x.norm()) for x,y in zip(g0,g1));assert max(err,grad)<1e-11;records.append(dict(n=n,rank=r,value_relative_error=err,gradient_relative_error=grad))
result=dict(records=records,passed=True);(p/'QUARTIC_LOWRANK_METRIC_CHECK_V1.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
