"""Independent dense24-permutation and autograd checks of quartic contraction."""
from pathlib import Path
import itertools,json,torch
from quartic_pair_metric import inner,squared_error,numerator_forms
p=Path(__file__).resolve().parent;torch.set_num_threads(2);torch.manual_seed(2158)
def dense(A,B):
 raw=torch.einsum('ij,kl->ijkl',A,B);return sum(raw.permute(order) for order in itertools.permutations(range(4)))/24
checks=[]
for d in [2,3,5]:
 raw=[torch.randn(d,d,dtype=torch.float64,requires_grad=True) for _ in range(4)];mats=[(v+v.T)/2 for v in raw];A,B,C,D=mats
 exact=(dense(A,B)-dense(C,D)).square().sum();implicit=squared_error(A,B,C,D);g0=torch.autograd.grad(exact,raw,retain_graph=True);g1=torch.autograd.grad(implicit,raw);value=float((exact.detach()-implicit.detach()).abs()/exact.detach());gradient=max(float((a-b).norm()/a.norm()) for a,b in zip(g0,g1));assert max(value,gradient)<1e-12;checks.append(dict(dimension=d,value_relative_error=value,gradient_relative_error=gradient))
Qa=torch.randn(4,4,dtype=torch.float64);Qa=(Qa+Qa.T)/2;Qb=torch.randn(4,4,dtype=torch.float64);Qb=(Qb+Qb.T)/2;A,B=numerator_forms(Qa,Qb,.7,-.4);x=torch.randn(39,7,dtype=torch.float64);x[:,-1]=1;z=x[:,:4];t=x[:,4];s=x[:,5];direct=(t-.5*((z@Qa)*z).sum(1)-.7*s)*(((z@Qb)*z).sum(1)+.4*s);pred=((x@A)*x).sum(1)*((x@B)*x).sum(1);replay=float((pred-direct).norm()/direct.norm());assert replay<1e-12
# Known repeated-input identity: (x1^2+x2^2)(x1^2-x2^2)=x1^4-x2^4.
A=torch.eye(2,dtype=torch.float64);B=torch.diag(torch.tensor([1.,-1.],dtype=torch.float64));T=dense(A,B);expected=torch.zeros_like(T);expected[0,0,0,0]=1;expected[1,1,1,1]=-1;assert (T-expected).norm()<1e-12
out=dict(predictions=dict(pred_a_dense=all(r['value_relative_error']<1e-12 for r in checks),pred_b_gradients=all(r['gradient_relative_error']<1e-12 for r in checks),pred_c_numerator=replay<1e-12),checks=checks,numerator_replay=replay,repeated_input_identity='PASS',scope='Exact coefficient metric primitive and gradients verified on small dense tensors, not a trained-model fit. Numerator uses independent augmented t/s/u coordinates; u=1 and RMS constraints required on execution. Functional data metric is different.');(p/'QUARTIC_PAIR_METRIC_CHECK_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
