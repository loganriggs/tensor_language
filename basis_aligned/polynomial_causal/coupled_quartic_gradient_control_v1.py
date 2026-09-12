"""Exact variable-projection objective: gradient controls for subsequent reader optimization."""
from pathlib import Path
import torch,json
from coupled_quartic_writer_v1 import gram,target_cross
P=Path(__file__).resolve().parent
torch.set_default_dtype(torch.float64);torch.set_num_threads(2);torch.manual_seed(91602)
b=torch.randn(3,5,2,requires_grad=True);nu=torch.randn(3,2,requires_grad=True)
target=torch.randn(2,5,5,5,5)
# Symmetrize by averaging all permutations; nonsymmetric target would invalidate target_cross.
import itertools
target=torch.stack([target.permute(0,*[i+1 for i in perm]) for perm in itertools.permutations(range(4))]).mean(0)
def oracle(x):return torch.einsum('oabcd,na,nb,nc,nd->no',target,x[:,0],x[:,1],x[:,2],x[:,3])
def objective(b,nu,envelope=False):
 k=gram(b,nu);c=target_cross(b,nu,oracle);a=torch.linalg.solve(k,c)
 if envelope:a=a.detach()
 return (a*(k@a)).sum()-2*(a*c).sum()
y=objective(b,nu);g=torch.autograd.grad(y,(b,nu));ge=torch.autograd.grad(objective(b,nu,True),(b,nu))
fd=[]
for index,parameter in enumerate((b,nu)):
 direction=torch.randn_like(parameter);direction=direction/direction.norm();eps=1e-5
 args1=[b.detach(),nu.detach()];args2=[b.detach(),nu.detach()]
 args1[index]=args1[index]+eps*direction;args2[index]=args2[index]-eps*direction
 numerical=(objective(*args1)-objective(*args2))/(2*eps);analytic=(g[index]*direction).sum()
 fd.append(float((numerical-analytic).abs()/torch.maximum(numerical.abs(),analytic.abs()).clamp_min(1e-10)))
equiv=[float((a-c).norm()/a.norm()) for a,c in zip(g,ge)]
result=dict(pred_a=max(fd)<=1e-5,pred_b=max(equiv)<=1e-8,finite_difference_relative_errors=fd,envelope_relative_errors=equiv,gradient_norms=[float(a.norm()) for a in g])
p=P/'COUPLED_QUARTIC_GRADIENT_V1_CONTROL.json';assert not p.exists();p.write_text(json.dumps(result,indent=2)+'\n');print(result);assert result['pred_a'] and result['pred_b']
