"""Exact constant/linear/quadratic coefficients around a fixed input center."""
import torch

def fold(teacher,mu):
 C,L,R,D,A,B=teacher;a= A@mu;b=B@mu;h0=D@(a*b);J=D@(a[:,None]*B+b[:,None]*A);left=L@h0;right=R@h0;readout=C@(left[:,None]*R+right[:,None]*L)
 constant=C@(left*right);linear=readout@J
 return constant,linear,torch.cat([readout@D,C],1),torch.cat([A,L@J],0),torch.cat([B,R@J],0)

def check():
 import json
 from pathlib import Path
 torch.set_num_threads(1);torch.set_default_dtype(torch.float64);torch.manual_seed(1848);teacher=[torch.randn(*shape) for shape in [(2,4),(4,3),(4,3),(3,5),(5,6),(5,6)]];mu=torch.randn(6);c,J,C,A,B=fold(teacher,mu)
 def f(x):
  co,L,R,D,a,b=teacher;h=D@((a@x)*(b@x));return co@((L@h)*(R@h))
 jac=torch.func.jacrev(f)(mu);hess=torch.func.hessian(f)(mu);tensor=.5*(torch.einsum('vk,ki,kj->vij',C,A,B)+torch.einsum('vk,kj,ki->vij',C,A,B));errors=dict(constant_relative=float((c-f(mu)).norm()/f(mu).norm()),linear_relative=float((J-jac).norm()/jac.norm()),quadratic_half_hessian_relative=float((tensor-hess/2).norm()/(hess/2).norm()));assert max(errors.values())<1e-12
 out=dict(checks=errors,scope='Exact weight-derived constant/linear/joint quadratic coefficients around fixed mu; autograd derivatives independently checked. Degree truncation usefulness and native factor price not established.')
 (Path(__file__).resolve().parent/'CENTERED_QUADRATIC_FOLD_CHECK_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
if __name__=='__main__':check()
