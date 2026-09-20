"""Direct exact coefficient fitting of a shared quadratic-square hypothesis."""
import json,itertools,math,time
from pathlib import Path
import torch
P=Path(__file__).resolve().parent

def inner(a,b):return ((a@b).square()+2*(a.square()*b.square()).sum())/3

def main():
 torch.set_num_threads(1);torch.set_default_dtype(torch.float64);rows=[];out=P/'SHARED_SQUARE_FIT_V1.json';assert not out.exists();start=time.perf_counter()
 # Independent dense symmetric coefficients at d4 validate loss and gradient.
 torch.manual_seed(1719);a=torch.randn(4,requires_grad=True);b=torch.randn(4);A=torch.diag(a);B=torch.diag(b)
 def tensor(Q):return (torch.einsum('ij,kl->ijkl',Q,Q)+torch.einsum('ik,jl->ijkl',Q,Q)+torch.einsum('il,jk->ijkl',Q,Q))/3
 got=inner(a,b);ref=(tensor(A)*tensor(B)).sum();g1=torch.autograd.grad(got,a,retain_graph=True)[0];g2=torch.autograd.grad(ref,a)[0];check=max(float((got-ref).abs().detach()),float((g1-g2).abs().max()));assert check<1e-11
 for d,lr,seed in itertools.product([16,128,1152],[.01,.05],[0,1]):
  torch.manual_seed(seed);a=torch.nn.Parameter(torch.randn(d));teacher=torch.ones(d);den=inner(teacher,teacher);opt=torch.optim.Adam([a],lr=lr);best=float('inf');best_a=None;history=[]
  for step in range(1001):
   opt.zero_grad();loss=(inner(a,a)-2*inner(a,teacher)+den)/den;e=float(loss.detach())
   if e<best:best=e;best_a=a.detach().clone();best_step=step
   if step%200==0:history.append(dict(step=step,signed_squared_error=e))
   if step==1000:break
   loss.backward();opt.step()
   for group in opt.param_groups:group['lr']=lr*(.01+.99*.5*(1+math.cos(math.pi*(step+1)/1000)))
  row=dict(d=d,lr=lr,seed=seed,error=math.sqrt(max(0,best)),signed_squared_error=best,best_step=best_step,quadratic_coefficient_error_mod_sign=float(torch.minimum((best_a-teacher).norm(),(best_a+teacher).norm())/teacher.norm()),parameters=d,history=history);rows.append(row);print(d,lr,seed,row['error'],flush=True)
 out.write_text(json.dumps(dict(records=rows,validation_max_absolute_error=check,seconds=time.perf_counter()-start,scope='Known diagonal quadratic-square family, random coefficients, exact Frobenius fit. Coordinate support and reuse structure supplied; not generic discovery or native recovery.'),indent=2)+'\n')
if __name__=='__main__':main()
