import itertools,json,math,time
from pathlib import Path
import numpy as np
import torch
from quadratic_square import square_inner,gaussian_square_inner,symmetric_bilinear
P=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(1);torch.set_default_dtype(torch.float64);torch.manual_seed(1724);rotation=torch.linalg.qr(torch.randn(8,8))[0];families={'rank4_balanced':([1,.8,-.6,-.4,0,0,0,0],2),'full_balanced':([1,.8,.6,.4,-1,-.8,-.6,-.4],4)};out=P/'SIGNED_WRITER_BASIN_V1.json';assert not out.exists();rows=[];start=time.perf_counter()
 raw=torch.randn(3,3,requires_grad=True);Q=(raw+raw.T)/2;z=torch.randn(3,3);S=(z+z.T)/2;n,w=np.polynomial.hermite.hermgauss(5);ix=torch.tensor(list(itertools.product(range(5),repeat=3)));x=torch.tensor(n)[ix]*2**.5;weights=torch.tensor(w/np.sqrt(np.pi))[ix].prod(1);ref=(weights*((x@Q)*x).sum(1).square()*((x@S)*x).sum(1).square()).sum();got=gaussian_square_inner(Q,S);g=torch.autograd.grad(got,raw,retain_graph=True)[0];h=torch.autograd.grad(ref,raw)[0];check=max(float(((got-ref).abs()/ref.abs()).detach()),float((g-h).norm()/h.norm()));assert check<1e-11
 for family,(spec,k) in families.items():
  teacher=(rotation*torch.tensor(spec))@rotation.T;teacher/=teacher.norm();fn=square_inner(teacher,teacher);gn=gaussian_square_inner(teacher,teacher)
  for mode,lr,seed in itertools.product(['fixed_positive','learned_positive','gaussian_projection'],[.01,.05],[0,1]):
   torch.manual_seed(seed);U=torch.nn.Parameter(torch.randn(8,k));V=torch.nn.Parameter(torch.randn(8,k));rawc=torch.nn.Parameter(torch.tensor(math.log(math.expm1(1.))));params=[U,V]+([rawc] if mode=='learned_positive' else []);opt=torch.optim.Adam(params,lr=lr);best=float('inf')
   for step in range(601):
    opt.zero_grad();q=symmetric_bilinear(U,V);q=q/q.norm();inner=gaussian_square_inner if mode=='gaussian_projection' else square_inner;den=gn if mode=='gaussian_projection' else fn;selfnorm=inner(q,q);cross=inner(q,teacher);c=cross/selfnorm if mode=='gaussian_projection' else (torch.nn.functional.softplus(rawc) if mode=='learned_positive' else q.new_tensor(1.));loss=(den+c*c*selfnorm-2*c*cross)/den;value=float(loss.detach())
    if value<best:best=value;bestq=q.detach().clone();bestc=c.detach().clone();beststep=step
    if step==600:break
    loss.backward();opt.step()
    for group in opt.param_groups:group['lr']=lr*(.01+.99*.5*(1+math.cos(math.pi*(step+1)/600)))
   errors={}
   for name,inner in [('frobenius',square_inner),('gaussian',gaussian_square_inner)]:
    den=inner(teacher,teacher);e=float((den+bestc.square()*inner(bestq,bestq)-2*bestc*inner(bestq,teacher))/den);errors[name]=math.sqrt(max(0,e))
   row=dict(family=family,width=k,mode=mode,lr=lr,seed=seed,errors=errors,selected_step=beststep,writer=float(bestc),quadratic_abs_cosine=float((bestq*teacher).sum().abs()));rows.append(row);print(family,mode,lr,seed,errors,flush=True)
  out.write_text(json.dumps(dict(records=rows,gaussian_quadrature_gradient_relative_check=check,seconds=time.perf_counter()-start,scope='Paired starts; positive writer justified only for planted square. Gaussian and coefficient objectives distinct; own-objective checkpoint selection.'),indent=2)+'\n')
if __name__=='__main__':main()
