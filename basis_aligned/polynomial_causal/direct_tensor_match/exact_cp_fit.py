"""Reusable exact symmetric quartic CP variable-projection fit."""
import math,time
import torch
from quartic_cp import cp_gram,cp_inner

def fit(tc,teacher,width,optname,seed,steps,initial=None):
 d=teacher[0].shape[1];torch.manual_seed(seed);raw=[torch.nn.Parameter(torch.randn(width,d,dtype=tc.dtype) if initial is None else a.clone()) for a in (teacher if initial is None else initial)];opt=torch.optim.Adam(raw,lr=.05) if optname=='adam' else torch.optim.Muon(raw,lr=.05,weight_decay=0.,adjust_lr_fn='match_rms_adamw');base=None;history=[];start=time.perf_counter();den=cp_inner(tc,teacher,tc,teacher)
 for step in range(steps):
  opt.zero_grad();f=[v/v.norm(dim=1,keepdim=True) for v in raw];K=cp_gram(f,f);cross=tc@cp_gram(teacher,f);C=torch.linalg.solve(K+1e-10*K.diag().mean()*torch.eye(width,dtype=tc.dtype),cross.T).T;loss=((C@K)*C).sum()-2*(cross*C).sum()
  if base is None:base=max(float(-loss.detach()),1e-30)
  if step%100==0:history.append(dict(step=step,explained_fraction=float((-loss/den).detach())))
  (loss/base).backward();opt.step()
  for group in opt.param_groups:group['lr']=.05*(.01+.99*.5*(1+math.cos(math.pi*(step+1)/steps)))
 with torch.no_grad():
  f=[v/v.norm(dim=1,keepdim=True) for v in raw];K=cp_gram(f,f);cross=tc@cp_gram(teacher,f);C=torch.linalg.solve(K+1e-10*K.diag().mean()*torch.eye(width,dtype=tc.dtype),cross.T).T;err=float((den+cp_inner(C,f,C,f)-2*cp_inner(tc,teacher,C,f))/den);alignment=cp_gram(teacher,f)/torch.outer(cp_gram(teacher,teacher).diag().sqrt(),K.diag().sqrt())
 return dict(relative_error=math.sqrt(max(0,err)),signed_squared_relative_error=err,seconds=time.perf_counter()-start,teacher_feature_correlations=alignment.tolist(),history=history),C.detach(),[v.detach() for v in f]
