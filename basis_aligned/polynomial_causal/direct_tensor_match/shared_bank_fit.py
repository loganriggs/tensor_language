import math,time
import torch
from shared_quadratic_bank import normalize_bank,bank_gram,native_bank_cross

def fit_bank(teacher,bank,width,optimizer,lr,seed,steps,initial=None):
 d=teacher[-1].shape[1];torch.manual_seed(seed)
 if initial is None:initial=([torch.randn(width,d)/d**.5 for _ in range(bank)],[torch.randn(width,d)/d**.5 for _ in range(bank)])
 us=[torch.nn.Parameter(v.clone()) for v in initial[0]];vs=[torch.nn.Parameter(v.clone()) for v in initial[1]];params=us+vs;opt=torch.optim.Adam(params,lr=lr) if optimizer=='adam' else torch.optim.Muon(params,lr=lr,weight_decay=0.,adjust_lr_fn='match_rms_adamw');best=float('inf');base=None;start=time.perf_counter()
 for step in range(steps+1):
  opt.zero_grad();U,V=normalize_bank(torch.stack(us),torch.stack(vs));G=bank_gram(U,V);cross=native_bank_cross(teacher,U,V);C=torch.linalg.solve(G+1e-8*G.diag().mean()*torch.eye(len(G)),cross.T).T;loss=((C@G)*C).sum()-2*(cross*C).sum();e=float((1+loss).detach())
  if e<best:best=e;beststep=step;bu=U.detach().clone();bv=V.detach().clone();bc=C.detach().clone()
  if base is None:base=max(float(-loss.detach()),1e-20);initial_error=math.sqrt(max(0,e))
  if step==steps:break
  (loss/base).backward();opt.step()
  for group in opt.param_groups:group['lr']=lr*(.01+.99*.5*(1+math.cos(math.pi*(step+1)/steps)))
 return dict(error=math.sqrt(max(0,best)),signed_squared_error=best,selected_step=beststep,initial_error=initial_error,seconds=time.perf_counter()-start),bu,bv,bc
