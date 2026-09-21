"""Feature-direction learning with profiled, per-root weighted ridge readouts."""
import math,torch
from empirical_quartic_dictionary import features
from shared_quadratic_bank import normalize_bank
from sensitive_root_readout import fit_roots

def objective(phi,target,weights,profiled=True,ridge_fraction=1e-6):
 w=weights/weights.mean(0,keepdim=True);scales=phi.square().mean(0).sqrt().clamp_min(1e-30);ys=target.square().mean(0).sqrt().clamp_min(1e-30)
 if profiled:
  with torch.no_grad():c=fit_roots(phi,target,w,ridge_fraction)*scales[None,:]/ys[:,None]
 else:c=fit_roots(phi,target,w,ridge_fraction)*scales[None,:]/ys[:,None]
 pred=(phi/scales)@c.T*ys;energy=(w*target.square()).sum(0);errors=(w*(pred-target).square()).sum(0);penalty=len(phi)*ridge_fraction*c.square().sum(1)*ys.square()
 return ((errors+penalty)/energy).mean(),c/scales[None,:]*ys[:,None]

def fit(x,target,weights,initial,steps=100,rate=.001):
 params=[torch.nn.Parameter(t.clone()) for t in initial];opt=torch.optim.Adam(params,lr=rate);best=None;history=[]
 for step in range(steps+1):
  u,v=normalize_bank(*params);phi=features(x,u,v);loss,c=objective(phi,target,weights);value=float(loss.detach())
  if best is None or value<best[0]:best=(value,step,u.detach().clone(),v.detach().clone(),c.detach().clone())
  if step%25==0 or step==steps:history.append(dict(step=step,objective=value))
  if step==steps:break
  opt.zero_grad();loss.backward();opt.step()
  for group in opt.param_groups:group['lr']=rate*(.01+.99*.5*(1+math.cos(math.pi*(step+1)/steps)))
 return dict(objective=best[0],selected_step=best[1],history=history),dict(U=best[2],V=best[3],coefficients=best[4])

def controls():
 torch.set_num_threads(2);dtype=torch.float64;rows=[]
 for seed,family in enumerate(['independent','shared_inputs','shared_outputs','squares','cancellation']):
  torch.manual_seed(719+seed);x=torch.randn(41,5,dtype=dtype);u=torch.randn(3,2,5,dtype=dtype);v=torch.randn_like(u);target=torch.randn(41,4,dtype=dtype);w=torch.rand_like(target)+.001
  if family=='shared_inputs':u[1]=u[0]
  if family=='shared_outputs':target[:,1]=target[:,0]
  if family=='squares':v=u.clone()
  if family=='cancellation':u[1]=u[0];v[1]=-v[0]
  u.requires_grad_();v.requires_grad_();phi=features(x,*normalize_bank(u,v));a,_=objective(phi,target,w,True);b,_=objective(phi,target,w,False);ga=torch.autograd.grad(a,(u,v),retain_graph=True);gb=torch.autograd.grad(b,(u,v));error=max(float((i-j).norm()/j.norm()) for i,j in zip(ga,gb));assert error<1e-8
  info,_=fit(x,target,w,(u.detach(),v.detach()),steps=2,rate=.0001);assert math.isfinite(info['objective']);rows.append(dict(family=family,profile_gradient_error=error,two_step_objective=info['objective']))
 return rows
if __name__=='__main__':
 import json
 from pathlib import Path
 rows=controls();Path(__file__).with_name('SENSITIVE_ROOT_FEATURE_CONTROLS_V1.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows))
