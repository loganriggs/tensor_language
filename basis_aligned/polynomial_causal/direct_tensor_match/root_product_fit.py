"""Exact covariance-metric fitting of products of four bank coordinates."""
import math
import torch

def coefficients(a,b):
 i,j=torch.triu_indices(a.shape[1],a.shape[1],device=a.device)
 return torch.where(i==j,a[:,i]*b[:,j],a[:,i]*b[:,j]+a[:,j]*b[:,i])

def objective(Z,G,a,b,penalty=.001):
 H=coefficients(a,b);scale=((H@G)*H).sum(1).clamp_min(1e-24).pow(.25);u,v=a/scale[:,None],b/scale[:,None];H=coefficients(u,v);gram=H@G@H.T;cross=Z@G@H.T;writer=torch.linalg.solve(gram+penalty*torch.eye(len(a),dtype=a.dtype,device=a.device),cross.T).T;loss=((writer@gram)*writer).sum()-2*(writer*cross).sum()+penalty*writer.square().sum();return loss,(u,v,writer)

def fit(Z,G,width,optimizer,lr,seed,steps=500,initial=None):
 torch.manual_seed(seed);a=torch.nn.Parameter(torch.randn(width,4,dtype=Z.dtype)/2 if initial is None else initial[0].clone());b=torch.nn.Parameter(torch.randn_like(a)/2 if initial is None else initial[1].clone());opt=torch.optim.Adam([a,b],lr=lr) if optimizer=='adam' else torch.optim.Muon([a,b],lr=lr,weight_decay=0.,adjust_lr_fn='match_rms_adamw');norm=((Z@G)*Z).sum();best=float('inf')
 for step in range(steps+1):
  opt.zero_grad();loss,program=objective(Z,G,a,b);value=float(loss.detach());assert math.isfinite(value)
  if value<best:best=value;saved=tuple(t.detach().clone() for t in program);selected=step
  if step==steps:break
  (loss/norm).backward();opt.step()
  for group in opt.param_groups:group['lr']=lr*(.01+.99*.5*(1+math.cos(math.pi*(step+1)/steps)))
 u,v,c=saved;delta=Z-c@coefficients(u,v);energy=((delta@G)*delta).sum();return dict(relative_error=float((energy/norm).clamp_min(0).sqrt()),penalized_objective=best,selected_step=selected),saved

def check():
 import json
 from pathlib import Path
 torch.set_num_threads(1);torch.manual_seed(2059);dtype=torch.float64;raw=torch.randn(10,10,dtype=dtype);G=raw@raw.T+.1*torch.eye(10,dtype=dtype);Z=torch.randn(3,10,dtype=dtype);a=torch.randn(4,4,dtype=dtype,requires_grad=True);b=torch.randn(4,4,dtype=dtype,requires_grad=True);loss,(u,v,c)=objective(Z,G,a,b);grad=torch.autograd.grad(loss,a)[0];eps=1e-5;ap=a.detach().clone();am=ap.clone();ap[0,0]+=eps;am[0,0]-=eps;fd=(objective(Z,G,ap,b)[0]-objective(Z,G,am,b)[0])/(2*eps);err=float(((fd-grad[0,0]).abs()/grad[0,0].abs()).detach());x=torch.randn(19,4,dtype=dtype);i,j=torch.triu_indices(4,4);actual=(x@u.T)*(x@v.T);reference=(x[:,i]*x[:,j])@coefficients(u,v).T;replay=float(((actual-reference).norm()/actual.norm()).detach());assert err<1e-6 and replay<1e-12;out=dict(gradient_relative_error=err,product_coefficient_replay=replay,scope='Covariance-metric root product objective and analytic writer check; no native root fit yet.');(Path(__file__).resolve().parent/'ROOT_PRODUCT_FIT_CHECK_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
if __name__=='__main__':check()
