"""Profiled quadratic dictionary with all shared quartic pair products.
Data-informed functional loss, not coefficient Frobenius matching. Output ridge is
fixed on RMS-normalized product features; export folds scaling into the writer.
"""
import torch,math
from shared_quadratic_bank import normalize_bank

def features(x,u,v):
 m,k,d=u.shape;q=((x@u.flatten(0,1).T)*(x@v.flatten(0,1).T)).reshape(len(x),m,k).sum(2);i,j=torch.triu_indices(m,m,device=x.device);return q[:,i]*q[:,j]

def readout(phi,y,ridge_fraction=1e-6):
 scales=phi.square().mean(0).sqrt().clamp_min(1e-12);z=phi/scales;ridge=len(phi)*ridge_fraction
 c=torch.linalg.solve(z.T@z+ridge*torch.eye(z.shape[1],device=z.device,dtype=z.dtype),z.T@y)
 return c,scales,ridge

def fit(x,y,width,products_per_feature=4,steps=100,rate=.03,optimizer='adam',seed=2719,initial=None):
 torch.manual_seed(seed);d=x.shape[1];k=products_per_feature
 if initial is None:initial=(torch.randn(width,k,d,device=x.device,dtype=x.dtype)/math.sqrt(d),torch.randn(width,k,d,device=x.device,dtype=x.dtype)/math.sqrt(d))
 params=[torch.nn.Parameter(t.reshape(width*k,d).clone()) for t in initial]
 opt=torch.optim.Adam(params,lr=rate) if optimizer=='adam' else torch.optim.Muon(params,lr=rate,weight_decay=0.,adjust_lr_fn='match_rms_adamw')
 den=y.square().sum();best=None;history=[]
 for step in range(steps+1):
  u,v=normalize_bank(*(t.reshape(width,k,d) for t in params));phi=features(x,u,v)
  with torch.no_grad():c,_,ridge=readout(phi,y)
  scales=phi.square().mean(0).sqrt().clamp_min(1e-12);error=(phi/scales@c-y).square().sum()/den;loss=error+ridge*c.square().sum()/den;value=float(loss.detach())
  if best is None or value<best[0]:best=(value,step,float(error.detach().sqrt()),u.detach().clone(),v.detach().clone(),(c/scales[:,None]).T.detach().clone())
  if step%25==0 or step==steps:history.append(dict(step=step,objective=value,error=float(error.detach().sqrt())))
  if step==steps:break
  opt.zero_grad();loss.backward();opt.step()
  for group in opt.param_groups:group['lr']=rate*(.01+.99*.5*(1+math.cos(math.pi*(step+1)/steps)))
 loss,selected,error,u,v,c=best
 return dict(objective=loss,selected_step=selected,training_error=error,history=history),dict(U=u,V=v,C=c)

def evaluate(program,x):return features(x,program['U'],program['V'])@program['C'].T

def controls():
 rows=[]
 for d in range(3,8):
  torch.manual_seed(1720+d);x=torch.randn(41,d,dtype=torch.float64);y=torch.randn(41,5,dtype=torch.float64);u=torch.randn(3,2,d,dtype=torch.float64,requires_grad=True);v=torch.randn_like(u,requires_grad=True);phi=features(x,u,v);c,scales,ridge=readout(phi,y);z=phi/scales
  design=torch.cat([z.detach(),ridge**.5*torch.eye(z.shape[1],dtype=z.dtype)]);target=torch.cat([y,torch.zeros(z.shape[1],5,dtype=z.dtype)]);reference=torch.linalg.lstsq(design,target,driver='gelsd').solution;solve=float(((z@(c-reference)).norm()/y.norm()).detach())
  # Compare envelope derivative to differentiating the complete ridge solve.
  direct=((z@c-y).square().sum()+ridge*c.square().sum())/y.square().sum();profile=((z@c.detach()-y).square().sum()+ridge*c.detach().square().sum())/y.square().sum();gd=torch.autograd.grad(direct,(u,v),retain_graph=True);gp=torch.autograd.grad(profile,(u,v));gradient=max(float((a-b).norm()/a.norm()) for a,b in zip(gd,gp));rows.append(dict(dimension=d,solve_error=solve,gradient_error=gradient))
 assert max(max(r['solve_error'],r['gradient_error']) for r in rows)<1e-9
 return rows
