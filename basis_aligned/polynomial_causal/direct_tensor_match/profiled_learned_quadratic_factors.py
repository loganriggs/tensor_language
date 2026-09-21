"""Variable-projection control: exact conditional outputs, learned input directions."""
import math,torch
from learned_quadratic_factors import Objective
from centered_product_response import features

def gram(a,b,l,r):return .5*((a@l.T)*(b@r.T)+(a@r.T)*(b@l.T))

def readout(objective,a,b):
 K=gram(a,b,a,b)/objective.energy;rhs=objective.c@gram(objective.a,objective.b,a,b)/objective.energy
 if objective.x is not None:
  g=features(objective.x,objective.v,a,b);K=K+g.T@g/objective.response_energy;rhs=rhs+objective.truth.T@g/objective.response_energy
 ridge=1e-10*K.diag().mean().detach();C=torch.linalg.solve(K+ridge*torch.eye(len(a),dtype=a.dtype,device=a.device),rhs.T).T
 return C

def fit(teacher,width,optimizer,lr,seed,steps,x=None,v=None,initial=None):
 torch.manual_seed(seed);c,a,b=teacher;rand=lambda *s:torch.randn(*s,dtype=c.dtype,device=c.device);init=(initial[1:] if initial is not None else (rand(width,a.shape[1]),rand(width,a.shape[1])));params=[torch.nn.Parameter(t.clone()) for t in init];objective=Objective(c,a,b,x,v);opt=(torch.optim.Adam(params,lr=lr) if optimizer=='adam' else torch.optim.Muon(params,lr=lr,weight_decay=0.,adjust_lr_fn='match_rms_adamw'));history=[]
 def directions():return [t/t.norm(dim=1,keepdim=True).clamp_min(1e-12) for t in params]
 for step in range(steps):
  opt.zero_grad();aa,bb=directions()
  with torch.no_grad():C=readout(objective,aa,bb)
  co,re=objective.losses(C,aa,bb);(co+re).backward();opt.step()
  if step%100==0:history.append(dict(step=step,coefficient_squared=float(co.detach()),response_squared=float(re.detach())))
  for group in opt.param_groups:group['lr']=lr*(.01+.99*.5*(1+math.cos(math.pi*(step+1)/steps)))
 with torch.no_grad():
  aa,bb=directions();C=readout(objective,aa,bb);co,re=objective.losses(C,aa,bb)
 return dict(coefficient_error=max(0,float(co))**.5,response_error=max(0,float(re))**.5,signed_coefficient_squared=float(co),history=history),(C.detach(),aa.detach(),bb.detach())
