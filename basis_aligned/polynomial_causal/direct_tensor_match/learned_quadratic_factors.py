"""Joint product-direction/output fitting with exact symmetric coefficient loss."""
import math,torch
from centered_product_response import features

def inner(c,a,b,d,l,r):
 return .5*((c.T@d)*((a@l.T)*(b@r.T)+(a@r.T)*(b@l.T))).sum()

def dense(c,a,b):
 t=torch.einsum('vk,ki,kj->vij',c,a,b)
 return (t+t.transpose(1,2))/2

class Objective:
 def __init__(self,c,a,b,x=None,v=None,weight=1.):
  self.c,self.a,self.b=c,a,b;self.energy=inner(c,a,b,c,a,b).detach();self.x,self.v=x,v;self.weight=weight
  self.truth=features(x,v,a,b)@c.T if x is not None else None
  self.response_energy=self.truth.square().sum().detach() if x is not None else None
 def losses(self,c,a,b):
  coefficient=(self.energy+inner(c,a,b,c,a,b)-2*inner(c,a,b,self.c,self.a,self.b))/self.energy
  response=((features(self.x,self.v,a,b)@c.T-self.truth).square().sum()/self.response_energy) if self.x is not None else coefficient.new_zeros(())
  return coefficient,response

def fit(teacher,width,optimizer,lr,seed,steps,x=None,v=None,initial=None):
 torch.manual_seed(seed);c,a,b=teacher;rand=lambda *s:torch.randn(*s,dtype=c.dtype,device=c.device)
 init=initial or (rand(c.shape[0],width)/math.sqrt(c.shape[0]*width),rand(width,a.shape[1]),rand(width,a.shape[1]))
 params=[torch.nn.Parameter(t.clone()) for t in init];objective=Objective(c,a,b,x,v);opt=(torch.optim.Adam(params,lr=lr) if optimizer=='adam' else torch.optim.Muon(params,lr=lr,weight_decay=0.,adjust_lr_fn='match_rms_adamw'));history=[]
 def factors():return params[0],params[1]/params[1].norm(dim=1,keepdim=True).clamp_min(1e-12),params[2]/params[2].norm(dim=1,keepdim=True).clamp_min(1e-12)
 for step in range(steps):
  opt.zero_grad();co,re=objective.losses(*factors());loss=co+re;loss.backward();opt.step()
  if step%100==0:history.append(dict(step=step,coefficient_squared=float(co.detach()),response_squared=float(re.detach())))
  for group in opt.param_groups:group['lr']=lr*(.01+.99*.5*(1+math.cos(math.pi*(step+1)/steps)))
 with torch.no_grad():
  f=tuple(t.detach() for t in factors());co,re=objective.losses(*f)
 return dict(coefficient_error=max(0,float(co))**.5,response_error=max(0,float(re))**.5,signed_coefficient_squared=float(co),history=history),f

def controls():
 torch.manual_seed(944);c,a,b=torch.randn(3,4,dtype=torch.float64),torch.randn(4,5,dtype=torch.float64),torch.randn(4,5,dtype=torch.float64);params=[torch.randn_like(t,requires_grad=True) for t in (c,a,b)]
 implicit=inner(*params,*params)-2*inner(*params,c,a,b)+inner(c,a,b,c,a,b);explicit=(dense(*params)-dense(c,a,b)).square().sum();g=torch.autograd.grad(implicit,params);h=torch.autograd.grad(explicit,params);value=float(abs(implicit-explicit)/explicit);grad=max(float((x-y).norm()/y.norm()) for x,y in zip(g,h));assert max(value,grad)<1e-12
 return dict(dense_value_error=value,dense_gradient_error=grad)
