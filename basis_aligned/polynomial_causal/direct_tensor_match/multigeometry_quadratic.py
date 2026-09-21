"""Exact conditional quadratic fit across multiple coefficient metrics and responses."""
import torch
from learned_quadratic_factors import inner,dense
from profiled_learned_quadratic_factors import gram
from centered_product_response import features

class Objective:
 def __init__(self,c,a,b,geometries,x=None,v=None,response_weight=0.):
  self.c=c;self.components=[];self.x,self.v=x,v;self.response_weight=response_weight
  for weight,transform in geometries:
   ta,tb=a@transform,b@transform;energy=inner(c,ta,tb,c,ta,tb).detach();self.components.append((weight,transform,ta,tb,energy))
  self.truth=features(x,v,a,b)@c.T if response_weight else None
  self.response_energy=self.truth.square().sum().detach() if response_weight else None
 def readout(self,a,b):
  K=a.new_zeros((len(a),len(a)));rhs=self.c.new_zeros((len(self.c),len(a)))
  for weight,transform,ta,tb,energy in self.components:
   aa,bb=a@transform,b@transform;K=K+weight*gram(aa,bb,aa,bb)/energy;rhs=rhs+weight*(self.c@gram(ta,tb,aa,bb))/energy
  if self.response_weight:
   g=features(self.x,self.v,a,b);K=K+self.response_weight*(g.T@g)/self.response_energy;rhs=rhs+self.response_weight*(self.truth.T@g)/self.response_energy
  ridge=1e-10*K.diag().mean().detach();return torch.linalg.solve(K+ridge*torch.eye(len(a),dtype=a.dtype,device=a.device),rhs.T).T
 def losses(self,c,a,b):
  raw=[];total=c.new_zeros(())
  for weight,transform,ta,tb,energy in self.components:
   aa,bb=a@transform,b@transform;error=(energy+inner(c,aa,bb,c,aa,bb)-2*inner(c,aa,bb,self.c,ta,tb))/energy;raw.append(error);total=total+weight*error
  response=(features(self.x,self.v,a,b)@c.T-self.truth).square().sum()/self.response_energy if self.response_weight else c.new_zeros(())
  return total+self.response_weight*response,raw,response

def controls():
 torch.manual_seed(995);records=[]
 for d in range(3,8):
  rand=lambda *s:torch.randn(*s,dtype=torch.float64);c,a,b=rand(3,4),rand(4,d),rand(4,d);a2,b2=rand(3,d),rand(3,d);a2.requires_grad_();b2.requires_grad_();q,_=torch.linalg.qr(rand(d,d));transform=q@torch.diag(torch.logspace(-2,1,d,dtype=torch.float64))@q.T;obj=Objective(c,a,b,[(1.,torch.eye(d,dtype=torch.float64)),(.1,transform)],rand(19,d),rand(19,d),1.);design=[];targets=[]
  for weight,tr,ta,tb,energy in obj.components:
   design.append(dense(torch.eye(3,dtype=torch.float64),a2@tr,b2@tr).flatten(1).T*(weight/energy).sqrt());targets.append(dense(c,ta,tb).flatten(1).T*(weight/energy).sqrt())
  design.append(features(obj.x,obj.v,a2,b2)/obj.response_energy.sqrt());targets.append(obj.truth/obj.response_energy.sqrt());X,Y=torch.cat(design),torch.cat(targets);ridge=1e-10*X.square().sum(0).mean().detach();ref=torch.linalg.lstsq(torch.cat([X.detach(),ridge.sqrt()*torch.eye(3,dtype=X.dtype)]),torch.cat([Y,torch.zeros(3,3,dtype=Y.dtype)])).solution.T;C=obj.readout(a2,b2);solve=float(((X@(C-ref).T).norm()/Y.norm()).detach());implicit=obj.losses(C.detach(),a2,b2)[0];explicit=(X@C.detach().T-Y).square().sum();gi=torch.autograd.grad(implicit,(a2,b2),retain_graph=True);ge=torch.autograd.grad(explicit,(a2,b2));grad=max(float((g-h).norm()/h.norm()) for g,h in zip(gi,ge));records.append(dict(dimension=d,solve_error=solve,gradient_error=grad))
 assert max(max(r['solve_error'],r['gradient_error']) for r in records)<1e-9
 return records
if __name__=='__main__':
 import json
 from pathlib import Path
 rows=controls();Path(__file__).with_name('MULTIGEOMETRY_QUADRATIC_CONTROLS_V1.json').write_text(json.dumps(rows,indent=2)+'\n');print(rows)
