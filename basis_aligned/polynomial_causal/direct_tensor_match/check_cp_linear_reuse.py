import itertools,json
import numpy as np
import torch
from cp_linear_reuse import compile_program,evaluate
from noncentral_gaussian_cp import affine_moment

def controls():
 torch.manual_seed(26000);dt=torch.float64;rows=[];x=torch.randn(17,4,dtype=dt);C=torch.randn(2,3,dtype=dt);base=torch.randn(3,4,dtype=dt)
 for kind in ['shared','squares','fourthpowers','signs','permutations']:
  fs=[base.clone() for _ in range(4)]
  if kind=='squares':fs[2]=fs[3]=torch.randn_like(base)
  if kind=='signs':fs[1]=-fs[1];fs[3]=-fs[3]
  if kind=='permutations':fs=[base.roll(i,0) for i in range(4)]
  p=compile_program(fs,C);ref=torch.stack([x@f.T for f in fs]).prod(0)@C.T;err=float((evaluate(p,x)-ref).norm()/ref.norm());assert err<1e-12
  rows.append(dict(kind=kind,replay=err,readers=len(p['bank'])))
 a,w=np.polynomial.hermite.hermgauss(5);z=torch.tensor(list(itertools.product(a*2**.5,repeat=3)),dtype=dt);weights=torch.tensor([a*b*c for a,b,c in itertools.product(w/np.pi**.5,repeat=3)],dtype=dt)
 fs=[torch.randn(1,3,dtype=dt) for _ in range(5)];bs=[torch.randn(1,dtype=dt) for _ in range(5)]
 old=fs[:4];new=[fs[4]]+fs[1:4];ob=bs[:4];nb=[bs[4]]+bs[1:4]
 oo=affine_moment(old+old,ob+ob);on=affine_moment(old+new,ob+nb);nn=affine_moment(new+new,nb+nb);beta=on/nn
 u=torch.stack([z@f.T+b for f,b in zip(old,ob)]).prod(0)[:,0];v=torch.stack([z@f.T+b for f,b in zip(new,nb)]).prod(0)[:,0];refbeta=(weights*u*v).sum()/(weights*v.square()).sum();referror=(weights*(u-beta*v).square()).sum();error=oo-2*beta*on+beta.square()*nn;assert abs(float(beta-refbeta))<1e-10 and abs(float(error-referror))<1e-9
 return dict(exact=rows,beta_error=abs(float(beta-refbeta)),edit_energy_error=abs(float(error-referror)))
if __name__=='__main__':print(json.dumps(controls(),indent=2))
