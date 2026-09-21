"""Exact conditional second-read metric under normalizer-clamped source interchange.
Source and context Cartesian products are intervention definitions, not a model
of the naturally occurring joint distribution.
"""
import torch

def metric(qa,qb,true_a,true_b,carry,scale,alpha,beta,psi,cartesian):
 if cartesian:
  first=(carry[:,None]+.5*qa[None,:])/scale[:,None]-alpha
  true_first=(carry[:,None]+.5*true_a[None,:])/scale[:,None]-alpha
  target=true_first*(true_b[None,:]/scale[:,None]-beta)
  error=first*(qb[None,:]/scale[:,None]-beta)-target
  sensitivity=first/scale[:,None]
  weights=sensitivity.square().sum(0);cross=(sensitivity*error).sum(0)
 else:
  first=(carry+.5*qa)/scale-alpha;target=((carry+.5*true_a)/scale-alpha)*(true_b/scale-beta);error=first*(qb/scale-beta)-target;sensitivity=first/scale;weights=sensitivity.square();cross=sensitivity*error
 denominator=(target-target.mean()).square().sum()
 G=psi.T@(weights[:,None]*psi)/denominator;b=psi.T@cross/denominator;c=error.square().sum()/denominator
 return G,b,c,target,error,sensitivity

def controls():
 gen=torch.Generator().manual_seed(921);d=torch.float64;out=[]
 for n,k in ((5,3),(7,4)):
  qa,qb,ta,tb,c=[torch.randn(n,generator=gen,dtype=d) for _ in range(5)];s=torch.rand(n,generator=gen,dtype=d)+.2;psi=torch.randn(n,k,generator=gen,dtype=d);x=torch.randn(k,generator=gen,dtype=d)
  for cart in (False,True):
   G,b,c0,target,error,sensitivity=metric(qa,qb,ta,tb,c,s,.4,-.7,psi,cart)
   change=psi@x;direct=error+sensitivity*(change[None,:] if cart else change)
   predicted=x@G@x+2*b@x+c0;actual=direct.square().sum()/(target-target.mean()).square().sum();relative=float(abs(predicted-actual)/max(float(actual),1.));assert relative<1e-12
   out.append(dict(n=n,features=k,cartesian=cart,replay=relative))
 return out
